import sys

import torch
from hyperpyyaml import load_hyperpyyaml
from data_prepare import prepare_data

import speechbrain as sb
import whisper

from utils_wspr import transcribe_embeddings


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def transcribe_dataset(self, dataset):
        """Apply masking to convert from noisy waveforms to enhanced signals.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            A dictionary with keys {"spec", "wav"} with predicted features.
        """
        # We first move the batch to the appropriate device, and
        # compute the features necessary for masking.

        from tqdm import tqdm

        wer_stats_noisy = sb.utils.metric_stats.ErrorRateStats()
        cer_stats_noisy = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)
        wer_stats_clean = sb.utils.metric_stats.ErrorRateStats()
        cer_stats_clean = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)

        whisper_model = whisper.load_model(
            self.hparams.whisper_model,
        )

        p1_loss_metric = sb.utils.metric_stats.MetricStats(
            metric=lambda predictions, targets: torch.mean(
                (predictions - targets) ** 2
            )[None]
        )

        with torch.no_grad():
            for batch in tqdm(dataset, dynamic_ncols=True):
                emb_noisy = batch["emb_noisy"].to(self.device)
                emb_clean = batch["emb_clean"].to(self.device)

                wspr_result = transcribe_embeddings(
                    whisper_model, emb_noisy, language="en"
                )["text"]
                print(wspr_result)
                print(self.hparams.text_normalizer(wspr_result))

                wer_stats_noisy.append(
                    ids=[batch["id"]],
                    predict=[self.hparams.text_normalizer(wspr_result).split(" ")],
                    target=[
                        self.hparams.text_normalizer(batch["txt_label"]).split(" ")
                    ],
                )
                cer_stats_noisy.append(
                    ids=[batch["id"]],
                    predict=[self.hparams.text_normalizer(wspr_result)],
                    target=[self.hparams.text_normalizer(batch["txt_label"])],
                )

                wspr_result = transcribe_embeddings(
                    whisper_model, emb_clean, language="en"
                )["text"]
                print(wspr_result)
                print(batch["txt_label"])
                wer_stats_clean.append(
                    ids=[batch["id"]],
                    predict=[self.hparams.text_normalizer(wspr_result).split(" ")],
                    target=[
                        self.hparams.text_normalizer(batch["txt_label"]).split(" ")
                    ],
                )
                cer_stats_clean.append(
                    ids=[batch["id"]],
                    predict=[self.hparams.text_normalizer(wspr_result).split(" ")],
                    target=[
                        self.hparams.text_normalizer(batch["txt_label"]).split(" ")
                    ],
                )

                p1_loss_metric.append(
                    ids=[batch["id"]], predictions=emb_noisy, targets=emb_clean
                )

        import os

        with open(os.path.join(self.hparams.save_folder, "p1_loss.txt"), "w") as f:
            p1_loss_metric.write_stats(f)

        with open(
            os.path.join(self.hparams.save_folder, "wer_stats_noisy.txt"), "w"
        ) as f:
            wer_stats_noisy.write_stats(f)

        with open(
            os.path.join(self.hparams.save_folder, "cer_stats_noisy.txt"), "w"
        ) as f:
            cer_stats_noisy.write_stats(f)

        with open(
            os.path.join(self.hparams.save_folder, "wer_stats_clean.txt"), "w"
        ) as f:
            wer_stats_clean.write_stats(f)

        with open(
            os.path.join(self.hparams.save_folder, "cer_stats_clean.txt"), "w"
        ) as f:
            cer_stats_clean.write_stats(f)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json` and `valid.json` manifest files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline. Adds noise, reverb, and babble on-the-fly.
    # Of course for a real enhancement dataset, you'd want a fixed valid set.
    @sb.utils.data_pipeline.takes(
        "path_emb_noisy", "path_emb_clean", "txt_noisy", "txt_clean", "txt_label"
    )
    @sb.utils.data_pipeline.provides(
        "emb_noisy", "emb_clean", "txt_noisy", "txt_clean", "txt_label"
    )
    def audio_pipeline(path_emb_noisy, path_emb_clean, txt_noisy, txt_clean, txt_label):
        emb_noisy = torch.load(path_emb_noisy)
        emb_clean = torch.load(path_emb_clean)
        return emb_noisy, emb_clean, txt_noisy, txt_clean, txt_label

    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=[
                "id",
                "emb_noisy",
                "emb_clean",
                "txt_noisy",
                "txt_clean",
                "txt_label",
            ],
        )
    return datasets


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    import os

    if not os.path.isdir(hparams["save_folder"]):
        os.mkdir(hparams["save_folder"])

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_data,
            kwargs={
                "data_folder": hparams["data_folder"],
                "whisper_model": hparams["whisper_model"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
            },
        )

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    se_brain = SEBrain(
        hparams=hparams,
        run_opts=run_opts,
    )

    se_brain.transcribe_dataset(datasets[hparams["split"]])
