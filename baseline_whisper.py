import sys
import os

import torch
from hyperpyyaml import load_hyperpyyaml
from data_prepare import prepare_data

import speechbrain as sb


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def transcribe_dataset(self, dataset, split):
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

        p1_loss_metric = sb.utils.metric_stats.MetricStats(
            metric=lambda predictions, targets: torch.mean(
                (predictions - targets) ** 2
            )[None]
        )

        self.hparams.test_search.model.to(self.device)

        with torch.no_grad():
            for batch in tqdm(dataset, dynamic_ncols=True):
                emb_noisy = batch["emb_noisy"][None, :, :].to(self.device)
                emb_clean = batch["emb_clean"][None, :, :].to(self.device)
                wav_lens = torch.tensor([1]).to(self.device)

                # eval noisy
                hyps, _, _, _ = self.hparams.test_search(emb_noisy.detach(), wav_lens)

                tokens = [batch["tokens"]]

                # Decode token terms to words
                predicted_words = [
                    self.tokenizer.decode(t, skip_special_tokens=True).strip()
                    for t in hyps
                ]

                # Convert indices to words
                target_words = self.tokenizer.batch_decode(
                    tokens, skip_special_tokens=True
                )
                if hasattr(self.hparams, "normalized_transcripts"):
                    if hasattr(self.tokenizer, "normalize"):
                        normalized_fn = self.tokenizer.normalize
                    else:
                        normalized_fn = self.tokenizer._normalize

                    predicted_words = [
                        normalized_fn(text).split(" ") for text in predicted_words
                    ]

                    target_words = [
                        normalized_fn(text).split(" ") for text in target_words
                    ]
                else:
                    predicted_words = [text.split(" ") for text in predicted_words]
                    target_words = [text.split(" ") for text in target_words]

                wer_stats_noisy.append([batch["id"]], predicted_words, target_words)
                cer_stats_noisy.append([batch["id"]], predicted_words, target_words)

                # eval clean
                hyps, _, _, _ = self.hparams.test_search(emb_clean.detach(), wav_lens)

                tokens = [batch["tokens"]]

                # Decode token terms to words
                predicted_words = [
                    self.tokenizer.decode(t, skip_special_tokens=True).strip()
                    for t in hyps
                ]

                # Convert indices to words
                target_words = self.tokenizer.batch_decode(
                    tokens, skip_special_tokens=True
                )
                if hasattr(self.hparams, "normalized_transcripts"):
                    if hasattr(self.tokenizer, "normalize"):
                        normalized_fn = self.tokenizer.normalize
                    else:
                        normalized_fn = self.tokenizer._normalize

                    predicted_words = [
                        normalized_fn(text).split(" ") for text in predicted_words
                    ]

                    target_words = [
                        normalized_fn(text).split(" ") for text in target_words
                    ]
                else:
                    predicted_words = [text.split(" ") for text in predicted_words]
                    target_words = [text.split(" ") for text in target_words]

                wer_stats_clean.append([batch["id"]], predicted_words, target_words)
                cer_stats_clean.append([batch["id"]], predicted_words, target_words)

                p1_loss_metric.append(
                    ids=[batch["id"]], predictions=emb_noisy, targets=emb_clean
                )

        save_root = os.path.join(self.hparams.save_folder, split)
        os.mkdir(save_root)

        with open(os.path.join(save_root, "p1_loss.txt"), "w") as f:
            p1_loss_metric.write_stats(f)

        with open(os.path.join(save_root, "wer_stats_noisy.txt"), "w") as f:
            wer_stats_noisy.write_stats(f)

        with open(os.path.join(save_root, "cer_stats_noisy.txt"), "w") as f:
            cer_stats_noisy.write_stats(f)

        with open(os.path.join(save_root, "wer_stats_clean.txt"), "w") as f:
            wer_stats_clean.write_stats(f)

        with open(os.path.join(save_root, "cer_stats_clean.txt"), "w") as f:
            cer_stats_clean.write_stats(f)


def dataio_prep(hparams, tokenizer):
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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("path_emb_noisy", "path_emb_clean", "wav_len")
    @sb.utils.data_pipeline.provides("emb_noisy", "emb_clean", "wav_len")
    def audio_pipeline(path_emb_noisy, path_emb_clean, wav_len):
        emb_noisy = torch.load(path_emb_noisy)
        emb_clean = torch.load(path_emb_clean)
        return emb_noisy, emb_clean, wav_len

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        if "normalized_transcripts" in hparams and hparams["normalized_transcripts"]:
            wrd = tokenizer.normalize(wrd)
        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

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
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=[
                "id",
                "emb_noisy",
                "emb_clean",
                "wav_len",
                "tokens_list",
                "tokens_bos",
                "tokens_eos",
                "tokens",
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
                "model_id": hparams["whisper_hub"].split("/")[-1],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
            },
        )

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(hparams, tokenizer)

    # Initialize the Brain object to prepare for mask training.
    se_brain = SEBrain(
        hparams=hparams,
        run_opts=run_opts,
    )

    se_brain.tokenizer = tokenizer

    se_brain.transcribe_dataset(datasets["valid"], "valid")
    se_brain.transcribe_dataset(datasets["test"], "test")
