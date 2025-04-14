import sys

import torch
from torchcfm.optimal_transport import wasserstein
from hyperpyyaml import load_hyperpyyaml
from data_prepare import prepare_data

import speechbrain as sb
import whisper
from torchdyn.core import NeuralODE

from utils_wspr import transcribe_embeddings


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def compute_forward(self, batch, stage):
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

        emb_noisy = batch.emb_noisy.data[:, 0, :, :].to(self.device)
        emb_clean = batch.emb_clean.data[:, 0, :, :].to(self.device)

        t, xt, ut = self.hparams.flow_matcher.sample_location_and_conditional_flow(
            emb_noisy, emb_clean
        )

        ut_pred, _ = self.modules.model(xt)

        # Return a dictionary so we don't have to remember the order
        return {"ut": ut, "ut_pred": ut_pred}

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        cfm_loss = torch.mean((predictions["ut_pred"] - predictions["ut"]) ** 2)

        self.loss_metric.append(
            ids=batch.id,
            predictions=predictions["ut_pred"],
            targets=predictions["ut"],
        )

        # Some evaluations are slower, and we only want to perform them
        # on the validation set.
        if stage != sb.Stage.TRAIN and self.epoch % self.hparams.validate_every == 0:
            txt_hyp, p1_pred_batch = self.inference(batch)
            self.p1_pred.append(p1_pred_batch)
            self.q1.append(batch.emb_clean)

            self.wer_stats.append(
                ids=batch.id,
                predict=[self.hparams.text_normalizer(txt) for txt in txt_hyp],
                target=[self.hparams.text_normalizer(txt) for txt in batch.txt_label],
            )
            self.cer_stats.append(
                ids=batch.id,
                predict=[self.hparams.text_normalizer(txt) for txt in txt_hyp],
                target=[self.hparams.text_normalizer(txt) for txt in batch.txt_label],
            )

        return cfm_loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=lambda predictions, targets: torch.mean(
                (predictions - targets) ** 2
            )[None]
        )

        self.epoch = epoch

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN and epoch % self.hparams.validate_every == 0:
            self.p1_pred = []
            self.q1 = []
            self.wer_stats = sb.utils.metric_stats.ErrorRateStats()
            self.cer_stats = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)
            self.whisper_model = whisper.load_model(
                self.hparams.whisper_model,
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            if epoch % self.hparams.validate_every == 0:
                ot_cost = wasserstein(
                    torch.concatenate(self.p1_pred), torch.concatenate(self.q1)
                )
                print(ot_cost)
                stats = {
                    "loss": stage_loss,
                    "wer": self.wer_stats.summarize("WER"),
                    "cer": self.cer_stats.summarize("WER"),
                }

                self.whisper_model = self.whisper_model.cpu()
                del self.whisper_model
                del self.p1_pred
                del self.q1

            else:
                stats = {
                    "loss": stage_loss,
                }

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best STOI score.
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["wer"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    @torch.no_grad
    def inference(self, batch):
        emb_noisy = batch.emb_noisy.data[:, 0, :, :].to(self.device)

        def vt(t, x, args):
            model_out, _ = self.modules.model(x)
            return model_out

        node = NeuralODE(vt, solver="euler", sensitivity="adjoint")
        traj = node.trajectory(emb_noisy, t_span=torch.linspace(0, 1, 50))

        results = []
        for emb_clean_pred in traj[-1]:
            wspr_result = transcribe_embeddings(
                self.whisper_model, emb_clean_pred, language="en"
            )

            results.append(wspr_result["text"])

        return results, traj[-1]


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

    # wer_stats = hparams["wer_stats"]
    # wer_stats.append("hi", predict="hello", target="hello")
    #
    # normalizer = hparams["text_normalizer"]
    #
    # def get_txt_list(split, key, normalizer):
    #     with open(hparams[f"{split}_annotation"], "r") as f:
    #         json_test = json.load(f)
    #     if key == "utt_id":
    #         return json_test.keys()
    #     else:
    #         txt = [normalizer(utt[key]) for utt in json_test.values()]
    #         return txt
    #
    # utt_ids = get_txt_list("test", "utt_id", normalizer)
    # txt_label = get_txt_list("test", "txt_label", normalizer)
    # txt_clean = get_txt_list("test", "txt_clean", normalizer)
    # txt_noisy = get_txt_list("test", "txt_noisy", normalizer)
    #
    # wer_stats.clear()
    # wer_stats.append(
    #     ids=utt_ids,
    #     predict=txt_clean,
    #     target=txt_label,
    # )
    # print(wer_stats.summarize())
    # wer_stats.clear()
    # wer_stats.append(
    #     ids=utt_ids,
    #     predict=txt_noisy,
    #     target=txt_label,
    # )
    # print(wer_stats.summarize())
    # wer_stats.clear()
    # wer_stats.append(
    #     ids=utt_ids,
    #     predict=txt_noisy,
    #     target=txt_clean,
    # )
    # print(wer_stats.summarize())
    #
    # utt_ids = get_txt_list("valid", "utt_id", normalizer)
    # txt_label = get_txt_list("valid", "txt_label", normalizer)
    # txt_clean = get_txt_list("valid", "txt_clean", normalizer)
    # txt_noisy = get_txt_list("valid", "txt_noisy", normalizer)
    #
    # wer_stats.clear()
    # wer_stats.append(
    #     ids=utt_ids,
    #     predict=txt_clean,
    #     target=txt_label,
    # )
    # print(wer_stats.summarize())
    # wer_stats.clear()
    # wer_stats.append(
    #     ids=utt_ids,
    #     predict=txt_noisy,
    #     target=txt_label,
    # )
    # print(wer_stats.summarize())
    # wer_stats.clear()
    # wer_stats.append(
    #     ids=utt_ids,
    #     predict=txt_noisy,
    #     target=txt_clean,
    # )
    # print(wer_stats.summarize())

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key="stoi",
        test_loader_kwargs=hparams["dataloader_options"],
    )
