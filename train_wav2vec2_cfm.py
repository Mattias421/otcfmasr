import sys
import torch.nn.functional as F

import torch
from hyperpyyaml import load_hyperpyyaml
from data_prepare import prepare_data

import speechbrain as sb
from speechbrain.utils import hpopt as hp
from torchdyn.core import NeuralODE
from otcfmasr.decoding import GreedyCTCDecoder
import torchaudio
from matcha.utils.model import sequence_mask, fix_len_compatibility


def loss_fn(ut_pred, ut, mask):
    loss = F.mse_loss(ut_pred, ut, reduction="sum") / (torch.sum(mask) * ut.shape[1])
    return loss


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    def pad_and_transpose(self, batch):
        emb_noisy = batch.emb_noisy.data.to(self.device)
        emb_clean = batch.emb_clean.data.to(self.device)

        B, L, C = emb_noisy.shape

        max_length = fix_len_compatibility(
            emb_noisy.shape[1], num_downsamplings_in_unet=2
        )
        lengths = batch.emb_noisy.lengths * emb_noisy.shape[1]
        mask = sequence_mask(lengths, max_length=max_length)
        mask = mask[:, None, :].to(self.device)

        emb_noisy_input = torch.zeros((B, C, max_length)).to(self.device)
        emb_noisy_input[:, :, :L] = emb_noisy.transpose(1, 2)

        emb_clean_input = torch.zeros((B, C, max_length)).to(self.device)
        emb_clean_input[:, :, :L] = emb_clean.transpose(1, 2)

        return emb_clean_input, emb_noisy_input, mask

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

        emb_noisy, emb_clean, mask = self.pad_and_transpose(batch)

        t, xt, ut = self.hparams.flow_matcher.sample_location_and_conditional_flow(
            emb_noisy, emb_clean
        )

        xt = xt * mask
        ut = ut * mask

        ut_pred = self.modules.model(xt, mask, emb_noisy, t)

        if (
            stage == sb.Stage.VALID and self.epoch % self.hparams.validate_every == 0
        ) or stage == sb.Stage.TEST:
            hyps, p1_pred = self.inference(batch, stage)
            return {
                "ut": ut,
                "ut_pred": ut_pred,
                "hyps": hyps,
                "p1_pred": p1_pred,
                "mask": mask,
            }
        else:
            return {"ut": ut, "ut_pred": ut_pred, "mask": mask}

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

        cfm_loss = loss_fn(
            predictions["ut_pred"], predictions["ut"], predictions["mask"]
        )

        self.loss_metric.append(
            ids=batch.id,
            ut_pred=predictions["ut_pred"],
            ut=predictions["ut"],
            mask=predictions["mask"],
        )

        # Some evaluations are slower, and we only want to perform them
        # on the validation set.
        if (stage != sb.Stage.TRAIN) and (
            self.epoch % self.hparams.validate_every == 0
        ):
            txt_hyp, p1_pred_batch = self.inference(batch)

            hyps = predictions["hyps"]

            import re

            pattern_unwanted = r"[^{}\s]".format(
                "".join(re.escape(c) for c in self.labels)
            )
            target_words = [
                re.sub(pattern_unwanted, "", wrd.upper()).split(" ")
                for wrd in batch.wrd
            ]
            target_words = target_words

            # eval noisy

            # Convert indices to words
            predicted_words = [text.split("|") for text in hyps]

            self.wer_stats.append(batch.id, predicted_words, target_words)
            self.cer_stats.append(batch.id, predicted_words, target_words)

            emb_noisy, emb_clean, mask = self.pad_and_transpose(batch)

            self.p1_loss_metric.append(
                ids=batch.id, ut_pred=predictions["p1_pred"], ut=emb_clean, mask=mask
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
            metric=lambda ut_pred, ut, mask: loss_fn(ut_pred, ut, mask)[None]
        )

        self.epoch = epoch

        # Set up evaluation-only statistics trackers
        if stage == sb.Stage.VALID:
            if epoch % self.hparams.validate_every == 0:
                self.asr_model.aux.to(self.device)
                self.wer_stats = sb.utils.metric_stats.ErrorRateStats()
                self.cer_stats = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)

                self.p1_loss_metric = sb.utils.metric_stats.MetricStats(
                    metric=lambda ut_pred, ut, mask: loss_fn(ut_pred, ut, mask)[None]
                )

        elif stage == sb.Stage.TEST:
            self.asr_model.aux.to(self.device)
            self.epoch = self.hparams.validate_every
            self.wer_stats = sb.utils.metric_stats.ErrorRateStats()
            self.cer_stats = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)

            self.p1_loss_metric = sb.utils.metric_stats.MetricStats(
                metric=lambda ut_pred, ut, mask: loss_fn(ut_pred, ut, mask)[None]
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
            if stage == sb.Stage.VALID:
                if epoch % self.hparams.validate_every == 0:
                    stats = {
                        "loss": stage_loss,
                        "wer": self.wer_stats.summarize("WER"),
                        "cer": self.cer_stats.summarize("WER"),
                        "p1_loss": self.p1_loss_metric.summarize("average"),
                    }
                    save_root = os.path.join(
                        self.hparams.save_folder, "valid_stats", str(epoch)
                    )
                    os.makedirs(save_root, exist_ok=True)
                    with open(os.path.join(save_root, "p1_loss.txt"), "w") as f:
                        self.p1_loss_metric.write_stats(f)

                    with open(os.path.join(save_root, "wer_stats.txt"), "w") as f:
                        self.wer_stats.write_stats(f)

                    with open(os.path.join(save_root, "cer_stats.txt"), "w") as f:
                        self.cer_stats.write_stats(f)

                else:
                    stats = {
                        "loss": stage_loss,
                    }

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
            if self.hparams.ckpt_enable:
                self.checkpointer.save_and_keep_only(meta=stats, min_keys=["wer"])

            hp.report_result(stats)
            self.asr_model.aux.cpu()

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            self.asr_model.aux.cpu()

    @torch.no_grad
    def inference(self, batch, stage=None):
        emb_noisy, emb_clean, mask = self.pad_and_transpose(batch)

        def vt(t, x, args):
            x = x * mask
            ut_pred = self.modules.model(x, mask, emb_noisy, t)
            return ut_pred * mask

        node = NeuralODE(vt, solver="euler", sensitivity="adjoint")
        traj = node.trajectory(emb_noisy, t_span=torch.linspace(0, 1, 50))

        emb_pred = traj[-1]
        traj.cpu()

        p1_pred = emb_pred  # save for loss stats

        emb_pred = emb_pred[:, :, : mask.sum(dim=-1).max()].transpose(1, 2)

        emission = self.asr_model.aux(emb_pred)
        hyps = [self.decoder(e) for e in emission]

        return hyps, p1_pred


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

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("path_emb_noisy", "path_emb_clean", "wav_len")
    @sb.utils.data_pipeline.provides("emb_noisy", "emb_clean", "wav_len")
    def audio_pipeline(path_emb_noisy, path_emb_clean, wav_len):
        emb_noisy = torch.load(path_emb_noisy)
        emb_clean = torch.load(path_emb_clean)
        return emb_noisy, emb_clean, wav_len

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd")
    def text_pipeline(wrd):
        yield wrd

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
                "wrd",
            ],
        )
    return datasets


# Recipe begins!
if __name__ == "__main__":
    with hp.hyperparameter_optimization(objective_key="loss") as hp_ctx:
        # Reading command line arguments
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:])

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
                    "model_id": hparams["model_id"],
                    "save_json_train": hparams["train_annotation"],
                    "save_json_valid": hparams["valid_annotation"],
                    "save_json_test": hparams["test_annotation"],
                },
            )

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

        # prepare asr model
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        decoder = GreedyCTCDecoder(labels=bundle.get_labels())
        model = bundle.get_model(
            dl_kwargs={
                "model_dir": hparams["model_dir"],
                "file_name": hparams["model_id"],
            }
        )

        se_brain.decoder = decoder
        se_brain.asr_model = model
        se_brain.labels = bundle.get_labels()

        # The `fit()` method iterates the training loop, calling the methods
        # necessary to update the parameters of the model. Since all objects
        # with changing state are managed by the Checkpointer, training can be
        # stopped at any point, and will be resumed on next call.
        se_brain.fit(
            epoch_counter=se_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options_eval"],
        )

        if hparams["test"]:
            # Load best checkpoint for evaluation
            test_stats = se_brain.evaluate(
                test_set=datasets["test"],
                min_key="wer",
                test_loader_kwargs=hparams["dataloader_options_eval"],
            )
