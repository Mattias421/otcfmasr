import re
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
import random


def loss_fn(ut_pred, ut, mask):
    loss = F.mse_loss(ut_pred, ut, reduction="sum") / (torch.sum(mask) * ut.shape[1])
    return loss


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    """Class that manages the training loop. See speechbrain.core.Brain."""

    # def pad_and_transpose(self, batch):
    #     emb_noisy = batch.emb_noisy.data.to(self.device)
    #     emb_clean = batch.emb_clean.data.to(self.device)
    #
    #     B, L, C = emb_noisy.shape
    #
    #     # if not pad, cut
    #     if self.hparams.pad:
    #         L = L if L > emb_clean.shape[1] else emb_clean.shape[1]
    #     else:
    #         L = L if L < emb_clean.shape[1] else emb_clean.shape[1]
    #
    #     max_length = fix_len_compatibility(L, num_downsamplings_in_unet=2)
    #     lengths_noisy = batch.emb_noisy.lengths * emb_noisy.shape[1]
    #     lengths_clean = batch.emb_clean.lengths * emb_clean.shape[1]
    #
    #     if self.hparams.pad:
    #         lengths = torch.maximum(lengths_noisy, lengths_clean)
    #     else:
    #         lengths = torch.minimum(lengths_noisy, lengths_clean)
    #
    #     mask = sequence_mask(lengths, max_length=max_length)
    #     mask = mask[:, None, :].to(self.device)
    #
    #     emb_noisy_input = torch.zeros((B, C, max_length)).to(self.device)
    #
    #     emb_clean_input = torch.zeros((B, C, max_length)).to(self.device)
    #
    #     if self.hparams.pad:
    #         emb_noisy_input[:, :, : emb_noisy.shape[1]] = emb_noisy.transpose(1, 2)
    #         emb_clean_input[:, :, : emb_clean.shape[1]] = emb_clean.transpose(1, 2)
    #     else:
    #         emb_noisy_input[:, :, :L] = emb_noisy.transpose(1, 2)[:, :, :L]
    #         emb_clean_input[:, :, :L] = emb_clean.transpose(1, 2)[:, :, :L]
    #
    #     return emb_clean_input, emb_noisy_input, mask

    def pad_and_transpose(self, batch, cut_segments=False):
        """
        Prepares noisy and clean embeddings for the model.

        Handles optional segment cutting, padding/truncation, and transposing.

        Arguments:
        ----------
        batch : PaddedBatch
            Must contain 'emb_noisy' and 'emb_clean' PaddedData fields.

        Returns:
        --------
        emb_clean_input : torch.Tensor
            Processed clean embeddings (B, C, L_out).
        emb_noisy_input : torch.Tensor
            Processed noisy embeddings (B, C, L_out).
        mask : torch.Tensor
            Boolean mask for the processed length (B, 1, L_out).
        """
        # 1. Get raw data and absolute lengths
        # Assuming batch.emb_*.data is shape (B, L_padded, C)
        # Assuming batch.emb_*.lengths is relative (0.0 to 1.0)
        emb_noisy_data = batch.emb_noisy.data.to(self.device)
        emb_clean_data = batch.emb_clean.data.to(self.device)
        B, L_noisy_padded, C = emb_noisy_data.shape
        _, L_clean_padded, _ = emb_clean_data.shape

        # Calculate absolute lengths in frames
        lengths_noisy_abs = torch.round(batch.emb_noisy.lengths * L_noisy_padded).int()
        lengths_clean_abs = torch.round(batch.emb_clean.lengths * L_clean_padded).int()

        # --- Branch based on whether segment cutting is enabled ---
        if cut_segments:
            segment_length = self.hparams.segment_length
            num_downsamplings = getattr(
                self.hparams, "num_downsamplings_in_unet", 2
            )  # Get from hparams or use default
            final_segment_length = fix_len_compatibility(
                segment_length, num_downsamplings
            )

            # Determine lengths to base the cut on (use minimum actual length)
            cut_base_lengths = torch.minimum(lengths_noisy_abs, lengths_clean_abs).to(
                self.device
            )

            # Calculate max offset for random start point
            # Ensure offset doesn't go past the possible start for a full segment
            max_offset = (cut_base_lengths - segment_length).clamp(min=0)

            # Generate random offsets for each batch item
            offset_ranges = list(
                zip([0] * B, max_offset.cpu().numpy())
            )  # Use cpu for range
            out_offset = torch.LongTensor(
                # random.choice picks from range(start, end). If end=start, range is empty.
                # Need range(start, end + 1) if end is inclusive, or handle end=start.
                [
                    random.choice(range(start, end + 1)) if end >= start else 0
                    for start, end in offset_ranges
                ]
            ).to(self.device)

            # Initialize tensors for cut segments (B, C, final_segment_length)
            # Note: We transpose *before* cutting for potential efficiency if C << L
            emb_noisy_data_t = emb_noisy_data.transpose(1, 2)  # (B, C, L_padded)
            emb_clean_data_t = emb_clean_data.transpose(1, 2)  # (B, C, L_padded)

            emb_noisy_cut = torch.zeros(
                (B, C, final_segment_length),
                dtype=emb_noisy_data_t.dtype,
                device=self.device,
            )
            emb_clean_cut = torch.zeros(
                (B, C, final_segment_length),
                dtype=emb_clean_data_t.dtype,
                device=self.device,
            )

            # Calculate the actual length of the segment that will be cut for each item
            # min(desired_length, available_length_from_offset)
            actual_cut_lengths = torch.minimum(
                torch.full_like(cut_base_lengths, segment_length),  # Desired length
                cut_base_lengths - out_offset,  # Available length
            ).clamp(min=0)  # Ensure non-negative length

            # Loop through batch and copy segments
            for i in range(B):
                start_idx = out_offset[i]
                length_to_copy = actual_cut_lengths[i].item()  # Use item() for scalar
                end_idx = start_idx + length_to_copy

                if length_to_copy > 0:
                    # Copy the slice into the target tensor, up to final_segment_length
                    emb_noisy_cut[i, :, :length_to_copy] = emb_noisy_data_t[
                        i, :, start_idx:end_idx
                    ]
                    emb_clean_cut[i, :, :length_to_copy] = emb_clean_data_t[
                        i, :, start_idx:end_idx
                    ]

            # Final outputs are the cut tensors
            emb_noisy_input = emb_noisy_cut
            emb_clean_input = emb_clean_cut
            # Lengths for the mask are the lengths of the segments actually copied
            final_lengths = actual_cut_lengths
            max_length_for_mask = final_segment_length  # Mask uses the padded length

        # --- Original Padding/Truncating Logic (if not cutting segments) ---
        else:
            num_downsamplings = getattr(
                self.hparams, "num_downsamplings_in_unet", 2
            )  # Get from hparams or use default

            # Determine target length L based on padding preference and actual lengths
            if self.hparams.pad:
                # Pad: use the maximum actual length observed in the batch
                L = torch.maximum(lengths_noisy_abs, lengths_clean_abs).max().item()
            else:
                # Cut/Truncate: use the minimum actual length observed in the batch
                L = torch.minimum(lengths_noisy_abs, lengths_clean_abs).min().item()
                # Note: This interpretation differs slightly from your original snippet's
                # comparison of padded lengths, but seems more robust.

            # Make target length compatible with model architecture (e.g., U-Net)
            max_length = fix_len_compatibility(L, num_downsamplings)

            # Determine final lengths for the mask based on padding/truncation choice
            if self.hparams.pad:
                # If padding, final length is the max original length (up to max_length)
                final_lengths = torch.maximum(
                    lengths_noisy_abs, lengths_clean_abs
                ).clamp(max=max_length)
            else:
                # If truncating, final length is the min original length (up to max_length)
                final_lengths = torch.minimum(
                    lengths_noisy_abs, lengths_clean_abs
                ).clamp(max=max_length)

            # Initialize output tensors (B, C, max_length)
            emb_noisy_input = torch.zeros(
                (B, C, max_length), dtype=emb_noisy_data.dtype, device=self.device
            )
            emb_clean_input = torch.zeros(
                (B, C, max_length), dtype=emb_clean_data.dtype, device=self.device
            )

            # Transpose originals for easier slicing (B, C, L_padded)
            emb_noisy_data_t = emb_noisy_data.transpose(1, 2)
            emb_clean_data_t = emb_clean_data.transpose(1, 2)

            # Loop and copy data, applying padding or truncation implicitly
            for i in range(B):
                # Determine length to copy for this item (min of its actual length and max_length)
                len_noisy_copy = min(lengths_noisy_abs[i].item(), max_length)
                len_clean_copy = min(lengths_clean_abs[i].item(), max_length)

                # Copy the data
                if len_noisy_copy > 0:
                    emb_noisy_input[i, :, :len_noisy_copy] = emb_noisy_data_t[
                        i, :, :len_noisy_copy
                    ]
                if len_clean_copy > 0:
                    emb_clean_input[i, :, :len_clean_copy] = emb_clean_data_t[
                        i, :, :len_clean_copy
                    ]

            max_length_for_mask = max_length  # Mask uses the padded/truncated length

        # --- Create Mask ---
        # Mask should reflect the actual content length within the final tensor shape
        mask = sequence_mask(final_lengths, max_length=max_length_for_mask)
        mask = mask.unsqueeze(1).to(
            self.device
        )  # Add channel dim: (B, 1, max_length_for_mask)

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

        emb_noisy, emb_clean, mask = self.pad_and_transpose(
            batch, self.hparams.cut_segments
        )

        if self.hparams.otcfm:
            emb_noisy_agg = emb_noisy.sum(dim=1) / (mask.sum(dim=-1))[:, None]
            emb_clean_agg = emb_clean.sum(dim=1) / (mask.sum(dim=-1))[:, None]

            _, _, label_x, label_y = self.hparams.ot_sampler.sample_plan_with_labels(
                emb_noisy_agg,
                emb_clean_agg,
                torch.arange(self.hparams.batch_size),
                torch.arange(self.hparams.batch_size),
                replace=self.hparams.ot_replace,
            )

            emb_noisy = emb_noisy[label_x]
            emb_clean = emb_clean[label_y]

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
            hyps = predictions["hyps"]

            pattern_unwanted = r"[^{}\s]".format(
                "".join(re.escape(c) for c in self.labels)
            )
            target_words = [
                [
                    wrd
                    for wrd in re.sub(pattern_unwanted, "", utt.upper()).split(" ")
                    if wrd != ""
                ]
                for utt in batch.wrd
            ]

            # Convert indices to words
            predicted_words = [
                [wrd for wrd in text.split("|") if wrd != ""] for text in hyps
            ]

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
                    "wer": self.wer_stats.summarize("WER"),
                    "cer": self.cer_stats.summarize("WER"),
                    "p1_loss": self.p1_loss_metric.summarize("average"),
                }
                save_root = os.path.join(self.hparams.save_folder, "test_stats")
                os.makedirs(save_root, exist_ok=True)
                with open(os.path.join(save_root, "p1_loss.txt"), "w") as f:
                    self.p1_loss_metric.write_stats(f)

                with open(os.path.join(save_root, "wer_stats.txt"), "w") as f:
                    self.wer_stats.write_stats(f)

                with open(os.path.join(save_root, "cer_stats.txt"), "w") as f:
                    self.cer_stats.write_stats(f)

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
            print(f"Final stats are {stats}")
            hp.report_result(stats)

    @torch.no_grad
    def inference(self, batch, stage=None):
        # make emb_clean same as noisy for correct mask
        # batch_infer = batch
        # batch_infer.emb_clean = batch_infer.emb_noisy
        # emb_noisy, emb_clean, mask = self.pad_and_transpose(batch_infer)
        emb_noisy, emb_clean, mask = self.pad_and_transpose(batch)

        def vt(t, x, args):
            x = x * mask
            ut_pred = self.modules.model(x, mask, emb_noisy, t)
            return ut_pred * mask

        node = NeuralODE(vt, solver="euler", sensitivity="adjoint")
        traj = node.trajectory(
            emb_noisy, t_span=torch.linspace(0, 1, self.hparams.n_ode_steps)
        )

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
                    "pairing_mode": hparams["pairing_mode"],
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
