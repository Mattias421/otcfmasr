import torch
from dataclasses import replace

import warnings
import tqdm  # Ensure tqdm is imported
from typing import List, Tuple, Optional, Union  # Ensure these are imported
from whisper.utils import (  # Make sure necessary utils are imported
    format_timestamp,
    make_safe,
)

# Import constants used in transcribe
from whisper.audio import (
    N_FRAMES,
    HOP_LENGTH,
    SAMPLE_RATE,
    FRAMES_PER_SECOND,  # This might need adjustment or reinterpretation
)

# Import other necessary components from whisper
from whisper.tokenizer import get_tokenizer
from whisper.decoding import (
    DecodingOptions,
    DecodingResult,
    DecodingTask,
)  # Add DecodingTask if not already imported


class EMBDecodingTask(DecodingTask):
    def _get_audio_features(self, emb):
        return emb


@torch.no_grad()
def decode_embeddings(
    model_wspr,
    mel,
    options,
    **kwargs,
):
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result = EMBDecodingTask(model_wspr, options).run(mel)

    return result[0] if single else result


# Potentially needed for word timestamps (if adapted, currently removed/warned)
# from .timing import add_word_timestamps, get_end

# Define constants relevant to embeddings
# The encoder halves the time dimension relative to mel frames
EMBEDDING_FRAMES_PER_SECOND = FRAMES_PER_SECOND // 2
# N_FRAMES corresponds to 30 seconds of mel frames
EMBEDDING_FRAMES_PER_30_SEC = (
    N_FRAMES // 2
)  # Usually 1500, same as model.dims.n_audio_ctx


def transcribe_embeddings(
    model,
    audio_embeddings: torch.Tensor,  # Expecting shape [TotalEmbeddingFrames, EmbeddingDim] or [1, TotalEmbeddingFrames, EmbeddingDim]
    *,
    language: str,  # Language is now mandatory
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    carry_initial_prompt: bool = False,
    # word_timestamps: bool = False, # Word timestamps are problematic with only embeddings
    # prepend_punctuations: str = "\"'“¿([{-",
    # append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    clip_timestamps: Union[
        str, List[float]
    ] = "0",  # In seconds relative to original audio duration
    # hallucination_silence_threshold: Optional[float] = None,
    **decode_options,
):
    """
    Transcribe audio represented by pre-computed Whisper encoder embeddings.

    Parameters
    ----------
    model: Whisper
        The Whisper model instance.

    audio_embeddings: torch.Tensor
        The pre-computed audio embeddings from the Whisper encoder.
        Expected shape: [N_total_embedding_frames, N_embedding_dims] or [1, N_total_embedding_frames, N_embedding_dims].

    language: str
        The language spoken in the audio. This MUST be specified.

    verbose: bool, temperature, ..., **decode_options:
        See the original `transcribe` function documentation.
        Note: `word_timestamps` and related options are currently disabled/ignored
              as they rely on Mel spectrograms for reliable alignment.

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language").
    """

    # --- Input Validation and Setup ---
    if language is None:
        raise ValueError(
            "The 'language' parameter must be specified when using transcribe_embeddings."
        )
    decode_options["language"] = language  # Ensure language is in decode_options

    if audio_embeddings.ndim == 2:
        audio_embeddings = audio_embeddings.unsqueeze(0)  # Add batch dimension
    elif audio_embeddings.ndim != 3:
        raise ValueError(
            f"Expected audio_embeddings with shape [1, N, D] or [N, D], got {audio_embeddings.shape}"
        )

    # Squeeze batch dim if it's 1 for easier slicing later? Or keep it? Keep it for consistency with mel.
    # audio_embeddings = audio_embeddings.squeeze(0) # Let's keep batch dim [1, N, D]

    n_batch, total_embedding_frames, n_dims = audio_embeddings.shape
    if n_batch != 1:
        warnings.warn(
            f"Processing embeddings with batch size {n_batch}. Only batch size 1 is fully tested for transcribe_embeddings."
        )

    # Check embedding dimension matches model
    expected_dims = model.dims.n_audio_state
    if n_dims != expected_dims:
        raise ValueError(
            f"Embedding dimension ({n_dims}) does not match model's expected audio state dimension ({expected_dims})"
        )

    device = audio_embeddings.device
    dtype = audio_embeddings.dtype  # Use the dtype of the provided embeddings

    # Warn if using CPU/FP16 (though embeddings might already be computed optimally)
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            # This check might be less relevant if embeddings are pre-computed
            warnings.warn("Using FP16 embeddings on CPU.")
            # Don't change dtype here, use the provided one.

    # Set fp16 option based on embedding dtype for decode options
    decode_options["fp16"] = dtype == torch.float16

    # Calculate total duration based on embeddings
    # Each embedding frame corresponds to input_stride * HOP_LENGTH / SAMPLE_RATE seconds
    # input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx) -> Typically 2
    time_per_embedding_frame = (HOP_LENGTH / SAMPLE_RATE) * 2  # Usually 0.02 seconds

    # --- Language/Tokenizer Setup (Simplified) ---
    # Language is mandatory, no detection needed here.
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    # --- Clipping Timestamps (Converted to Embedding Frames) ---
    if isinstance(clip_timestamps, str):
        clip_timestamps = [
            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
        ]
    # Convert clip timestamps (seconds) to embedding frame indices
    seek_points_emb: List[int] = [
        round(ts / time_per_embedding_frame) for ts in clip_timestamps
    ]

    if len(seek_points_emb) == 0:
        seek_points_emb.append(0)
    if len(seek_points_emb) % 2 == 1:
        seek_points_emb.append(total_embedding_frames)
    # Ensure points are within bounds
    seek_points_emb = [min(max(0, p), total_embedding_frames) for p in seek_points_emb]

    seek_clips_emb: List[Tuple[int, int]] = list(
        zip(seek_points_emb[::2], seek_points_emb[1::2])
    )

    # --- Word Timestamp Warning ---
    if decode_options.get("word_timestamps", False):
        warnings.warn(
            "Word timestamps are experimental and may be unreliable when using pre-computed embeddings. Disabling."
        )
        decode_options["word_timestamps"] = False  # Force disable

    # --- Decoding Fallback Logic (Adapted for Embeddings) ---
    def decode_with_fallback_embeddings(
        segment_embeddings: torch.Tensor,
    ) -> DecodingResult:
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None

        # Ensure segment has batch dimension
        if segment_embeddings.ndim == 2:
            segment_embeddings = segment_embeddings.unsqueeze(0)

        # Pad or trim the embedding segment to the expected context length (N_CTX)
        # N_CTX is typically EMBEDDING_FRAMES_PER_30_SEC (1500)
        n_ctx = model.dims.n_audio_ctx
        current_len = segment_embeddings.shape[1]

        if current_len < n_ctx:
            padding = n_ctx - current_len
            # Pad with zeros? Or replicate last frame? Zeros seems safer.
            segment_embeddings = torch.nn.functional.pad(
                segment_embeddings,
                (0, 0, 0, padding),  # Pad last dim (time/ctx)
            )
        elif current_len > n_ctx:
            segment_embeddings = segment_embeddings[:, :n_ctx, :]

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            # *** Call the new decode_embeddings function ***
            decode_result = decode_embeddings(
                model, segment_embeddings, options
            )  # Pass embedding segment

            # Fallback logic remains the same
            needs_fallback = False
            decode_result = decode_result[0]
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
                and logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                # Check if this segment is potentially silent despite the logprob.
                # This might be less reliable without the direct encoder run,
                # but we keep the logic for consistency.
                needs_fallback = False  # Considered silence

            if not needs_fallback:
                break

        return decode_result

    # --- Main Transcription Loop (Iterating over Embeddings) ---
    clip_idx = 0
    seek = seek_clips_emb[clip_idx][0]  # seek is now in embedding frame indices
    time_precision = time_per_embedding_frame  # time per output token: 0.02 (seconds)

    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    remaining_prompt_length = model.dims.n_text_ctx // 2 - 1
    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
        remaining_prompt_length -= len(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def new_segment(
        *,
        current_seek: int,
        start: float,
        end: float,
        tokens: torch.Tensor,
        result: DecodingResult,
    ):
        # `seek` in the returned segment should arguably be the original audio seek point
        # or maybe the embedding seek point. Let's use embedding seek point for now.
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": current_seek,  # The starting *embedding frame index* of this chunk
            "start": start,  # Start time in seconds
            "end": end,  # End time in seconds
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    # Show the progress bar based on embedding frames
    with tqdm.tqdm(
        total=total_embedding_frames, unit="frames", disable=verbose is not False
    ) as pbar:
        # last_speech_timestamp = 0.0 # Not used if word timestamps disabled

        while clip_idx < len(seek_clips_emb):
            seek_clip_start, seek_clip_end = seek_clips_emb[clip_idx]
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips_emb):
                    seek = seek_clips_emb[clip_idx][0]
                continue

            # Calculate time offset based on embedding frame seek position
            time_offset = float(seek * time_precision)
            # window_end_time = float((seek + EMBEDDING_FRAMES_PER_30_SEC) * time_precision) # Not strictly needed?

            # Determine the size of the current embedding segment to process
            segment_size_emb = min(
                EMBEDDING_FRAMES_PER_30_SEC,
                total_embedding_frames - seek,
                seek_clip_end - seek,
            )
            if (
                segment_size_emb <= 0
            ):  # Should not happen with proper clip logic, but check
                break

            # Slice the embedding tensor [batch, time, dim]
            embedding_segment = audio_embeddings[:, seek : seek + segment_size_emb, :]
            segment_duration = segment_size_emb * time_precision  # Duration in seconds

            # --- Prompt Handling ---
            if carry_initial_prompt:
                nignored = max(len(initial_prompt_tokens), prompt_reset_since)
                remaining_prompt = all_tokens[nignored:][-remaining_prompt_length:]
                decode_options["prompt"] = initial_prompt_tokens + remaining_prompt
            else:
                decode_options["prompt"] = all_tokens[prompt_reset_since:]

            # --- Decode the Embedding Segment ---
            # Note: padding/trimming happens inside decode_with_fallback_embeddings
            result: DecodingResult = decode_with_fallback_embeddings(embedding_segment)
            tokens = torch.tensor(
                result.tokens, device=device
            )  # Put tokens back on device maybe?

            # --- No Speech Check ---
            if no_speech_threshold is not None:
                should_skip = result.no_speech_prob > no_speech_threshold
                if (
                    logprob_threshold is not None
                    and result.avg_logprob > logprob_threshold
                ):
                    should_skip = False

                if should_skip:
                    seek += (
                        segment_size_emb  # Fast-forward to the next segment boundary
                    )
                    pbar.update(segment_size_emb)  # Update progress for skipped segment
                    continue  # Skip processing this segment

            # --- Process Tokens and Timestamps ---
            previous_seek = seek
            current_segments = []

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)

            if len(consecutive) > 0:
                # Output contains multiple consecutive timestamps
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            current_seek=seek,  # Pass the start seek of the chunk
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # No speech after the last timestamp. Advance seek by the full segment.
                    seek += segment_size_emb
                else:
                    # Ignore the unfinished segment and seek to the last timestamp predicted
                    last_timestamp_pos = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    # Seek is advanced by the number of *embedding frames* corresponding to the timestamp
                    seek += round(
                        last_timestamp_pos * time_precision / time_per_embedding_frame
                    )
                    # Ensure seek doesn't go backwards or stay static if timestamp is 0
                    seek = max(seek, previous_seek + 1)

            else:
                # No consecutive timestamps. Treat as a single segment.
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if (
                    len(timestamps) > 0
                    and timestamps[-1].item()
                    >= tokenizer.timestamp_begin  # Check it's a valid timestamp token
                ):
                    # Use the last predicted timestamp if available
                    last_timestamp_pos = (
                        timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision

                current_segments.append(
                    new_segment(
                        current_seek=seek,
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        result=result,
                    )
                )
                # Advance seek by the full segment size
                seek += segment_size_emb

            # --- Word Timestamp Logic (Removed/Disabled) ---
            # if word_timestamps:
            #    warnings.warn("Word timestamps skipped as they are not supported for embeddings.")
            # Hallucination logic based on word timestamps also removed

            if verbose:
                for segment in current_segments:
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                    print(make_safe(line))

            # Clean up empty segments
            for i, segment in enumerate(current_segments):
                if segment["start"] >= segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    # segment["words"] = [] # No words key if word timestamps disabled

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(
                        current_segments, start=len(all_segments)
                    )
                    if segment["text"]  # Only add segments with actual text
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )

            if not condition_on_previous_text or result.temperature > 0.5:
                prompt_reset_since = len(all_tokens)

            # Update progress bar: seek might jump, calculate actual advancement
            pbar.update(max(0, seek - previous_seek))

    # --- Final Output ---
    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language,
    )
