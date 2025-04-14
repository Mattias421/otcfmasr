# NOTE https://github.com/Blair-Johnson/batch-whisper/blob/main/whisper/transcribe.py
import warnings
import copy
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import torch
import tqdm

from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH
from whisper.tokenizer import get_tokenizer
from whisper.utils import exact_div, format_timestamp

from .decoding import EmbDecodingTask, DecodingOptions, DecodingResult

if TYPE_CHECKING:
    from whisper.model import Whisper


def transcribe_embeddings(
    model: "Whisper",
    embedding: List[torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    **decode_options,
):
    """
    Transcribe multiple audio files in parallel using the batch dimension of the Whisper model

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[List[str], List[np.ndarray], List[torch.Tensor]]
        The list of paths to the audio files to open, or the audio waveforms

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A list of dictionaries containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    batch_size = embedding.shape[0]

    segments = [emb for emb in embedding]

    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            languages = ["en"] * batch_size
        else:
            # if verbose:
            #     print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            # language_probs = [model.detect_language(segment)[1] for segment in segments]
            # languages = [max(probs, key=probs.get) for probs in language_probs]
            # if verbose is not None:
            #     print(f"Detected languages: {[LANGUAGES[opt].title() for opt in languages]}")
            raise ValueError("Detecting language is not supported for embs")
    else:
        lang = decode_options.get("language")
        if isinstance(lang, str):
            languages = [lang] * batch_size
        elif isinstance(lang, list):
            assert all(isinstance(lan, str) for lan in lang), (
                "If a list of languages is specified in DecodeOptions, all languages must be strings."
            )
            assert len(lang) == batch_size, (
                "If a list of languages is specified in DecodeOptions, the list length must match the number of audio files specified."
            )
            languages = lang
        else:
            raise NotImplementedError(
                "Only string and list arguments are supported for the language DecodeOption."
            )

    task = decode_options.get("task", "transcribe")
    tokenizers = {}
    for lang in languages:
        if lang not in tokenizers.keys():
            tokenizers[lang] = get_tokenizer(
                model.is_multilingual, language=lang, task=task
            )

    def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None
        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = EmbDecodingTask(model, options).run(segment)

            needs_fallback = False
            if isinstance(decode_result, list):
                for dr in decode_result:
                    if (
                        compression_ratio_threshold is not None
                        and dr.compression_ratio > compression_ratio_threshold
                    ):
                        needs_fallback = True  # too repetitive
                    if (
                        logprob_threshold is not None
                        and dr.avg_logprob < logprob_threshold
                    ):
                        needs_fallback = True  # average log probability is too low
            else:
                if (
                    compression_ratio_threshold is not None
                    and decode_result.compression_ratio > compression_ratio_threshold
                ):
                    needs_fallback = True  # too repetitive
                if (
                    logprob_threshold is not None
                    and decode_result.avg_logprob < logprob_threshold
                ):
                    needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return decode_result

    seekers = [0] * batch_size
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        HOP_LENGTH / SAMPLE_RATE
    ) * 2  # time per output token: 0.02 (seconds)
    all_tokens = [[] for _ in range(batch_size)]
    all_segments = [[] for _ in range(batch_size)]
    prompt_reset_since = [0] * batch_size

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    initial_prompts = []
    if initial_prompt:
        assert len(initial_prompt) == batch_size, (
            "Number of initial prompts must match batch size."
        )
        for i in range(batch_size):
            initial_prompts.append(
                tokenizers[languages[i]].encode(" " + initial_prompt[i].strip())
            )
            all_tokens.extend(initial_prompt)

    def add_segment(
        *,
        seeker: int,
        segments: list,
        start: float,
        end: float,
        text_tokens: torch.Tensor,
        result: DecodingResult,
        tokenizer,
    ):
        text = tokenizer.decode(
            [token for token in text_tokens if token < tokenizer.eot]
        )
        if len(text.strip()) == 0:  # skip empty text output
            return

        segments.append(
            {
                "id": len(segments),
                "seek": seeker,
                "start": start,
                "end": end,
                "text": text,
                "tokens": text_tokens.tolist(),
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = [embedding.shape[1] for i in range(batch_size)]
    previous_seek_values = copy.deepcopy(seekers)
    breakpoint()

    def check_cursors(seekers: List[int], num_frames: List[int]) -> bool:
        """Return False when all seekers have exhausted the length of their audio clips."""
        return any([seeker < nf for seeker, nf in list(zip(seekers, num_frames))])

    with tqdm.tqdm(
        total=max(num_frames), unit="frames", disable=verbose is not False
    ) as pbar:
        while check_cursors(seekers, num_frames):
            continue_processing = [
                seeker < nf for seeker, nf in list(zip(seekers, num_frames))
            ]
            # Only those segments for clips that are not done being processed
            imap = [i for i, v in enumerate(continue_processing) if v]
            batch_segments = []
            batch_segment_durations = []
            batch_timestamp_offsets = []
            for i, emb in enumerate(embedding):
                if continue_processing[i]:
                    timestamp_offset = float(seekers[i] * HOP_LENGTH / SAMPLE_RATE)
                    batch_timestamp_offsets.append(timestamp_offset)
                    segment = emb  # assume we want the whole embedding
                    segment_duration = 30  # assume embedding is full 30s
                    batch_segments.append(segment)
                    batch_segment_durations.append(segment_duration)
                else:
                    continue

            decode_options["prompt"] = [
                all_tokens[imap[i]][prompt_reset_since[imap[i]] :]
                for i in range(len(batch_segments))
            ]
            decode_options["language"] = [
                lan for i, lan in enumerate(languages) if continue_processing[i]
            ]
            results: List[DecodingResult] = decode_with_fallback(
                torch.stack(batch_segments)
            )
            batch_tokens = [torch.tensor(result.tokens) for result in results]

            no_speech_results = [False] * len(results)
            if no_speech_threshold is not None:
                for i, result in enumerate(results):
                    # no voice activity check
                    should_skip = result.no_speech_prob[i] > no_speech_threshold
                    if (
                        logprob_threshold is not None
                        and result.avg_logprob > logprob_threshold
                    ):
                        # don't skip if the logprob is high enough, despite the no_speech_prob
                        should_skip = False

                    if should_skip:
                        seekers[imap[i]] += segment.shape[
                            -1
                        ]  # fast-forward to the next segment boundary
                        no_speech_results[i] = True

            batch_timestamp_tokens: List[torch.Tensor] = [
                tokens.ge(tokenizers[languages[imap[i]]].timestamp_begin)
                for i, tokens in enumerate(batch_tokens)
            ]
            batch_consecutive = [
                torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
                for timestamp_tokens in batch_timestamp_tokens
            ]

            for i, consecutive in enumerate(batch_consecutive):
                if no_speech_results[i]:
                    continue
                if (
                    len(consecutive) > 0
                ):  # if the output contains two consecutive timestamp tokens
                    last_slice = 0
                    for current_slice in consecutive:
                        sliced_tokens = batch_tokens[i][last_slice:current_slice]
                        start_timestamp_position = (
                            sliced_tokens[0].item()
                            - tokenizers[languages[imap[i]]].timestamp_begin
                        )
                        end_timestamp_position = (
                            sliced_tokens[-1].item()
                            - tokenizers[languages[imap[i]]].timestamp_begin
                        )
                        add_segment(
                            seeker=seekers[imap[i]],
                            segments=all_segments[imap[i]],
                            start=batch_timestamp_offsets[i]
                            + start_timestamp_position * time_precision,
                            end=batch_timestamp_offsets[i]
                            + end_timestamp_position * time_precision,
                            text_tokens=sliced_tokens[1:-1],
                            result=results[i],
                            tokenizer=tokenizers[languages[imap[i]]],
                        )
                        last_slice = current_slice
                    last_timestamp_position = (
                        batch_tokens[i][last_slice - 1].item()
                        - tokenizers[languages[imap[i]]].timestamp_begin
                    )
                    seekers[imap[i]] += last_timestamp_position * input_stride
                    all_tokens[imap[i]].extend(
                        batch_tokens[i][: last_slice + 1].tolist()
                    )
                else:
                    duration = batch_segment_durations[i]
                    timestamps = batch_tokens[i][
                        batch_timestamp_tokens[i].nonzero().flatten()
                    ]
                    if (
                        len(timestamps) > 0
                        and timestamps[-1].item()
                        != tokenizers[languages[imap[i]]].timestamp_begin
                    ):
                        # no consecutive timestamps but it has a timestamp; use the last one.
                        # single timestamp at the end means no speech after the last timestamp.
                        last_timestamp_position = (
                            timestamps[-1].item()
                            - tokenizers[languages[imap[i]]].timestamp_begin
                        )
                        duration = last_timestamp_position * time_precision

                    add_segment(
                        seeker=seekers[imap[i]],
                        segments=all_segments[imap[i]],
                        start=batch_timestamp_offsets[i],
                        end=batch_timestamp_offsets[i] + duration,
                        text_tokens=batch_tokens[i],
                        result=results[i],
                        tokenizer=tokenizers[languages[imap[i]]],
                    )

                    seekers[imap[i]] += segments[imap[i]].shape[-1]
                    all_tokens[imap[i]].extend(batch_tokens[i].tolist())

                if not condition_on_previous_text or results[i].temperature > 0.5:
                    # do not feed the prompt tokens if a high temperature was used
                    prompt_reset_since[imap[i]] = len(all_tokens[imap[i]])

            # update progress bar
            midx = num_frames.index(max(num_frames))
            pbar.update(
                min(num_frames[midx], seekers[midx]) - previous_seek_values[midx]
            )
            previous_seek_values = copy.deepcopy(seekers)

    return [
        dict(
            text=tokenizers[languages[i]].decode(
                [
                    token
                    for token in all_tokens[i][len(initial_prompt) :]
                    if token < tokenizers[languages[i]].eot
                ]
            ),
            segments=all_segments[i],
            language=languages[i],
        )
        for i in range(len(all_segments))
    ]
