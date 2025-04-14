import warnings
import copy
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import torch
from torch import Tensor
import tqdm
import numpy as np

from whisper.audio import (
    SAMPLE_RATE,
    N_FRAMES,
    HOP_LENGTH,
    CHUNK_LENGTH,
)
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.decoding import (
    PyTorchInference,
    BeamSearchDecoder,
    MaximumLikelihoodRanker,
    GreedyDecoder,
    SuppressTokens,
    SuppressBlank,
    ApplyTimestampRules,
)
from whisper.tokenizer import get_tokenizer, Tokenizer
from whisper.utils import exact_div, format_timestamp, compression_ratio

N_FRAMES = N_FRAMES // 2  # div by 2 to go from mel frames to audio_emb

if TYPE_CHECKING:
    from whisper.model import Whisper


class EmbDecodingTask:
    # https://github.com/Blair-Johnson/batch-whisper
    # inference: Inference
    # sequence_ranker: SequenceRanker
    # decoder: TokenDecoder
    # logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        # NOTE: This is the main decoding loop
        self.model = model

        language = options.language or "en"
        if isinstance(language, list):
            self.tokenizers = [
                get_tokenizer(model.is_multilingual, language=lang, task=options.task)
                for lang in language
            ]
        else:
            tokenizer = get_tokenizer(
                model.is_multilingual, language=language, task=options.task
            )
            self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.logit_filters = []
        if isinstance(language, list):
            self.sot_sequence: List[Tuple[int]] = []
            for i in range(len(language)):
                if self.options.without_timestamps:
                    self.sot_sequence.append(
                        self.tokenizers[i].sot_sequence_including_notimestamps
                    )
                else:
                    self.sot_sequence.append(self.tokenizers[i].sot_sequence)

            # self.initial_tokens: Tuple[int] = self._get_initial_tokens()
            self.initial_tokens = self._get_initial_tokens()

            # branch to handle batched case
            self.sample_begin: List[int] = [
                len(tokens) for tokens in self.initial_tokens
            ]
            self.sot_index: List[int] = [
                tokens.index(self.tokenizers[i].sot)
                for i, tokens in enumerate(self.initial_tokens)
            ]

            # inference: implements the forward pass through the decoder, including kv caching
            self.inference = PyTorchInference(
                model, max([len(tokens) for tokens in self.initial_tokens])
            )

            # sequence ranker: implements how to rank a group of sampled sequences
            self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

            # logit filters: applies various rules to suppress or penalize certain tokens
            self.decoder = []
            self.logit_filters = [[]] * len(self.initial_tokens)
            for i in range(len(self.initial_tokens)):
                # decoder: implements how to select the next tokens, given the autoregressive distribution
                if options.beam_size is not None:
                    self.decoder.append(
                        BeamSearchDecoder(
                            options.beam_size,
                            self.tokenizers[i].eot,
                            self.inference,
                            options.patience,
                        )
                    )
                else:
                    self.decoder.append(
                        GreedyDecoder(options.temperature, self.tokenizers[i].eot)
                    )

                if self.options.suppress_blank:
                    self.logit_filters[i].append(
                        SuppressBlank(self.tokenizers[i], self.sample_begin[i])
                    )
                if self.options.suppress_tokens:
                    self.logit_filters[i].append(
                        SuppressTokens(self._get_suppress_tokens(self.tokenizers[i]))
                    )
                if not options.without_timestamps:
                    precision = (
                        CHUNK_LENGTH / model.dims.n_audio_ctx
                    )  # usually 0.02 seconds
                    max_initial_timestamp_index = None
                    if options.max_initial_timestamp:
                        max_initial_timestamp_index = round(
                            self.options.max_initial_timestamp / precision
                        )
                    self.logit_filters[i].append(
                        ApplyTimestampRules(
                            self.tokenizers[i],
                            self.sample_begin[i],
                            max_initial_timestamp_index,
                        )
                    )
        else:
            if self.options.without_timestamps:
                self.sot_sequence = tokenizer.sot_sequence_including_notimestamps
            else:
                self.sot_sequence: Tuple[int] = tokenizer.sot_sequence

            # self.initial_tokens: Tuple[int] = self._get_initial_tokens()
            self.initial_tokens = self._get_initial_tokens()

            self.sample_begin: int = len(self.initial_tokens)
            self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

            # inference: implements the forward pass through the decoder, including kv caching
            self.inference = PyTorchInference(model, len(self.initial_tokens))

            # sequence ranker: implements how to rank a group of sampled sequences
            self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

            # decoder: implements how to select the next tokens, given the autoregressive distribution
            if options.beam_size is not None:
                self.decoder = BeamSearchDecoder(
                    options.beam_size, tokenizer.eot, self.inference, options.patience
                )
            else:
                self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

            # logit filters: applies various rules to suppress or penalize certain tokens
            if self.options.suppress_blank:
                self.logit_filters.append(
                    SuppressBlank(self.tokenizer, self.sample_begin)
                )
            if self.options.suppress_tokens:
                self.logit_filters.append(
                    SuppressTokens(self._get_suppress_tokens(self.tokenizer))
                )
            if not options.without_timestamps:
                precision = (
                    CHUNK_LENGTH / model.dims.n_audio_ctx
                )  # usually 0.02 seconds
                max_initial_timestamp_index = None
                if options.max_initial_timestamp:
                    max_initial_timestamp_index = round(
                        self.options.max_initial_timestamp / precision
                    )
                self.logit_filters.append(
                    ApplyTimestampRules(
                        tokenizer, self.sample_begin, max_initial_timestamp_index
                    )
                )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt
        if (len(prompt) >= 1) and isinstance(prompt[0], list):
            # branch to batched version if prompt is a list of prompts
            return self._get_batched_initial_tokens()

        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_batched_initial_tokens(self) -> List[Tuple[int]]:
        tokens = self.sot_sequence
        prefixes = self.options.prefix
        if not isinstance(prefixes, list):
            prefixes = [prefixes] * len(self.options.prompt)
        prompts = self.options.prompt
        min_prompt_len = min([len(prompt) for prompt in prompts])

        # res_tokens = [tokens]*len(prompts)
        res_tokens = tokens
        for i, (prefix, prompt) in enumerate(list(zip(prefixes, prompts))):
            if prefix:
                prefix_tokens = (
                    self.tokenizers[i].encode(" " + prefix.strip())
                    if isinstance(prefix, str)
                    else prefix
                )
                if self.sample_len is not None:
                    max_prefix_len = self.n_ctx // 2 - self.sample_len
                    prefix_tokens = prefix_tokens[-max_prefix_len:]
                res_tokens[i] = res_tokens[i] + prefix_tokens

            if prompt:
                prompt_tokens = (
                    self.tokenizers[i].encode(" " + prompt.strip())
                    if isinstance(prompt, str)
                    else prompt
                )
                # truncate longer prompts to the length of the shortest prompt
                prompt_tokens = prompt_tokens[-min_prompt_len:]
                res_tokens[i] = (
                    [self.tokenizers[i].sot_prev]
                    + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                    + list(res_tokens[i])
                )
        return [tuple(tokens) for tokens in res_tokens]

    def _get_suppress_tokens(self, tokenizer: Tokenizer) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend([tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm])
        if tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                if isinstance(self.sot_index, list):
                    for i in range(tokens.shape[0]):
                        tokens[i, self.sot_index[i] + 1] = lang_tokens[
                            i
                        ]  # write language tokens
                else:
                    tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                # NOTE: **********Here is the model inference****************
                logits = self.inference.logits(tokens, audio_features)

                if isinstance(self.sot_index, list):
                    if i == 0 and not any(
                        isinstance(tok.no_speech, type(None)) for tok in self.tokenizers
                    ):  # save no_speech_probs
                        probs_at_sot = []
                        no_speech_probs = []
                        for i in range(len(self.sot_index)):
                            probs_at_sot.append(
                                logits[:, self.sot_index[i]].float().softmax(dim=-1)
                            )
                            no_speech_probs.append(
                                probs_at_sot[i][
                                    :, self.tokenizers[i].no_speech
                                ].tolist()
                            )
                else:
                    if (
                        i == 0 and self.tokenizer.no_speech is not None
                    ):  # save no_speech_probs
                        probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                        no_speech_probs = probs_at_sot[
                            :, self.tokenizer.no_speech
                        ].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                if (len(self.logit_filters) > 0) and (
                    isinstance(self.logit_filters[0], list)
                ):
                    # for batched case
                    for i, logit_filter_group in enumerate(self.logit_filters):
                        for logit_filter in logit_filter_group:
                            logit_filter.apply(
                                logits[i].unsqueeze(0), tokens[i].unsqueeze(0)
                            )
                elif len(self.logit_filters) > 0:
                    for logit_filter in self.logit_filters:
                        logit_filter.apply(logits, tokens)

                if isinstance(self.decoder, list):
                    # handle batched case
                    completed = []
                    new_tokens = []
                    for i in range(len(self.decoder)):
                        # expand the tokens tensor with the selected next tokens
                        token_slice, comp = self.decoder[i].update(
                            tokens[i].unsqueeze(0),
                            logits[i].unsqueeze(0),
                            sum_logprobs[i].unsqueeze(0),
                        )
                        new_tokens.append(token_slice)
                        completed.append(comp)
                    tokens = torch.cat(new_tokens, dim=0)
                else:
                    # expand the tokens tensor with the selected next tokens
                    tokens, completed = self.decoder.update(
                        tokens, logits, sum_logprobs
                    )

                if isinstance(completed, list):
                    completed = all(completed)
                    if completed or tokens.shape[-1] > self.n_ctx:
                        break
                else:
                    if completed or tokens.shape[-1] > self.n_ctx:
                        break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, audio_features: Tensor) -> List[DecodingResult]:
        if isinstance(self.decoder, list):
            _ = [decoder.reset() for decoder in self.decoder]
            tokenizer: List[Tokenizer] = self.tokenizers
        else:
            self.decoder.reset()
            tokenizer: Tokenizer = self.tokenizer
        n_audio: int = audio_features.shape[0]

        if isinstance(self.initial_tokens, list):
            # if batched, then stack prompts together in batch dimension
            tokens = [list(token) for token in self.initial_tokens]
            min_len = min([len(t) for t in tokens])
            tokens = [t[:min_len] for t in tokens]
            tokens: Tensor = torch.tensor(tokens)
        else:
            tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)

        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]
        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        if isinstance(self.decoder, list):
            new_tokens = []
            sum_logprobs_new = []
            for i in range(len(self.decoder)):
                # get the final candidates for each group, and slice between the first sampled token and EOT
                token_slice, sum_logprob_slice = self.decoder[i].finalize(
                    tokens[i].unsqueeze(0), sum_logprobs[i].unsqueeze(0)
                )
                new_tokens.append(token_slice)
                sum_logprobs_new.append(sum_logprob_slice[0])
            tokens = torch.cat(new_tokens, dim=0)
            sum_logprobs = sum_logprobs_new
        else:
            # get the final candidates for each group, and slice between the first sampled token and EOT
            tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)

        if isinstance(self.sample_begin, list):
            tokens: List[List[Tensor]] = [
                [
                    t[self.sample_begin[i] : (t == tokenizer[i].eot).nonzero()[0, 0]]
                    for t in s
                ]
                for i, s in enumerate(tokens)
            ]
        else:
            tokens: List[List[Tensor]] = [
                [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
                for s in tokens
            ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        if isinstance(tokenizer, list):
            texts: List[str] = [
                tokenizer[i].decode(t).strip() for i, t in enumerate(tokens)
            ]
        else:
            texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


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
