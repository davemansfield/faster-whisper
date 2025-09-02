import itertools
import json
import logging
import os
import zlib

from dataclasses import asdict, dataclass
from inspect import signature
from math import ceil
from typing import Any, BinaryIO, Iterable, List, Optional, Tuple, Union
from warnings import warn

import ctranslate2
import numpy as np
import tokenizers
import soundfile as sf

from tqdm import tqdm

from faster_whisper.audio import decode_audio, pad_or_trim
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer
from faster_whisper.utils import download_model, format_timestamp, get_end, get_logger
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
    merge_segments,
)

#impprt tools from preprocess_pipeline package
from preprocess_pipeline.utils import calculate_all_metrics
from preprocess_pipeline.enhancement import Enhancer
from preprocess_pipeline.separation import Separator
from preprocess_pipeline.diarization import Diarizer
import torch
import gc
import os
import tempfile


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float

    def _asdict(self):
        warn(
            "Word._asdict() method is deprecated, use dataclasses.asdict(Word) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    temperature: Optional[float]

    def _asdict(self):
        warn(
            "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


@dataclass
class TranscriptionOptions:
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: List[float]
    initial_prompt: Optional[Union[str, Iterable[int]]]
    prefix: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    multilingual: bool
    max_new_tokens: Optional[int]
    clip_timestamps: Union[str, List[float]]
    hallucination_silence_threshold: Optional[float]
    hotwords: Optional[str]
    # Hallucination recovery parameters
    enable_hallucination_recovery: bool = False
    max_beam_size: int = 7
    min_beam_size: int = 3
    ngram_threshold: float = 20.0
    compression_threshold: float = 2.0
    red_light: bool = False
    yellow_light: bool = False
    green_light: bool = False


@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions


class BatchedInferencePipeline:
    def __init__(
        self,
        model,
    ):
        self.model: WhisperModel = model
        self.last_speech_timestamp = 0.0

    def forward(self, features, tokenizer, chunks_metadata, options):
        encoder_output, outputs = self.generate_segment_batched(
            features, tokenizer, options
        )

        segmented_outputs = []
        segment_sizes = []
        for chunk_metadata, output in zip(chunks_metadata, outputs):
            duration = chunk_metadata["end_time"] - chunk_metadata["start_time"]
            segment_size = int(ceil(duration) * self.model.frames_per_second)
            segment_sizes.append(segment_size)
            (
                subsegments,
                seek,
                single_timestamp_ending,
            ) = self.model._split_segments_by_timestamps(
                tokenizer=tokenizer,
                tokens=output["tokens"],
                time_offset=chunk_metadata["start_time"],
                segment_size=segment_size,
                segment_duration=duration,
                seek=0,
            )
            segmented_outputs.append(
                [
                    dict(
                        text=tokenizer.decode(subsegment["tokens"]),
                        avg_logprob=output["avg_logprob"],
                        no_speech_prob=output["no_speech_prob"],
                        tokens=subsegment["tokens"],
                        start=subsegment["start"],
                        end=subsegment["end"],
                        compression_ratio=get_compression_ratio(
                            tokenizer.decode(subsegment["tokens"])
                        ),
                        seek=int(
                            chunk_metadata["start_time"] * self.model.frames_per_second
                        ),
                    )
                    for subsegment in subsegments
                ]
            )
        if options.word_timestamps:
            self.last_speech_timestamp = self.model.add_word_timestamps(
                segmented_outputs,
                tokenizer,
                encoder_output,
                segment_sizes,
                options.prepend_punctuations,
                options.append_punctuations,
                self.last_speech_timestamp,
            )

        return segmented_outputs

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ):
        batch_size = features.shape[0]

        prompt = self.model.get_prompt(
            tokenizer,
            previous_tokens=(
                tokenizer.encode(options.initial_prompt)
                if options.initial_prompt is not None
                else []
            ),
            without_timestamps=options.without_timestamps,
            hotwords=options.hotwords,
        )

        if options.max_new_tokens is not None:
            max_length = len(prompt) + options.max_new_tokens
        else:
            max_length = self.model.max_length

        if max_length > self.model.max_length:
            raise ValueError(
                f"The length of the prompt is {len(prompt)}, and the `max_new_tokens` "
                f"{max_length - len(prompt)}. Thus, the combined length of the prompt "
                f"and `max_new_tokens` is: {max_length}. This exceeds the "
                f"`max_length` of the Whisper model: {self.model.max_length}. "
                "You should either reduce the length of your prompt, or "
                "reduce the value of `max_new_tokens`, "
                f"so that their combined length is less that {self.model.max_length}."
            )

        encoder_output = self.model.encode(features)
        prompts = [prompt.copy() for _ in range(batch_size)]

        if options.multilingual:
            language_tokens = [
                tokenizer.tokenizer.token_to_id(segment_langs[0][0])
                for segment_langs in self.model.model.detect_language(encoder_output)
            ]
            language_token_index = prompt.index(tokenizer.language)

            for i, language_token in enumerate(language_tokens):
                prompts[i][language_token_index] = language_token

        results = self.model.model.generate(
            encoder_output,
            prompts,
            beam_size=options.beam_size,
            patience=options.patience,
            length_penalty=options.length_penalty,
            max_length=max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
            return_scores=True,
            return_no_speech_prob=True,
            sampling_temperature=options.temperatures[0],
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
        )

        output = []
        for result in results:
            # return scores
            seq_len = len(result.sequences_ids[0])
            cum_logprob = result.scores[0] * (seq_len**options.length_penalty)

            output.append(
                dict(
                    avg_logprob=cum_logprob / (seq_len + 1),
                    no_speech_prob=result.no_speech_prob,
                    tokens=result.sequences_ids[0],
                )
            )

        return encoder_output, output

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = True,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        multilingual: bool = False,
        vad_filter: bool = True,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        clip_timestamps: Optional[List[dict]] = None,
        hallucination_silence_threshold: Optional[float] = None,
        batch_size: int = 8,
        hotwords: Optional[str] = None,
        language_detection_threshold: Optional[float] = 0.5,
        language_detection_segments: int = 1,
        # Hallucination recovery parameters
        enable_hallucination_recovery: bool = False,
        max_beam_size: int = 7,
        min_beam_size: int = 3,
        ngram_threshold: float = 20.0,
        compression_threshold: float = 2.0,
        red_light: bool = False,
        yellow_light: bool = False,
        green_light: bool = False,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """transcribe audio in chunks in batched fashion and return with language info.

        Arguments:
            audio: Path to the input file (or a file-like object), or the audio waveform.
            language: The language spoken in the audio. It should be a language code such
                as "en" or "fr". If not set, the language will be detected in the first 30 seconds
                of audio.
            task: Task to execute (transcribe or translate).
            log_progress: whether to show progress bar or not.
            beam_size: Beam size to use for decoding.
            best_of: Number of candidates when sampling with non-zero temperature.
            patience: Beam search patience factor.
            length_penalty: Exponential length penalty constant.
            repetition_penalty: Penalty applied to the score of previously generated tokens
                (set > 1 to penalize).
            no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
            temperature: Temperature for sampling. If a list or tuple is passed,
                only the first value is used.
            initial_prompt: Optional text string or iterable of token ids to provide as a
                prompt for the each window.
            suppress_blank: Suppress blank outputs at the beginning of the sampling.
            suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                of symbols as defined in `tokenizer.non_speech_tokens()`.
            without_timestamps: Only sample text tokens.
            word_timestamps: Extract word-level timestamps using the cross-attention pattern
                and dynamic time warping, and include the timestamps for each word in each segment.
                Set as False.
            prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
                with the next word
            append_punctuations: If word_timestamps is True, merge these punctuation symbols
                with the previous word
            multilingual: Perform language detection on every segment.
            vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
                without speech. This step is using the Silero VAD model
                https://github.com/snakers4/silero-vad.
            vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
                parameters and default values in the class `VadOptions`).
            max_new_tokens: Maximum number of new tokens to generate per-chunk. If not set,
                the maximum will be set by the default max_length.
            chunk_length: The length of audio segments. If it is not None, it will overwrite the
                default chunk_length of the FeatureExtractor.
            clip_timestamps: Optionally provide list of dictionaries each containing "start" and
                "end" keys that specify the start and end of the voiced region within
                `chunk_length` boundary. vad_filter will be ignored if clip_timestamps is used.
            batch_size: the maximum number of parallel requests to model for decoding.
            hotwords:
                Hotwords/hint phrases to the model. Has no effect if prefix is not None.
            language_detection_threshold: If the maximum probability of the language tokens is
                higher than this value, the language is detected.
            language_detection_segments: Number of segments to consider for the language detection.

        Unused Arguments
            compression_ratio_threshold: If the gzip compression ratio is above this value,
                treat as failed.
            log_prob_threshold: If the average log probability over sampled tokens is
                below this value, treat as failed.
            no_speech_threshold: If the no_speech probability is higher than this value AND
                the average log probability over sampled tokens is below `log_prob_threshold`,
                consider the segment as silent.
            condition_on_previous_text: If True, the previous output of the model is provided
                as a prompt for the next window; disabling may make the text inconsistent across
                windows, but the model becomes less prone to getting stuck in a failure loop,
                such as repetition looping or timestamps going out of sync. Set as False
            prompt_reset_on_temperature: Resets prompt if temperature is above this value.
                Arg has effect only if condition_on_previous_text is True. Set at 0.5
            prefix: Optional text to provide as a prefix at the beginning of each window.
            max_initial_timestamp: The initial timestamp cannot be later than this, set at 0.0.
            hallucination_silence_threshold: Optional[float]
                When word_timestamps is True, skip silent periods longer than this threshold
                (in seconds) when a possible hallucination is detected. set as None.
        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """

        sampling_rate = self.model.feature_extractor.sampling_rate

        if multilingual and not self.model.model.is_multilingual:
            self.model.logger.warning(
                "The current model is English-only but the multilingual parameter is set to"
                "True; setting to False instead."
            )
            multilingual = False

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)
        duration = audio.shape[0] / sampling_rate



        self.model.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )

        chunk_length = chunk_length or self.model.feature_extractor.chunk_length
        # if no segment split is provided, use vad_model and generate segments
        if not clip_timestamps:
            if vad_filter:
                if vad_parameters is None:
                    vad_parameters = VadOptions(
                        max_speech_duration_s=chunk_length,
                        min_silence_duration_ms=160,
                    )
                elif isinstance(vad_parameters, dict):
                    if "max_speech_duration_s" in vad_parameters.keys():
                        vad_parameters.pop("max_speech_duration_s")

                    vad_parameters = VadOptions(
                        **vad_parameters, max_speech_duration_s=chunk_length
                    )

                active_segments = get_speech_timestamps(audio, vad_parameters)
                clip_timestamps = merge_segments(active_segments, vad_parameters)
            # run the audio if it is less than 30 sec even without clip_timestamps
            elif duration < chunk_length:
                clip_timestamps = [{"start": 0, "end": audio.shape[0]}]
            else:
                raise RuntimeError(
                    "No clip timestamps found. "
                    "Set 'vad_filter' to True or provide 'clip_timestamps'."
                )

        duration_after_vad = (
            sum((segment["end"] - segment["start"]) for segment in clip_timestamps)
            / sampling_rate
        )

        self.model.logger.info(
            "VAD filter removed %s of audio",
            format_timestamp(duration - duration_after_vad),
        )

        audio_chunks, chunks_metadata = collect_chunks(audio, clip_timestamps)
        features = (
            [self.model.feature_extractor(chunk)[..., :-1] for chunk in audio_chunks]
            if duration_after_vad
            else []
        )

        all_language_probs = None
        # detecting the language if not provided
        if language is None:
            if not self.model.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                (
                    language,
                    language_probability,
                    all_language_probs,
                ) = self.model.detect_language(
                    features=np.concatenate(
                        features
                        + [
                            np.full((self.model.model.n_mels, 1), -1.5, dtype="float32")
                        ],
                        axis=1,
                    ),  # add a dummy feature to account for empty audio
                    language_detection_segments=language_detection_segments,
                    language_detection_threshold=language_detection_threshold,
                )

                self.model.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.model.is_multilingual and language != "en":
                self.model.logger.warning(
                    "The current model is English-only but the language parameter is set to '%s'; "
                    "using 'en' instead." % language
                )
                language = "en"

            language_probability = 1

        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task=task,
            language=language,
        )

        features = (
            np.stack([pad_or_trim(feature) for feature in features]) if features else []
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            temperatures=(
                temperature[:1]
                if isinstance(temperature, (list, tuple))
                else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                get_suppressed_tokens(tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            multilingual=multilingual,
            max_new_tokens=max_new_tokens,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
            enable_hallucination_recovery=enable_hallucination_recovery,
            max_beam_size=max_beam_size,
            min_beam_size=min_beam_size,
            ngram_threshold=ngram_threshold,
            compression_threshold=compression_threshold,
            red_light=red_light,
            yellow_light=yellow_light,
            green_light=green_light,
        )

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )

        segments = self._batched_segments_generator(
            features,
            tokenizer,
            chunks_metadata,
            batch_size,
            options,
            log_progress,
        )

        return segments, info

    def _batched_segments_generator(
        self, features, tokenizer, chunks_metadata, batch_size, options, log_progress
    ):
        pbar = tqdm(total=len(features), disable=not log_progress, position=0)
        seg_idx = 0
        for i in range(0, len(features), batch_size):
            results = self.forward(
                features[i : i + batch_size],
                tokenizer,
                chunks_metadata[i : i + batch_size],
                options,
            )

            for result in results:
                for segment in result:
                    seg_idx += 1
                    yield Segment(
                        seek=segment["seek"],
                        id=seg_idx,
                        text=segment["text"],
                        start=round(segment["start"], 3),
                        end=round(segment["end"], 3),
                        words=(
                            None
                            if not options.word_timestamps
                            else [Word(**word) for word in segment["words"]]
                        ),
                        tokens=segment["tokens"],
                        avg_logprob=segment["avg_logprob"],
                        no_speech_prob=segment["no_speech_prob"],
                        compression_ratio=segment["compression_ratio"],
                        temperature=options.temperatures[0],
                    )

                pbar.update(1)

        pbar.close()
        self.last_speech_timestamp = 0.0


class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        files: dict = None,
        revision: Optional[str] = None,
        use_auth_token: Optional[Union[str, bool]] = None,
        **model_kwargs,
    ):
        """Initializes the Whisper model.

        Args:
          model_size_or_path: Size of the model to use (tiny, tiny.en, base, base.en,
            small, small.en, distil-small.en, medium, medium.en, distil-medium.en, large-v1,
            large-v2, large-v3, large, distil-large-v2, distil-large-v3, large-v3-turbo, or turbo),
            a path to a converted model directory, or a CTranslate2-converted Whisper model ID from
            the HF Hub. When a size or a model ID is configured, the converted model is downloaded
            from the Hugging Face Hub.
          device: Device to use for computation ("cpu", "cuda", "auto").
          device_index: Device ID to use.
            The model can also be loaded on multiple GPUs by passing a list of IDs
            (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel
            when transcribe() is called from multiple Python threads (see also num_workers).
          compute_type: Type to use for computation.
            See https://opennmt.net/CTranslate2/quantization.html.
          cpu_threads: Number of threads to use when running on CPU (4 by default).
            A non zero value overrides the OMP_NUM_THREADS environment variable.
          num_workers: When transcribe() is called from multiple Python threads,
            having multiple workers enables true parallelism when running the model
            (concurrent calls to self.model.generate() will run in parallel).
            This can improve the global throughput at the cost of increased memory usage.
          download_root: Directory where the models should be saved. If not set, the models
            are saved in the standard Hugging Face cache directory.
          local_files_only:  If True, avoid downloading the file and return the path to the
            local cached file if it exists.
          files: Load model files from the memory. This argument is a dictionary mapping file names
            to file contents as file-like or bytes objects. If this is set, model_path acts as an
            identifier for this model.
          revision:
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash.
          use_auth_token: HuggingFace authentication token or True to use the
            token stored by the HuggingFace config folder.
        """
        self.logger = get_logger()

        tokenizer_bytes, preprocessor_bytes = None, None
        if files:
            model_path = model_size_or_path
            tokenizer_bytes = files.pop("tokenizer.json", None)
            preprocessor_bytes = files.pop("preprocessor_config.json", None)
        elif os.path.isdir(model_size_or_path):
            model_path = model_size_or_path
        else:
            model_path = download_model(
                model_size_or_path,
                local_files_only=local_files_only,
                cache_dir=download_root,
                revision=revision,
                use_auth_token=use_auth_token,
            )

        self.model = ctranslate2.models.Whisper(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            intra_threads=cpu_threads,
            inter_threads=num_workers,
            files=files,
            **model_kwargs,
        )

        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        if tokenizer_bytes:
            self.hf_tokenizer = tokenizers.Tokenizer.from_buffer(tokenizer_bytes)
        elif os.path.isfile(tokenizer_file):
            self.hf_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        else:
            self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
                "openai/whisper-tiny" + ("" if self.model.is_multilingual else ".en")
            )
        self.feat_kwargs = self._get_feature_kwargs(model_path, preprocessor_bytes)
        self.feature_extractor = FeatureExtractor(**self.feat_kwargs)
        self.input_stride = 2
        self.num_samples_per_token = (
            self.feature_extractor.hop_length * self.input_stride
        )
        self.frames_per_second = (
            self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
        )
        self.tokens_per_second = (
            self.feature_extractor.sampling_rate // self.num_samples_per_token
        )
        self.time_precision = 0.02
        self.max_length = 448

    @property
    def supported_languages(self) -> List[str]:
        """The languages supported by the model."""
        return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

    def _get_feature_kwargs(self, model_path, preprocessor_bytes=None) -> dict:
        config = {}
        try:
            config_path = os.path.join(model_path, "preprocessor_config.json")
            if preprocessor_bytes:
                config = json.loads(preprocessor_bytes)
            elif os.path.isfile(config_path):
                with open(config_path, "r", encoding="utf-8") as file:
                    config = json.load(file)
            else:
                return config
            valid_keys = signature(FeatureExtractor.__init__).parameters.keys()
            return {k: v for k, v in config.items() if k in valid_keys}
        except json.JSONDecodeError as e:
            self.logger.warning("Could not load preprocessor config: %s", e)

        return config

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        multilingual: bool = False,
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        hotwords: Optional[str] = None,
        language_detection_threshold: Optional[float] = 0.5,
        language_detection_segments: int = 1,
        # Hallucination recovery parameters
        enable_hallucination_recovery: bool = False,
        max_beam_size: int = 7,
        min_beam_size: int = 3,
        ngram_threshold: float = 20.0,
        compression_threshold: float = 2.0,
        red_light: bool = False,
        yellow_light: bool = False,
        green_light: bool = False,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribes an input file.

        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform.
          language: The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.
          task: Task to execute (transcribe or translate).
          log_progress: whether to show progress bar or not.
          beam_size: Beam size to use for decoding.
          best_of: Number of candidates when sampling with non-zero temperature.
          patience: Beam search patience factor.
          length_penalty: Exponential length penalty constant.
          repetition_penalty: Penalty applied to the score of previously generated tokens
            (set > 1 to penalize).
          no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `log_prob_threshold`.
          compression_ratio_threshold: If the gzip compression ratio is above this value,
            treat as failed.
          log_prob_threshold: If the average log probability over sampled tokens is
            below this value, treat as failed.
          no_speech_threshold: If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
          condition_on_previous_text: If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.
          prompt_reset_on_temperature: Resets prompt if temperature is above this value.
            Arg has effect only if condition_on_previous_text is True.
          initial_prompt: Optional text string or iterable of token ids to provide as a
            prompt for the first window.
          prefix: Optional text to provide as a prefix for the first window.
          suppress_blank: Suppress blank outputs at the beginning of the sampling.
          suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
            of symbols as defined in `tokenizer.non_speech_tokens()`.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.
          word_timestamps: Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.
          prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the next word
          append_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the previous word
          multilingual: Perform language detection on every segment.
          vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model
            https://github.com/snakers4/silero-vad.
          vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
            parameters and default values in the class `VadOptions`).
          max_new_tokens: Maximum number of new tokens to generate per-chunk. If not set,
            the maximum will be set by the default max_length.
          chunk_length: The length of audio segments. If it is not None, it will overwrite the
            default chunk_length of the FeatureExtractor.
          clip_timestamps:
            Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to
             process. The last end timestamp defaults to the end of the file.
             vad_filter will be ignored if clip_timestamps is used.
          hallucination_silence_threshold:
            When word_timestamps is True, skip silent periods longer than this threshold
             (in seconds) when a possible hallucination is detected
          hotwords:
            Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.
          language_detection_threshold: If the maximum probability of the language tokens is higher
           than this value, the language is detected.
          language_detection_segments: Number of segments to consider for the language detection.
        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """
        sampling_rate = self.feature_extractor.sampling_rate


        if multilingual and not self.model.is_multilingual:
            self.logger.warning(
                "The current model is English-only but the multilingual parameter is set to"
                "True; setting to False instead."
            )
            multilingual = False

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        # Store raw audio for potential enhancement
        self.raw_audio = audio
        self.sampling_rate = sampling_rate

        duration = audio.shape[0] / sampling_rate
        duration_after_vad = duration

        self.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )

        if vad_filter and clip_timestamps == "0":
            if vad_parameters is None:
                vad_parameters = VadOptions()
            elif isinstance(vad_parameters, dict):
                vad_parameters = VadOptions(**vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_parameters)
            audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
            audio = np.concatenate(audio_chunks, axis=0)
            duration_after_vad = audio.shape[0] / sampling_rate

            self.logger.info(
                "VAD filter removed %s of audio",
                format_timestamp(duration - duration_after_vad),
            )

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "VAD filter kept the following audio segments: %s",
                    ", ".join(
                        "[%s -> %s]"
                        % (
                            format_timestamp(chunk["start"] / sampling_rate),
                            format_timestamp(chunk["end"] / sampling_rate),
                        )
                        for chunk in speech_chunks
                    ),
                )

        else:
            speech_chunks = None

        features = self.feature_extractor(audio, chunk_length=chunk_length)

        encoder_output = None
        all_language_probs = None

        # detecting the language if not provided
        if language is None:
            if not self.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                start_timestamp = (
                    float(clip_timestamps.split(",")[0])
                    if isinstance(clip_timestamps, str)
                    else clip_timestamps[0]
                )
                content_frames = features.shape[-1] - 1
                seek = (
                    int(start_timestamp * self.frames_per_second)
                    if start_timestamp * self.frames_per_second < content_frames
                    else 0
                )
                (
                    language,
                    language_probability,
                    all_language_probs,
                ) = self.detect_language(
                    features=features[..., seek:],
                    language_detection_segments=language_detection_segments,
                    language_detection_threshold=language_detection_threshold,
                )

                self.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.is_multilingual and language != "en":
                self.logger.warning(
                    "The current model is English-only but the language parameter is set to '%s'; "
                    "using 'en' instead." % language
                )
                language = "en"

            language_probability = 1

        tokenizer = Tokenizer(
            self.hf_tokenizer,
            self.model.is_multilingual,
            task=task,
            language=language,
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                get_suppressed_tokens(tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            multilingual=multilingual,
            max_new_tokens=max_new_tokens,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
            enable_hallucination_recovery=enable_hallucination_recovery,
            max_beam_size=max_beam_size,
            min_beam_size=min_beam_size,
            ngram_threshold=ngram_threshold,
            compression_threshold=compression_threshold,
            red_light=red_light,
            yellow_light=yellow_light,
            green_light=green_light,
        )

        segments = self.generate_segments(
            features, tokenizer, options, log_progress, encoder_output
        )

        if speech_chunks:
            segments = restore_speech_timestamps(segments, speech_chunks, sampling_rate)

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )

        return segments, info

    def _split_segments_by_timestamps(
        self,
        tokenizer: Tokenizer,
        tokens: List[int],
        time_offset: float,
        segment_size: int,
        segment_duration: float,
        seek: int,
    ) -> List[List[int]]:
        current_segments = []
        single_timestamp_ending = (
            len(tokens) >= 2 and tokens[-2] < tokenizer.timestamp_begin <= tokens[-1]
        )

        consecutive_timestamps = [
            i
            for i in range(len(tokens))
            if i > 0
            and tokens[i] >= tokenizer.timestamp_begin
            and tokens[i - 1] >= tokenizer.timestamp_begin
        ]

        if len(consecutive_timestamps) > 0:
            slices = list(consecutive_timestamps)
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = sliced_tokens[0] - tokenizer.timestamp_begin
                end_timestamp_position = sliced_tokens[-1] - tokenizer.timestamp_begin
                start_time = (
                    time_offset + start_timestamp_position * self.time_precision
                )
                end_time = time_offset + end_timestamp_position * self.time_precision

                current_segments.append(
                    dict(
                        seek=seek,
                        start=start_time,
                        end=end_time,
                        tokens=sliced_tokens,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                seek += segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                last_timestamp_position = (
                    tokens[last_slice - 1] - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * self.input_stride

        else:
            duration = segment_duration
            timestamps = [
                token for token in tokens if token >= tokenizer.timestamp_begin
            ]
            if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
                last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                duration = last_timestamp_position * self.time_precision

            current_segments.append(
                dict(
                    seek=seek,
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                )
            )

            seek += segment_size

        return current_segments, seek, single_timestamp_ending

    def _attempt_hallucination_recovery(
        self,
        encoder_output,
        prompt,
        tokenizer,
        options,
        modified_options,
        current_audio_chunk,
        seek,
        segment_size,
        result,
        avg_logprob,
        temperature,
        compression_ratio,
        text):


        """
        Attempt to recover from hallucination using available strategies.
        
        Returns: (result, avg_logprob, temperature, compression_ratio, text, success)
        """
        #define available recovery strategies based on light settings
        available_strategies = []
        
        if options.green_light:
            available_strategies.extend([
                ("increase_beam_size", 0),
                ("decrease_beam_size", 1), 
                ("drop_beam_size", 2),
                ("vad_cleaning", 3)
            ])
        
        if options.yellow_light:
            available_strategies.append(("audio_enhancement", 4))
        
        if options.red_light:
            available_strategies.append(("speaker_separation", 5))
        
        if not available_strategies:
            print("No recovery strategies enabled - skipping hallucination recovery")
            return result, avg_logprob, temperature, compression_ratio, text, False
        
        print(f"Available recovery strategies: {[s[0] for s in available_strategies]}")
        
        #loop until hallucination is fixed or all available strategies are exhausted
        strategy_index = 0
        while strategy_index < len(available_strategies):
            duration = self.get_segment_duration(seek, segment_size)
            #check if segment is too short (less than 1 second)
            if duration < 1.0:
                print(f"Skipping chunk at seek {seek} - duration too short: {duration:.2f}s likely to cause hallucination")
                return result, avg_logprob, temperature, compression_ratio, text, False

            strategy_name, attempt_num = available_strategies[strategy_index]
            print(f"Attempting recovery strategy {strategy_index + 1}/{len(available_strategies)}: {strategy_name}")
            
            #apply recovery strategy
            try:
                if strategy_name == "increase_beam_size":
                    result_, avg_logprob_, temperature_, compression_ratio_ = self._recovery_increase_beam_size(
                        encoder_output, prompt, tokenizer, modified_options
                    )
                elif strategy_name == "decrease_beam_size":
                    result_, avg_logprob_, temperature_, compression_ratio_ = self._recovery_decrease_beam_size(
                        encoder_output, prompt, tokenizer, modified_options
                    )
                elif strategy_name == "drop_beam_size":
                    result_, avg_logprob_, temperature_, compression_ratio_ = self._recovery_drop_beam_size(
                        encoder_output, prompt, tokenizer, modified_options
                    )
                elif strategy_name == "vad_cleaning":
                    result_, avg_logprob_, temperature_, compression_ratio_ = self._recovery_vad_cleaning(
                        encoder_output, prompt, tokenizer, modified_options, current_audio_chunk
                    )
                elif strategy_name == "audio_enhancement":
                    result_, avg_logprob_, temperature_, compression_ratio_ = self._recovery_audio_enhancement(
                        encoder_output, prompt, tokenizer, modified_options, current_audio_chunk
                    )
                elif strategy_name == "speaker_separation":
                    # check if separation is actually needed and get diarization result
                    needs_separation, diarization_result = self._check_for_speaker_overlap(current_audio_chunk)
                    if not needs_separation:
                        print("Speaker separation not needed, skipping this recovery attempt")
                        strategy_index += 1
                        modified_options = options  # Reset options
                        continue
                    
                    result_, avg_logprob_, temperature_, compression_ratio_ = self._recovery_speaker_separation(
                        encoder_output, prompt, tokenizer, modified_options, current_audio_chunk, diarization_result
                    )
                else:
                    print(f"Unknown strategy: {strategy_name}")
                    strategy_index += 1
                    continue
                    
            except Exception as e:
                print(f"Recovery strategy '{strategy_name}' failed: {e}")
                strategy_index += 1
                modified_options = options  # Reset options
                continue

            #check if hallucination is fixed with the current strategy
            text_ = tokenizer.decode(result_.sequences_ids[0])
            all_scores_ = calculate_all_metrics(text_)
            ngram_score_ = all_scores_['ngram_score']
            compression_rate_ = all_scores_['compression_rate']

            is_hallucination = self.detect_hallucination(
                ngram_score_, compression_rate_,
                ngram_threshold=options.ngram_threshold,
                compression_threshold=options.compression_threshold
            )
            
            if not is_hallucination:
                print(f"Hallucination fixed with {strategy_name}!")
                #ppdate result and metrics
                return result_, avg_logprob_, temperature_, compression_ratio_, text_, True
            
            #move to next strategy
            strategy_index += 1
            modified_options = options  # Reset options between attempts
        
        #if tried all strategies and still have hallucination... 
        #TODO add spectrogram comp
        print("All available recovery strategies exhausted")
        return result, avg_logprob, temperature, compression_ratio, text, False

    def generate_segments(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        log_progress,
        encoder_output: Optional[ctranslate2.StorageView] = None,
    ) -> Iterable[Segment]:
        content_frames = features.shape[-1] - 1
        content_duration = float(content_frames * self.feature_extractor.time_per_frame)

        modified_options = TranscriptionOptions(
            beam_size=options.beam_size,
            best_of=options.best_of,
            patience=options.patience,
            length_penalty=options.length_penalty,
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
            log_prob_threshold=options.log_prob_threshold,
            no_speech_threshold=options.no_speech_threshold,
            compression_ratio_threshold=options.compression_ratio_threshold,
            condition_on_previous_text=options.condition_on_previous_text,
            prompt_reset_on_temperature=options.prompt_reset_on_temperature,
            temperatures=options.temperatures,
            initial_prompt=options.initial_prompt,
            prefix=options.prefix,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
            without_timestamps=options.without_timestamps,
            max_initial_timestamp=options.max_initial_timestamp,
            word_timestamps=options.word_timestamps,
            prepend_punctuations=options.prepend_punctuations,
            append_punctuations=options.append_punctuations,
            multilingual=options.multilingual,
            max_new_tokens=options.max_new_tokens,
            clip_timestamps=options.clip_timestamps,
            hallucination_silence_threshold=options.hallucination_silence_threshold,
            hotwords=options.hotwords,
            enable_hallucination_recovery=options.enable_hallucination_recovery,
            max_beam_size=options.max_beam_size,
            min_beam_size=options.min_beam_size,
            ngram_threshold=options.ngram_threshold,
            compression_threshold=options.compression_threshold,
            red_light=options.red_light,
            yellow_light=options.yellow_light,
            green_light=options.green_light,
        )
        
        if isinstance(options.clip_timestamps, str):
            options.clip_timestamps = [
                float(ts)
                for ts in (
                    options.clip_timestamps.split(",")
                    if options.clip_timestamps
                    else []
                )
            ]

        seek_points: List[int] = [
            round(ts * self.frames_per_second) for ts in options.clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        seek_clips: List[Tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2])
        )

        punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

        idx = 0
        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)

        pbar = tqdm(total=content_duration, unit="seconds", disable=not log_progress)
        last_speech_timestamp = 0.0
        # NOTE: This loop is obscurely flattened to make the diff readable.
        # A later commit should turn this into a simpler nested loop.
        # for seek_clip_start, seek_clip_end in seek_clips:
        #     while seek < seek_clip_end
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek_clip_end > content_frames:
                seek_clip_end = content_frames
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue

            #core segment extraction or chunking
            #seek is keeping track of what chunk we are processing
            time_offset = seek * self.feature_extractor.time_per_frame
            window_end_time = float(
                (seek + self.feature_extractor.nb_max_frames)
                * self.feature_extractor.time_per_frame
            ) #nb_max_frames is whats deciding the chunk size

            #determine size of the chunk, never longer than 30s usually shorter by audio or clip boundaries
            segment_size = min(
                self.feature_extractor.nb_max_frames,
                content_frames - seek,
                seek_clip_end - seek,
            )

            #extract mel spectrogram chunk from audio
            segment = features[:, seek : seek + segment_size]
            #calculate duration of the chunk in seconds
            segment_duration = segment_size * self.feature_extractor.time_per_frame
            #pad or trim the chunk to 3000 mel features as expected by the encoder
            segment = pad_or_trim(segment)


            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Processing segment at %s", format_timestamp(time_offset)
                )

            previous_tokens = all_tokens[prompt_reset_since:]

            #if we are at the start of the audio or we don't have an encoder output yet, encode the chunk
            if seek > 0 or encoder_output is None:
                encoder_output = self.encode(segment)

            if options.multilingual:
                results = self.model.detect_language(encoder_output)
                language_token, language_probability = results[0][0]
                language = language_token[2:-2]

                tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
                tokenizer.language_code = language

            #generate the prompt for the chunk
            prompt = self.get_prompt(
                tokenizer,
                previous_tokens,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix if seek == 0 else None,
                hotwords=options.hotwords,
            )
            

            #first generation attempt
            (result,avg_logprob,temperature,compression_ratio) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)
            text = tokenizer.decode(result.sequences_ids[0])

            if options.enable_hallucination_recovery:
                #check for halluciantion
                all_Scores=calculate_all_metrics(text)
                ngram_score=all_Scores['ngram_score']
                compression_rate=all_Scores['compression_rate']

                recovery_attempts = 0

                
                is_hallucination = self.detect_hallucination(ngram_score,compression_rate,
                    ngram_threshold=options.ngram_threshold,
                    compression_threshold=options.compression_threshold
                    )
                
                if is_hallucination:
                    # attempt hallucination recovery

                    current_audio_chunk = self.get_audio_chunk_for_segment(seek, segment_size)

                    (result, avg_logprob, temperature, compression_ratio, text, recovery_success) = self._attempt_hallucination_recovery(
                        encoder_output, prompt, tokenizer, options, modified_options, 
                        current_audio_chunk, seek, segment_size, 
                        result, avg_logprob, temperature, compression_ratio, text
                    )
                    
                    if recovery_success:
                        print("Hallucination recovery completed successfully")
                    else:
                        print("Hallucination recovery failed - using original result")
            
            if options.no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > options.no_speech_threshold

                if (
                    options.log_prob_threshold is not None
                    and avg_logprob > options.log_prob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    self.logger.debug(
                        "No speech threshold is met (%f > %f)",
                        result.no_speech_prob,
                        options.no_speech_threshold,
                    )

                    # fast-forward to the next segment boundary
                    seek += segment_size
                    continue

            tokens = result.sequences_ids[0]

            previous_seek = seek

            # anomalous words are very long/short/improbable
            def word_anomaly_score(word: dict) -> float:
                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0
                if probability < 0.15:
                    score += 1.0
                if duration < 0.133:
                    score += (0.133 - duration) * 15
                if duration > 2.0:
                    score += duration - 2.0
                return score

            def is_segment_anomaly(segment: Optional[dict]) -> bool:
                if segment is None or not segment["words"]:
                    return False
                words = [w for w in segment["words"] if w["word"] not in punctuation]
                words = words[:8]
                score = sum(word_anomaly_score(w) for w in words)
                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments: List[dict]) -> Optional[dict]:
                return next((s for s in segments if s["words"]), None)

            (
                current_segments,
                seek,
                single_timestamp_ending,
            ) = self._split_segments_by_timestamps(
                tokenizer=tokenizer,
                tokens=tokens,
                time_offset=time_offset,
                segment_size=segment_size,
                segment_duration=segment_duration,
                seek=seek,
            )

            if options.word_timestamps:
                self.add_word_timestamps(
                    [current_segments],
                    tokenizer,
                    encoder_output,
                    segment_size,
                    options.prepend_punctuations,
                    options.append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )
                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * self.frames_per_second)

                # skip silence before possible hallucinations
                if options.hallucination_silence_threshold is not None:
                    threshold = options.hallucination_silence_threshold

                    # if first segment might be a hallucination, skip leading silence
                    first_segment = next_words_segment(current_segments)
                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset
                        if gap > threshold:
                            seek = previous_seek + round(gap * self.frames_per_second)
                            continue

                    # skip silence before any possible hallucination that is surrounded
                    # by silence or more hallucinations
                    hal_last_end = last_speech_timestamp
                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]:
                            continue
                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(
                                current_segments[si + 1 :]
                            )
                            if next_segment is not None:
                                hal_next_start = next_segment["words"][0]["start"]
                            else:
                                hal_next_start = time_offset + segment_duration
                            silence_before = (
                                segment["start"] - hal_last_end > threshold
                                or segment["start"] < threshold
                                or segment["start"] - time_offset < 2.0
                            )
                            silence_after = (
                                hal_next_start - segment["end"] > threshold
                                or is_segment_anomaly(next_segment)
                                or window_end_time - segment["end"] < 2.0
                            )
                            if silence_before and silence_after:
                                seek = round(
                                    max(time_offset + 1, segment["start"])
                                    * self.frames_per_second
                                )
                                if content_duration - segment["end"] < threshold:
                                    seek = content_frames
                                current_segments[si:] = []
                                break
                        hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None:
                    last_speech_timestamp = last_word_end

            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.decode(tokens)

                if segment["start"] == segment["end"] or not text.strip():
                    continue

                all_tokens.extend(tokens)
                idx += 1

                yield Segment(
                    id=idx,
                    seek=previous_seek,
                    start=segment["start"],
                    end=segment["end"],
                    text=text,
                    tokens=tokens,
                    temperature=temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                    no_speech_prob=result.no_speech_prob,
                    words=(
                        [Word(**word) for word in segment["words"]]
                        if options.word_timestamps
                        else None
                    ),
                )

            if (
                not options.condition_on_previous_text
                or temperature > options.prompt_reset_on_temperature
            ):
                if options.condition_on_previous_text:
                    self.logger.debug(
                        "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
                        temperature,
                        options.prompt_reset_on_temperature,
                    )

                prompt_reset_since = len(all_tokens)

            pbar.update(
                (min(content_frames, seek) - previous_seek)
                * self.feature_extractor.time_per_frame,
            )
        pbar.close()

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

        if features.ndim == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

    def generate_with_fallback(
        self,
        encoder_output: ctranslate2.StorageView,
        prompt: List[int],
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ) -> Tuple[ctranslate2.models.WhisperGenerationResult, float, float, float]:
        decode_result = None
        all_results = []
        below_cr_threshold_results = []

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        if options.max_new_tokens is not None:
            max_length = len(prompt) + options.max_new_tokens
        else:
            max_length = self.max_length

        if max_length > self.max_length:
            raise ValueError(
                f"The length of the prompt is {len(prompt)}, and the `max_new_tokens` "
                f"{max_length - len(prompt)}. Thus, the combined length of the prompt "
                f"and `max_new_tokens` is: {max_length}. This exceeds the "
                f"`max_length` of the Whisper model: {self.max_length}. "
                "You should either reduce the length of your prompt, or "
                "reduce the value of `max_new_tokens`, "
                f"so that their combined length is less that {self.max_length}."
            )

        for temperature in options.temperatures:
            if temperature > 0:
                kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": options.best_of,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                kwargs = {
                    "beam_size": options.beam_size,
                    "patience": options.patience,
                }

            result = self.model.generate(
                encoder_output,
                [prompt],
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                max_length=max_length,
                return_scores=True,
                return_no_speech_prob=True,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]

            # Recover the average log prob from the returned score.
            seq_len = len(tokens)
            cum_logprob = result.scores[0] * (seq_len**options.length_penalty)
            avg_logprob = cum_logprob / (seq_len + 1)

            #full chunk here.
            text = tokenizer.decode(tokens).strip()
            compression_ratio = get_compression_ratio(text)


            decode_result = (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            )
            all_results.append(decode_result)

            needs_fallback = False

            if options.compression_ratio_threshold is not None:
                if compression_ratio > options.compression_ratio_threshold:
                    needs_fallback = True  # too repetitive

                    self.logger.debug(
                        "Compression ratio threshold is not met with temperature %.1f (%f > %f)",
                        temperature,
                        compression_ratio,
                        options.compression_ratio_threshold,
                    )
                else:
                    below_cr_threshold_results.append(decode_result)

            if (
                options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = True  # average log probability is too low

                self.logger.debug(
                    "Log probability threshold is not met with temperature %.1f (%f < %f)",
                    temperature,
                    avg_logprob,
                    options.log_prob_threshold,
                )

            if (
                options.no_speech_threshold is not None
                and result.no_speech_prob > options.no_speech_threshold
                and options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = False  # silence

            if not needs_fallback:
                break
        else:
            # all failed, select the result with the highest average log probability
            decode_result = max(
                below_cr_threshold_results or all_results, key=lambda x: x[1]
            )
            # to pass final temperature for prompt_reset_on_temperature
            decode_result = (
                decode_result[0],
                decode_result[1],
                temperature,
                decode_result[3],
            )

        return decode_result



    def detect_hallucination(self, ngram_score, compression_rate, ngram_threshold=90.0,compression_threshold=15.0):
        """
        Detect hallucination using simple threshold-based approach.
        
        Args:
            ngram_score: N-gram score from metrics
            compression_rate: Compression rate from metrics
            ngram_threshold: Threshold for n-gram score (high values indicate repetition)
            compression_threshold: Threshold for compression rate (high values indicate repetition)
            is_chunk_level: Whether this is chunk-level detection (uses different thresholds)
            
        Returns: True if hallucination detected, False otherwise
        """
        
        #hallucination is detected if EITHER threshold is exceeded
        is_hallucination = ngram_score > ngram_threshold or compression_rate > compression_threshold
        return is_hallucination

    def get_prompt(
        self,
        tokenizer: Tokenizer,
        previous_tokens: List[int],
        without_timestamps: bool = False,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> List[int]:
        prompt = []

        if previous_tokens or (hotwords and not prefix):
            prompt.append(tokenizer.sot_prev)
            if hotwords and not prefix:
                hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
                if len(hotwords_tokens) >= self.max_length // 2:
                    hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
                prompt.extend(hotwords_tokens)
            if previous_tokens:
                prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt.extend(tokenizer.sot_sequence)

        if without_timestamps:
            prompt.append(tokenizer.no_timestamps)

        if prefix:
            prefix_tokens = tokenizer.encode(" " + prefix.strip())
            if len(prefix_tokens) >= self.max_length // 2:
                prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
            if not without_timestamps:
                prompt.append(tokenizer.timestamp_begin)
            prompt.extend(prefix_tokens)

        return prompt

    def add_word_timestamps(
        self,
        segments: List[dict],
        tokenizer: Tokenizer,
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: float,
    ) -> float:
        if len(segments) == 0:
            return

        text_tokens = []
        text_tokens_per_segment = []
        for segment in segments:
            segment_tokens = [
                [token for token in subsegment["tokens"] if token < tokenizer.eot]
                for subsegment in segment
            ]
            text_tokens.append(list(itertools.chain.from_iterable(segment_tokens)))
            text_tokens_per_segment.append(segment_tokens)

        alignments = self.find_alignment(
            tokenizer, text_tokens, encoder_output, num_frames
        )
        median_max_durations = []
        for alignment in alignments:
            word_durations = np.array(
                [word["end"] - word["start"] for word in alignment]
            )
            word_durations = word_durations[word_durations.nonzero()]
            median_duration = (
                np.median(word_durations) if len(word_durations) > 0 else 0.0
            )
            median_duration = min(0.7, float(median_duration))
            max_duration = median_duration * 2

            # hack: truncate long words at sentence boundaries.
            # a better segmentation algorithm based on VAD should be able to replace this.
            if len(word_durations) > 0:
                sentence_end_marks = ".。!！?？"
                # ensure words at sentence boundaries
                # are not longer than twice the median word duration.
                for i in range(1, len(alignment)):
                    if alignment[i]["end"] - alignment[i]["start"] > max_duration:
                        if alignment[i]["word"] in sentence_end_marks:
                            alignment[i]["end"] = alignment[i]["start"] + max_duration
                        elif alignment[i - 1]["word"] in sentence_end_marks:
                            alignment[i]["start"] = alignment[i]["end"] - max_duration

            merge_punctuations(alignment, prepend_punctuations, append_punctuations)
            median_max_durations.append((median_duration, max_duration))

        for segment_idx, segment in enumerate(segments):
            word_index = 0
            time_offset = segment[0]["seek"] / self.frames_per_second
            median_duration, max_duration = median_max_durations[segment_idx]
            for subsegment_idx, subsegment in enumerate(segment):
                saved_tokens = 0
                words = []

                while word_index < len(alignments[segment_idx]) and saved_tokens < len(
                    text_tokens_per_segment[segment_idx][subsegment_idx]
                ):
                    timing = alignments[segment_idx][word_index]

                    if timing["word"]:
                        words.append(
                            dict(
                                word=timing["word"],
                                start=round(time_offset + timing["start"], 2),
                                end=round(time_offset + timing["end"], 2),
                                probability=timing["probability"],
                            )
                        )

                    saved_tokens += len(timing["tokens"])
                    word_index += 1

                # hack: truncate long words at segment boundaries.
                # a better segmentation algorithm based on VAD should be able to replace this.
                if len(words) > 0:
                    # ensure the first and second word after a pause is not longer than
                    # twice the median word duration.
                    if words[0][
                        "end"
                    ] - last_speech_timestamp > median_duration * 4 and (
                        words[0]["end"] - words[0]["start"] > max_duration
                        or (
                            len(words) > 1
                            and words[1]["end"] - words[0]["start"] > max_duration * 2
                        )
                    ):
                        if (
                            len(words) > 1
                            and words[1]["end"] - words[1]["start"] > max_duration
                        ):
                            boundary = max(
                                words[1]["end"] / 2, words[1]["end"] - max_duration
                            )
                            words[0]["end"] = words[1]["start"] = boundary
                        words[0]["start"] = max(0, words[0]["end"] - max_duration)

                    # prefer the segment-level start timestamp if the first word is too long.
                    if (
                        subsegment["start"] < words[0]["end"]
                        and subsegment["start"] - 0.5 > words[0]["start"]
                    ):
                        words[0]["start"] = max(
                            0,
                            min(words[0]["end"] - median_duration, subsegment["start"]),
                        )
                    else:
                        subsegment["start"] = words[0]["start"]

                    # prefer the segment-level end timestamp if the last word is too long.
                    if (
                        subsegment["end"] > words[-1]["start"]
                        and subsegment["end"] + 0.5 < words[-1]["end"]
                    ):
                        words[-1]["end"] = max(
                            words[-1]["start"] + median_duration, subsegment["end"]
                        )
                    else:
                        subsegment["end"] = words[-1]["end"]

                    last_speech_timestamp = subsegment["end"]
                segments[segment_idx][subsegment_idx]["words"] = words
        return last_speech_timestamp

    def find_alignment(
        self,
        tokenizer: Tokenizer,
        text_tokens: List[int],
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        median_filter_width: int = 7,
    ) -> List[dict]:
        if len(text_tokens) == 0:
            return []

        results = self.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            text_tokens,
            num_frames,
            median_filter_width=median_filter_width,
        )
        return_list = []
        for result, text_token in zip(results, text_tokens):
            text_token_probs = result.text_token_probs
            alignments = result.alignments
            text_indices = np.array([pair[0] for pair in alignments])
            time_indices = np.array([pair[1] for pair in alignments])

            words, word_tokens = tokenizer.split_to_word_tokens(
                text_token + [tokenizer.eot]
            )
            if len(word_tokens) <= 1:
                # return on eot only
                # >>> np.pad([], (1, 0))
                # array([0.])
                # This results in crashes when we lookup jump_times with float, like
                # IndexError: arrays used as indices must be of integer (or boolean) type
                return_list.append([])
                continue
            word_boundaries = np.pad(
                np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0)
            )
            if len(word_boundaries) <= 1:
                return_list.append([])
                continue

            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(
                bool
            )
            jump_times = time_indices[jumps] / self.tokens_per_second
            start_times = jump_times[word_boundaries[:-1]]
            end_times = jump_times[word_boundaries[1:]]
            word_probabilities = [
                np.mean(text_token_probs[i:j])
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
            ]

            return_list.append(
                [
                    dict(
                        word=word,
                        tokens=tokens,
                        start=start,
                        end=end,
                        probability=probability,
                    )
                    for word, tokens, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    )
                ]
            )
        return return_list

    def detect_language(
        self,
        audio: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        vad_filter: bool = False,
        vad_parameters: Union[dict, VadOptions] = None,
        language_detection_segments: int = 1,
        language_detection_threshold: float = 0.5,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Use Whisper to detect the language of the input audio or features.

        Arguments:
            audio: Input audio signal, must be a 1D float array sampled at 16khz.
            features: Input Mel spectrogram features, must be a float array with
                shape (n_mels, n_frames), if `audio` is provided, the features will be ignored.
                Either `audio` or `features` must be provided.
            vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
                without speech. This step is using the Silero VAD model.
            vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
                parameters and default values in the class `VadOptions`).
            language_detection_threshold: If the maximum probability of the language tokens is
                higher than this value, the language is detected.
            language_detection_segments: Number of segments to consider for the language detection.

        Returns:
            language: Detected language.
            languege_probability: Probability of the detected language.
            all_language_probs: List of tuples with all language names and probabilities.
        """
        assert (
            audio is not None or features is not None
        ), "Either `audio` or `features` must be provided."

        if audio is not None:
            if vad_filter:
                speech_chunks = get_speech_timestamps(audio, vad_parameters)
                audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
                audio = np.concatenate(audio_chunks, axis=0)

            audio = audio[
                : language_detection_segments * self.feature_extractor.n_samples
            ]
            features = self.feature_extractor(audio)

        features = features[
            ..., : language_detection_segments * self.feature_extractor.nb_max_frames
        ]

        detected_language_info = {}
        for i in range(0, features.shape[-1], self.feature_extractor.nb_max_frames):
            encoder_output = self.encode(
                pad_or_trim(features[..., i : i + self.feature_extractor.nb_max_frames])
            )
            # results is a list of tuple[str, float] with language names and probabilities.
            results = self.model.detect_language(encoder_output)[0]

            # Parse language names to strip out markers
            all_language_probs = [(token[2:-2], prob) for (token, prob) in results]
            # Get top language token and probability
            language, language_probability = all_language_probs[0]
            if language_probability > language_detection_threshold:
                break
            detected_language_info.setdefault(language, []).append(language_probability)
        else:
            # If no language detected for all segments, the majority vote of the highest
            # projected languages for all segments is used to determine the language.
            language = max(
                detected_language_info,
                key=lambda lang: len(detected_language_info[lang]),
            )
            language_probability = max(detected_language_info[language])

        return language, language_probability, all_language_probs

    def get_audio_chunk_for_segment(self, seek, segment_size):
        """
        Extract the exact raw audio chunk corresponding to a mel spectrogram segment.
        
        Args:
            seek: Starting mel frame index
            segment_size: Number of mel frames in the segment
            
        Returns: Raw audio chunk as numpy array
        """
        #convert mel frame indices to audio sample indices
        audio_start_sample = int(seek * self.feature_extractor.hop_length)
        audio_end_sample = int((seek + segment_size) * self.feature_extractor.hop_length)
        
        #extract the audio chunk
        audio_chunk = self.raw_audio[audio_start_sample:audio_end_sample]
        
        return audio_chunk


    def convert_audio_to_features(self, enhanced_audio_chunk):
        """
        Convert audio chunk back to mel spectrogram features.
        
        Args:
            audio_chunk: Enhanced raw audio chunk as numpy array
            
        Returns: Mel spectrogram features as numpy array padded/trimmed to expected size
        """
        #extract mel features from the enhanced audio using the model's feature extractor
        features = self.feature_extractor(enhanced_audio_chunk)
        
        #pad or trim to the expected size - 3000 mel frames by default.
        features = pad_or_trim(features)
        return features

    def get_segment_duration(self, seek, segment_size):
        """
        Calculate the duration in seconds of the current audio segment.
        
        Args:
            seek: Starting mel frame index
            segment_size: Number of mel frames in the segment
            
        Returns: Duration of the segment in seconds
        """
        #convert mel frame indices to audio sample indices
        audio_start_sample = int(seek * self.feature_extractor.hop_length)
        audio_end_sample = int((seek + segment_size) * self.feature_extractor.hop_length)
        
        #calculate duration in seconds
        duration_seconds = (audio_end_sample - audio_start_sample) / self.sampling_rate
        
        return duration_seconds


    def _recovery_increase_beam_size(self, encoder_output, prompt, tokenizer, modified_options):
        """First recovery attempt: increase beam size and remove context."""
        print("First recovery attempt - increase beam size")
        modified_options.condition_on_previous_text = False
        modified_options.beam_size = min(modified_options.beam_size + 2, modified_options.max_beam_size)
        
        result, avg_logprob, temperature, compression_ratio = self.generate_with_fallback(
            encoder_output, prompt, tokenizer, modified_options
        )
        return result, avg_logprob, temperature, compression_ratio

    def _recovery_decrease_beam_size(self, encoder_output, prompt, tokenizer, modified_options):
        """Second recovery attempt: decrease beam size and remove context."""
        print("Second recovery attempt - decrease beam size")
        modified_options.condition_on_previous_text = False
        modified_options.beam_size = max(modified_options.beam_size - 2, modified_options.min_beam_size)
        
        result, avg_logprob, temperature, compression_ratio = self.generate_with_fallback(
            encoder_output, prompt, tokenizer, modified_options
        )
        return result, avg_logprob, temperature, compression_ratio
    
    def _recovery_drop_beam_size(self, encoder_output, prompt, tokenizer, modified_options):
        """Third recovery attempt: decrease beam size to 1 (minimum value)."""
        print("Third recovery attempt - drop beam size to 1")
        modified_options.condition_on_previous_text = False
        modified_options.beam_size = 1
        
        result, avg_logprob, temperature, compression_ratio = self.generate_with_fallback(
            encoder_output, prompt, tokenizer, modified_options
        )
        return result, avg_logprob, temperature, compression_ratio

    def _recovery_vad_cleaning(self, encoder_output, prompt, tokenizer, modified_options, current_audio_chunk):
        """Forth recovery attempt: VAD-based audio cleaning."""
        print("Fourth recovery attempt - VAD-based Audio Cleaning")
        
        # Apply VAD to clean the audio chunk
        vad_parameters = VadOptions(max_speech_duration_s=3000, min_silence_duration_ms=160)
        speech_chunks = get_speech_timestamps(current_audio_chunk, vad_parameters)
        audio_chunks, chunks_metadata = collect_chunks(current_audio_chunk, speech_chunks)
        cleaned_audio = np.concatenate(audio_chunks, axis=0)
        
        # Calculate new duration and adjust parameters
        cleaned_duration = cleaned_audio.shape[0] / self.sampling_rate
        print(f"VAD reduced audio from {current_audio_chunk.shape[0]/self.sampling_rate:.2f}s to {cleaned_duration:.2f}s")
        
        # Convert cleaned audio to features and encode
        cleaned_features = self.convert_audio_to_features(cleaned_audio)
        cleaned_encoder_output = self.encode(cleaned_features)
        
        # Generate result with cleaned audio
        result, avg_logprob, temperature, compression_ratio = self.generate_with_fallback(
            cleaned_encoder_output, prompt, tokenizer, modified_options
        )
        return result, avg_logprob, temperature, compression_ratio

    def _recovery_audio_enhancement(self, encoder_output, prompt, tokenizer, modified_options, current_audio_chunk):
        """Fifth recovery attempt: Audio enhancement with Convtasnet."""
        print("Fifth recovery attempt - Audio Enhancement with Convtasnet")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
            sf.write(temp_audio_path, current_audio_chunk, self.sampling_rate)
        
        try:
            # Enhance the audio using Convtasnet
            enhancer = Enhancer(temp_audio_path, 'convtasnet')
            enhanced_audio_tensor = enhancer.get_enhanced_audio()
            
            # Convert tensor to numpy array
            enhanced_audio_chunk = enhanced_audio_tensor.squeeze().detach().cpu().numpy()
            
            # Clean up enhancement model memory
            del enhanced_audio_tensor
            del enhancer
            
            # Force garbage collection and clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            
            # Convert enhanced audio back to features and encode
            enhanced_features = self.convert_audio_to_features(enhanced_audio_chunk)
            enhanced_encoder_output = self.encode(enhanced_features)
            
            # Generate result with enhanced audio
            result, avg_logprob, temperature, compression_ratio = self.generate_with_fallback(
                enhanced_encoder_output, prompt, tokenizer, modified_options
            )
            return result, avg_logprob, temperature, compression_ratio
            
        except Exception as e:
            print(f"Audio enhancement failed: {e}")
            raise
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    def _recovery_speaker_separation(self, encoder_output, prompt, tokenizer, modified_options, current_audio_chunk, diarization_result=None):
        """Sixth recovery attempt: Speaker separation and separate transcription."""
        print("Sixth recovery attempt - Speaker Separation")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
            sf.write(temp_audio_path, current_audio_chunk, self.sampling_rate)
        
        try:
            # Use provided diarization result if available, otherwise run diarization
            diarizer = None
            if diarization_result is None:
                print("No diarization result provided, running diarization...")
                diarizer = Diarizer(temp_audio_path)
                diarization_result = diarizer.diarization
            
            # Separate speakers using the separation model
            separator = Separator(temp_audio_path, 'convtasnet_libri2mix_noisy')
            
            # Get the separated sources using the existing method
            separated_audio_tensors = separator.get_sources()
            
            # Convert tensors to numpy arrays
            separated_audio_chunks = []
            for audio_tensor in separated_audio_tensors:
                if audio_tensor.dim() == 2:
                    audio_chunk = audio_tensor[0].detach().cpu().numpy()
                else:
                    audio_chunk = audio_tensor.detach().cpu().numpy()
                separated_audio_chunks.append(audio_chunk)
            
            # Clean up separation model memory
            del separated_audio_tensors
            del separator
            
            # Force garbage collection and clear GPU cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            
            # Transcribe each separated track separately
            all_transcriptions = []
            for i, audio_chunk in enumerate(separated_audio_chunks):
                print(f"Transcribing speaker {i+1} track...")
                
                features = self.convert_audio_to_features(audio_chunk)
                encoder_output_sep = self.encode(features)
                
                result, avg_logprob, temperature, compression_ratio = self.generate_with_fallback(
                    encoder_output_sep, prompt, tokenizer, modified_options
                )
                
                text = tokenizer.decode(result.sequences_ids[0])
                
                all_transcriptions.append({
                    'text': text,
                    'result': result,
                    'avg_logprob': avg_logprob,
                    'temperature': temperature,
                    'compression_ratio': compression_ratio,
                    'speaker_id': i
                })
            
            # Merge transcriptions with diarization timing
            merged_result = self._merge_transcriptions(all_transcriptions, diarization_result)
            
            return merged_result
            
        except Exception as e:
            print(f"Speaker separation failed: {e}")
            raise
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            # Clean up diarization model memory only if we created one
            if diarizer is not None:
                try:
                    del diarizer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

    def _merge_transcriptions(self, transcriptions, diarization_result=None):
        """
        Merge transcriptions from separated speaker tracks based on timestamps and diarization.
        
        Args:
            transcriptions: List of transcription dictionaries
            diarization_result: Optional diarization result for timing information
            
        Returns:
            tuple: (result, avg_logprob, temperature, compression_ratio)
        """
        if not transcriptions:
            raise ValueError("No transcriptions to merge")
        
        if len(transcriptions) == 1:
            # Only one speaker, return as-is
            trans = transcriptions[0]
            return trans['result'], trans['avg_logprob'], trans['temperature'], trans['compression_ratio']
        
        # Simple fallback merging strategy for multiple speakers
        merged_text = ""
        best_result = None
        best_avg_logprob = float('-inf')
        
        for trans in transcriptions:
            speaker_label = f"[Speaker {trans['speaker_id']+1}] "
            merged_text += speaker_label + trans['text'] + " "
            
            if trans['avg_logprob'] > best_avg_logprob:
                best_avg_logprob = trans['avg_logprob']
                best_result = trans['result']
        
        print(f"Simple merged transcription: {merged_text[:100]}...")
        
        return best_result, best_avg_logprob, transcriptions[0]['temperature'], transcriptions[0]['compression_ratio']

    def _check_for_speaker_overlap(self, audio_chunk: np.ndarray, overlap_threshold: float = 0.1) -> tuple[bool, Any]:
        """
        Check if speaker separation is needed using diarization.
        
        Args:
            audio_chunk: Raw audio chunk as numpy array
            overlap_threshold: Threshold for overlap percentage (default 0.25 = 25%)
            
        Returns:
            tuple: (needs_separation, diarization_result) where needs_separation is bool
        """
        try:
            # Save audio chunk to temporary file for diarization
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_path = temp_file.name
                sf.write(temp_audio_path, audio_chunk, self.sampling_rate)
            
            try:
                # Use diarization to detect speakers and their timings
                diarizer = Diarizer(temp_audio_path)
                diarization_result = diarizer.diarization
                
                # Get speaker count first
                num_speakers = len(set(label for _, _, label in diarization_result.itertracks(yield_label=True)))
                
                if num_speakers <= 1:
                    print(f"Only {num_speakers} speaker detected, skipping separation")
                    return False, diarization_result
                
                # Extract speaker segments from diarization result
                speaker_segments = []
                for segment, track, label in diarization_result.itertracks(yield_label=True):
                    speaker_segments.append({
                        'speaker': label,
                        'start': segment.start,
                        'end': segment.end,
                        'duration': segment.end - segment.start
                    })
                
                # Calculate overlap percentage
                total_duration = audio_chunk.shape[0] / self.sampling_rate
                overlap_duration = 0.0
                
                # Check for overlapping speech segments
                for i, seg1 in enumerate(speaker_segments):
                    for seg2 in speaker_segments[i+1:]:
                        # Calculate overlap between two segments
                        overlap_start = max(seg1['start'], seg2['start'])
                        overlap_end = min(seg1['end'], seg2['end'])
                        
                        if overlap_end > overlap_start:
                            overlap_duration += overlap_end - overlap_start
                
                overlap_percentage = overlap_duration / total_duration
                
                print(f"Detected {num_speakers} speakers with {overlap_percentage:.2%} overlap")
                
                # Separation is needed if overlap exceeds threshold
                needs_separation = overlap_percentage > overlap_threshold
                
                if needs_separation:
                    print(f"Overlap {overlap_percentage:.2%} exceeds threshold {overlap_threshold:.2%}, separation needed")
                else:
                    print(f"Overlap {overlap_percentage:.2%} below threshold {overlap_threshold:.2%}, separation not needed")
                
                return needs_separation, diarization_result
                
            finally:
                # Clean up diarization model memory
                try:
                    del diarizer
                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                
        except Exception as e:
            print(f"Diarization check failed: {e}, defaulting to separation")
            # If diarization fails, default to using separation to be safe
            return True, None
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)


def restore_speech_timestamps(
    segments: Iterable[Segment],
    speech_chunks: List[dict],
    sampling_rate: int,
) -> Iterable[Segment]:
    ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)

    for segment in segments:
        if segment.words:
            words = []
            for word in segment.words:
                # Ensure the word start and end times are resolved to the same chunk.
                middle = (word.start + word.end) / 2
                chunk_index = ts_map.get_chunk_index(middle)
                word.start = ts_map.get_original_time(word.start, chunk_index)
                word.end = ts_map.get_original_time(word.end, chunk_index)
                words.append(word)

            segment.start = words[0].start
            segment.end = words[-1].end
            segment.words = words

        else:
            segment.start = ts_map.get_original_time(segment.start)
            segment.end = ts_map.get_original_time(segment.end)

        yield segment


def get_ctranslate2_storage(segment: np.ndarray) -> ctranslate2.StorageView:
    segment = np.ascontiguousarray(segment)
    segment = ctranslate2.StorageView.from_array(segment)
    return segment


def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def get_suppressed_tokens(
    tokenizer: Tokenizer,
    suppress_tokens: Tuple[int],
) -> Optional[List[int]]:
    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

    suppress_tokens.extend(
        [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
        ]
    )

    return tuple(sorted(set(suppress_tokens)))


def merge_punctuations(alignment: List[dict], prepended: str, appended: str) -> None:
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous["word"].startswith(" ") and previous["word"].strip() in prepended:
            # prepend it to the following word
            following["word"] = previous["word"] + following["word"]
            following["tokens"] = previous["tokens"] + following["tokens"]
            previous["word"] = ""
            previous["tokens"] = []
        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous["word"].endswith(" ") and following["word"] in appended:
            # append it to the previous word
            previous["word"] = previous["word"] + following["word"]
            previous["tokens"] = previous["tokens"] + following["tokens"]
            following["word"] = ""
            following["tokens"] = []
        else:
            i = j
        j += 1
