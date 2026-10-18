"""GigaAM engine: checkpoint loading, VAD chunking, batched decoding.

Upstream's own ``transcribe_longform`` cuts audio with pyannote, whose segmentation
checkpoint is gated on HuggingFace and needs a token. Here the same job is done with
silero-vad (MIT, weights ship inside the wheel), so nothing gated is fetched at runtime
and the container works on a host with no HF credentials.

Everything below the VAD is upstream's batched longform path (AudioDataset -> forward ->
_decode), which is why the GigaAM commit is pinned in the Dockerfile.
"""

from __future__ import annotations

import contextlib
import io
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

import gigaam
from gigaam.preprocess import SAMPLE_RATE, load_audio
from gigaam.utils import AudioDataset

LOG = logging.getLogger("gigaam.asr")

# Upstream raises above this in transcribe(); every chunk we build stays under it.
SINGLE_SHOT_LIMIT_S = 25.0


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcription:
    text: str
    duration: float
    segments: List[Segment]


def model_id(checkpoint_name: str) -> str:
    """Public model id served over HTTP: v3_e2e_rnnt -> gigaam-v3-e2e-rnnt."""
    return "gigaam-" + checkpoint_name.replace("_", "-")


def plan_chunks(
    speech: Sequence[Tuple[float, float]], max_chunk_s: float, max_gap_s: float
) -> List[Tuple[float, float]]:
    """Glue VAD speech regions into chunks of at most ``max_chunk_s`` seconds.

    Regions closer than ``max_gap_s`` are merged while the result fits the limit; a single
    speech run longer than the limit is cut into equal pieces (rare: continuous shouting,
    music, or a VAD miss).
    """
    chunks: List[List[float]] = []
    for start, end in speech:
        pieces: List[Tuple[float, float]] = []
        cursor = start
        while end - cursor > max_chunk_s:
            pieces.append((cursor, cursor + max_chunk_s))
            cursor += max_chunk_s
        pieces.append((cursor, end))

        for piece_start, piece_end in pieces:
            if (
                chunks
                and piece_start - chunks[-1][1] <= max_gap_s
                and piece_end - chunks[-1][0] <= max_chunk_s
            ):
                chunks[-1][1] = piece_end
            else:
                chunks.append([piece_start, piece_end])
    return [(start, end) for start, end in chunks]


class AsrEngine:
    """Holds the loaded checkpoints and serializes GPU work behind one lock."""

    def __init__(
        self,
        checkpoints: Sequence[str],
        device: str = "cuda",
        fp16_encoder: bool = True,
        cache_dir: Optional[str] = None,
        batch_size: int = 8,
        max_chunk_s: float = 22.0,
        max_gap_s: float = 0.6,
        vad_threshold: float = 0.5,
        vad_min_silence_ms: int = 300,
        vad_speech_pad_ms: int = 200,
    ):
        if max_chunk_s >= SINGLE_SHOT_LIMIT_S:
            raise ValueError(
                f"max_chunk_s must stay below {SINGLE_SHOT_LIMIT_S} s, got {max_chunk_s}"
            )

        self.batch_size = batch_size
        self.max_chunk_s = max_chunk_s
        self.max_gap_s = max_gap_s
        self.vad_threshold = vad_threshold
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms

        # One lock for the whole request: the GPU is single, and the silero JIT model keeps
        # internal RNN state, so two threads inside the VAD segfault the process.
        self._lock = threading.Lock()

        from silero_vad import get_speech_timestamps, load_silero_vad

        self._vad = load_silero_vad()
        self._vad_speech_timestamps = get_speech_timestamps

        self.models: Dict[str, torch.nn.Module] = {}
        self.checkpoints: Dict[str, str] = {}
        for name in checkpoints:
            LOG.info("loading GigaAM checkpoint %s on %s", name, device)
            model = gigaam.load_model(
                name,
                fp16_encoder=fp16_encoder,
                device=device,
                download_root=cache_dir,
            )
            self.models[model_id(name)] = model
            self.checkpoints[model_id(name)] = name
        self.default_model = next(iter(self.models))
        LOG.info("models ready: %s (default %s)", list(self.models), self.default_model)

    # -- model registry ----------------------------------------------------

    def resolve(self, requested: Optional[str]) -> Optional[str]:
        """Map a client-supplied model name to a loaded model id.

        Aliases keep drop-in clients happy: an empty value, ``gigaam``, ``whisper-1``
        or the bare checkpoint name all land on the default model.
        """
        if not requested:
            return self.default_model
        name = requested.strip()
        if name in self.models:
            return name
        by_checkpoint = {ckpt: mid for mid, ckpt in self.checkpoints.items()}
        if name in by_checkpoint:
            return by_checkpoint[name]
        if name.lower() in {"gigaam", "whisper-1", "whisper", "default"}:
            return self.default_model
        return None

    # -- VAD ---------------------------------------------------------------

    def _speech_regions(self, wav: torch.Tensor) -> List[Tuple[float, float]]:
        stamps = self._vad_speech_timestamps(
            wav,
            self._vad,
            sampling_rate=SAMPLE_RATE,
            threshold=self.vad_threshold,
            min_silence_duration_ms=self.vad_min_silence_ms,
            speech_pad_ms=self.vad_speech_pad_ms,
        )
        return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]

    def _plan(self, wav: torch.Tensor, duration: float) -> List[Tuple[float, float]]:
        if duration <= self.max_chunk_s:
            return [(0.0, duration)]

        regions = self._speech_regions(wav)
        if not regions:
            LOG.warning(
                "VAD found no speech in %.1f s of audio, falling back to fixed windows",
                duration,
            )
            regions = [(0.0, duration)]
        return plan_chunks(regions, self.max_chunk_s, self.max_gap_s)

    # -- inference ---------------------------------------------------------

    @torch.inference_mode()
    def _decode_chunks(
        self,
        model: torch.nn.Module,
        wav: torch.Tensor,
        chunks: Sequence[Tuple[float, float]],
        word_timestamps: bool,
    ) -> List[Segment]:
        items = [
            wav[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)].numpy()
            for start, end in chunks
        ]
        items = [item for item in items if item.size]
        if not items:
            return []

        # AudioDataset prints a "filtered by duration" line on every construction.
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = AudioDataset(items, tokenizer=None)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=AudioDataset.collate,
        )

        segments: List[Segment] = []
        index = 0
        for wav_pad, wav_lens in loader:
            wav_pad = wav_pad.to(model._device).to(model._dtype)
            wav_lens = wav_lens.to(model._device)
            encoded, encoded_len = model.forward(wav_pad, wav_lens)
            for text, words in model._decode(
                encoded, encoded_len, wav_lens, word_timestamps
            ):
                start, end = chunks[index]
                index += 1
                segments.append(
                    Segment(
                        start=round(start, 3),
                        end=round(end, 3),
                        text=text.strip(),
                        words=[
                            Word(
                                text=word.text,
                                start=round(word.start + start, 3),
                                end=round(word.end + start, 3),
                            )
                            for word in (words or [])
                        ],
                    )
                )
        return segments

    def transcribe(
        self, audio_path: str, model_name: Optional[str] = None, word_timestamps: bool = False
    ) -> Transcription:
        model_key = self.resolve(model_name)
        if model_key is None:
            raise KeyError(model_name)
        model = self.models[model_key]

        wav = load_audio(audio_path)
        duration = wav.numel() / SAMPLE_RATE
        if duration <= 0:
            return Transcription(text="", duration=0.0, segments=[])

        with self._lock:
            chunks = self._plan(wav, duration)
            segments = self._decode_chunks(model, wav, chunks, word_timestamps)

        spoken = [segment for segment in segments if segment.text]
        return Transcription(
            text=" ".join(segment.text for segment in spoken).strip(),
            duration=round(duration, 3),
            segments=spoken,
        )
