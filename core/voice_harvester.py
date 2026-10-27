"""Autonomous local voice harvester.

Pipeline stages (all pluggable so unit tests run offline in milliseconds):

1. YouTube discovery + download (``yt-dlp``) with ``processed_video_ids`` dedup.
2. Vocal isolation + Vocal-to-BGM energy gate (``>= 8.0 dB``).
3. Smart VAD slicing with mandatory ``+200 ms`` head/tail silence padding.
4. Dominant speaker clustering (rejects guest voices / inserted ads).
5. Word-level ASR confidence gate (``>= 0.92``) and F5-TTS / GPT-SoVITS export.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import soundfile as sf

logger = logging.getLogger('epub2m4b.voice_harvester')

SAMPLE_RATE = 24000
CHANNELS = 1
SUBTYPE = 'PCM_16'

FRAME_MS = 50
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1200 samples

# Speech island detection on 50ms frames.
SPEECH_FRAME_DBFS = -38.0
# Islands separated by less than this much silence are merged into one sentence.
MERGE_GAP_SEC = 0.35

MIN_CLIP_SEC = 3.5
MAX_CLIP_SEC = 12.0
PAD_MS = 200

MIN_VOCAL_TO_BGM_DB = 8.0
CLIP_PEAK_CEIL_DBFS = -0.2
CLIP_RMS_FLOOR_DBFS = -42.0

MIN_WORD_PROB = 0.92
MIN_CHARS_PER_SEC = 8.0
MAX_CHARS_PER_SEC = 28.0

GOLDEN_MIN_SEC = 7.0
GOLDEN_MAX_SEC = 11.0

MIN_VIDEO_SEC = 120.0
MAX_VIDEO_SEC = 3600.0
INTRO_SKIP_SEC = 15.0
OUTRO_SKIP_SEC = 15.0

DIACRITIC_CHARS = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'

# Vietnamese tone diacritics, grouped into the 6 tone marks used for the
# "contains at least 4/6 tones" golden-reference diversity heuristic.
TONE_MARKS = ('\u0300', '\u0301', '\u0303', '\u0309', '\u0323')


def _db(x: np.ndarray | float) -> float:
    """Convert a linear amplitude/energy value to dBFS (``-inf`` for silence)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return -math.inf
    value = float(np.sqrt(np.mean(np.square(arr)))) if arr.ndim > 0 else abs(float(arr))
    if not math.isfinite(value) or value <= 1e-12:
        return -math.inf
    return float(20.0 * math.log10(value))


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32, preserving its native sample rate."""
    audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sr)


def compute_rms_dbfs(samples: np.ndarray) -> float:
    """Root-mean-square level of ``samples`` in dBFS."""
    return _db(samples)


def compute_peak_dbfs(samples: np.ndarray) -> float:
    """Peak absolute level of ``samples`` in dBFS."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return -math.inf
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-12:
        return -math.inf
    return float(20.0 * math.log10(peak))


@dataclass
class RawAudioCandidate:
    """A sliced, padded speech segment that survived the energy gates."""

    path: Path
    start_sec: float
    end_sec: float
    vocal_to_bgm_db: float
    rms_dbfs: float
    peak_dbfs: float

    @property
    def duration_sec(self) -> float:
        return float(self.end_sec - self.start_sec)


def _frame_rms_db(samples: np.ndarray, frame_samples: int = FRAME_SAMPLES) -> np.ndarray:
    """Per-frame RMS in dB (linear dBFS), ``-120.0`` used as the silence floor."""
    n_frames = samples.size // frame_samples
    if n_frames <= 0:
        return np.array([-120.0], dtype=np.float64)
    trimmed = samples[: n_frames * frame_samples].reshape(n_frames, frame_samples).astype(np.float64)
    ms = np.sqrt(np.mean(np.square(trimmed), axis=1))
    with np.errstate(divide='ignore'):
        return 20.0 * np.log10(np.maximum(ms, 1e-6))


def _merge_islands(active: np.ndarray, gap_frames: int) -> list[tuple[int, int]]:
    islands: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            islands.append((start, idx))
            start = None
    if start is not None:
        islands.append((start, active.size))
    if not islands:
        return []

    merged: list[tuple[int, int]] = [islands[0]]
    for begin, end in islands[1:]:
        prev_begin, prev_end = merged[-1]
        if begin - prev_end < gap_frames:
            merged[-1] = (prev_begin, end)
        else:
            merged.append((begin, end))
    return merged


def _split_long_island(begin: int, end: int, frame_db: np.ndarray, max_frames: int) -> list[tuple[int, int]]:
    """Split an over-long island at its quietest internal valley, recursively."""
    if end - begin <= max_frames:
        return [(begin, end)]

    lo = begin + max(1, int(max_frames * 0.4))
    hi = end - max(1, int(max_frames * 0.4))
    if hi <= lo:
        return [(begin, end)]

    valley = int(np.argmin(frame_db[lo:hi])) + lo
    left = _split_long_island(begin, valley, frame_db, max_frames)
    right = _split_long_island(valley, end, frame_db, max_frames)
    return left + right


def _crop_inst(inst: np.ndarray, inst_sr: int, start_sec: float, end_sec: float, length: int, out_sr: int) -> np.ndarray:
    """Resample-free crop of the instrumental track with a crude linear resample."""
    if inst_sr != out_sr:
        ratio = out_sr / float(inst_sr)
        new_len = max(1, int(round(inst.size * ratio)))
        src_idx = np.linspace(0, inst.size - 1, new_len)
        inst = np.interp(src_idx, np.arange(inst.size), inst).astype(np.float32)
    begin = int(round(start_sec * out_sr))
    end = begin + length
    if begin >= inst.size:
        return np.zeros(length, dtype=np.float32)
    chunk = inst[begin:end]
    if chunk.size < length:
        chunk = np.pad(chunk, (0, length - chunk.size))
    return chunk


def slice_and_filter_segments(
    vocal_wav: Path,
    out_dir: Path,
    inst_wav: Path | None = None,
    min_sec: float = MIN_CLIP_SEC,
    max_sec: float = MAX_CLIP_SEC,
    pad_ms: int = PAD_MS,
    min_snr_db: float = MIN_VOCAL_TO_BGM_DB,
) -> list[RawAudioCandidate]:
    """Slice ``vocal_wav`` into clean, padded sentences.

    Speech islands are detected on 50ms frames (``frame_rms_dbfs > -38 dBFS``),
    islands separated by ``< 0.35s`` silence are merged, islands longer than
    ``max_sec`` are split at their quietest internal valley, ``pad_ms`` of
    silence is added at both ends, then clips are filtered by:

    * ``min_sec <= duration <= max_sec``
    * ``vocal_to_bgm_db >= min_snr_db`` (background music too loud -> drop)
    * ``peak_dbfs < -0.2`` (clipping -> drop)
    * ``rms_dbfs >= -42.0`` (too quiet -> drop)
    """
    vocal_path = Path(vocal_wav)
    vocals, sr = _read_mono(vocal_path)
    instruments: np.ndarray | None = None
    inst_sr = sr
    if inst_wav is not None and Path(inst_wav).exists():
        instruments, inst_sr = _read_mono(Path(inst_wav))
        if inst_sr != sr:
            ratio = sr / float(inst_sr)
            new_len = max(1, int(round(instruments.size * ratio)))
            src_idx = np.linspace(0, instruments.size - 1, new_len)
            instruments = np.interp(src_idx, np.arange(instruments.size), instruments).astype(np.float32)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_samples = max(1, int(round(sr * FRAME_MS / 1000.0)))
    frame_db = _frame_rms_db(vocals, frame_samples)
    active = frame_db > SPEECH_FRAME_DBFS
    gap_frames = int(round(MERGE_GAP_SEC * sr / frame_samples))
    max_frames = int(round(max_sec * sr / frame_samples))
    pad_samples = int(round(pad_ms * sr / 1000.0))

    islands: list[tuple[int, int]] = []
    for begin, end in _merge_islands(active, gap_frames):
        islands.extend(_split_long_island(begin, end, frame_db, max_frames))

    candidates: list[RawAudioCandidate] = []
    for idx, (begin_frame, end_frame) in enumerate(islands, start=1):
        start_sample = begin_frame * frame_samples
        end_sample = min(vocals.size, end_frame * frame_samples)

        start_sample = max(0, start_sample - pad_samples)
        end_sample = min(vocals.size, end_sample + pad_samples)
        if end_sample <= start_sample:
            continue

        clip = vocals[start_sample:end_sample]
        duration = clip.size / float(sr)
        if duration < min_sec or duration > max_sec:
            continue

        peak = compute_peak_dbfs(clip)
        rms = compute_rms_dbfs(clip)
        if peak >= CLIP_PEAK_CEIL_DBFS or rms < CLIP_RMS_FLOOR_DBFS:
            continue

        vocal_rms = _db(clip)
        if instruments is not None:
            inst_clip = _crop_inst(instruments, sr, start_sample / float(sr), end_sample / float(sr), clip.size, sr)
            bgm_rms = _db(inst_clip)
        else:
            bgm_rms = -math.inf
        vocal_to_bgm_db = vocal_rms - bgm_rms
        if vocal_to_bgm_db < min_snr_db:
            continue

        clip_path = out_dir / f'clip_{idx:04d}.wav'
        sf.write(str(clip_path), clip, sr, subtype=SUBTYPE)
        candidates.append(
            RawAudioCandidate(
                path=clip_path,
                start_sec=start_sample / float(sr),
                end_sec=end_sample / float(sr),
                vocal_to_bgm_db=float(vocal_to_bgm_db),
                rms_dbfs=float(rms),
                peak_dbfs=float(peak),
            )
        )

    logger.info(
        f'✂️ [SLICE] {vocal_path.name}: {len(islands)} câu thô -> giữ {len(candidates)} câu sạch '
        f'(pad={pad_ms}ms, min_snr={min_snr_db}dB)'
    )
    return candidates
