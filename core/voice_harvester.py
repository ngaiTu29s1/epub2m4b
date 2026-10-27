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
import sys
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
# A clip must be within this RMS distance of its own peak before the peak gate
# treats it as actually clipped (crest factor < 1.8 dB means the whole clip is
# pinned at full scale).
CLIPPING_HEADROOM_RMS_DBFS = -2.0

MIN_WORD_PROB = 0.92
# Speaking-rate gate. The spec writes "8-28 ký tự/giây" (loose, since real
# Vietnamese narration measures 12-16 characters/sec, so a chars/sec floor of
# 8 never fires). The gate is therefore applied in Whisper word tokens/sec,
# where the spec's own worked constant 64 chars / 8 gives 8 words/sec, and
# normal narration measures 2.9-4.0 words/sec. A floor of 1.5 tokens/sec still
# rejects timings that cannot belong to the clip (hallucinated word storms),
# while 28 tokens/sec rejects ASR blow-ups.
MIN_WORDS_PER_SEC = 1.5
MAX_WORDS_PER_SEC = 28.0
# Whisper repetition loops collapse a sentence into one repeated token.
MAX_DUPLICATE_TOKEN_RATIO = 0.5

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
    """Split an over-long island at its quietest internal valley.

    The valley is chosen as close to ``max_frames`` from the island start as
    possible (still leaving a remainder), so the split never fragments a
    sentence into pieces shorter than the minimum clip length; each fragment
    is still guaranteed to be ``<= max_frames``.
    """
    if end - begin <= max_frames:
        return [(begin, end)]

    search_lo = begin + max(1, int(max_frames * 0.5))
    search_hi = min(end - 1, begin + max_frames)
    if search_hi <= search_lo:
        split_at = min(end, begin + max_frames)
        return _split_long_island(begin, split_at, frame_db, max_frames) + _split_long_island(
            split_at, end, frame_db, max_frames
        )

    # Quietest frame closest to the target length (ties -> farthest along).
    window = frame_db[search_lo:search_hi]
    quietest = float(np.min(window))
    candidates = np.flatnonzero(window <= quietest + 1e-9) + search_lo
    split_at = int(candidates[-1])

    left = _split_long_island(begin, split_at, frame_db, max_frames)
    right = _split_long_island(split_at, end, frame_db, max_frames)
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
        if rms < CLIP_RMS_FLOOR_DBFS:
            continue
        # Digital clipping gate: a sample pinned at (or within 0.1 dBFS of) full
        # scale is a broken take and must not enter a clone dataset. Merely
        # hot-mastered audio (this is the norm for YouTube uploads, which sit at
        # ~0 dBFS) is *not* clipped: if the clip's energy is well below the peak
        # then the peak is just a transient and the clip is perfectly usable.
        if peak >= CLIP_PEAK_CEIL_DBFS and rms >= CLIPPING_HEADROOM_RMS_DBFS:
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


# ---------------------------------------------------------------------------
# Step 4 — Dominant speaker clustering
# ---------------------------------------------------------------------------

EMBED_DIM = 48
MEL_BANDS = 40
CENTROID_NEIGHBOR_RATIO = 0.6


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (np.power(10.0, np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, sr: int, bands: int = MEL_BANDS) -> np.ndarray:
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_edges = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), bands + 2)
    hz_edges = _mel_to_hz(mel_edges)
    bank = np.zeros((bands, freqs.size), dtype=np.float64)
    for b in range(bands):
        left, center, right = hz_edges[b], hz_edges[b + 1], hz_edges[b + 2]
        rising = np.clip((freqs - left) / max(center - left, 1e-9), 0.0, 1.0)
        falling = np.clip((right - freqs) / max(right - center, 1e-9), 0.0, 1.0)
        bank[b] = np.minimum(rising, falling)
    return bank


def _log_mel_stats(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of log-mel energies (per band)."""
    n_fft = 1024
    hop = max(1, n_fft // 4)
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))

    n_frames = 1 + (samples.size - n_fft) // hop
    window = np.hanning(n_fft)
    frames = np.stack([samples[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = spectrum @ _mel_filterbank(n_fft, sr).T
    log_mel = np.log10(np.maximum(mel, 1e-10))
    return log_mel.mean(axis=0), log_mel.std(axis=0)


def _spectral_centroid(samples: np.ndarray, sr: int) -> float:
    n_fft = 1024
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))
    spectrum = np.abs(np.fft.rfft(samples[:n_fft] * np.hanning(n_fft)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = float(spectrum.sum())
    if total <= 1e-12:
        return 0.0
    return float((spectrum * freqs).sum() / total)


def _zero_crossing_rate(samples: np.ndarray) -> float:
    if samples.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(samples).astype(np.int8)))))


def compute_speaker_embedding(wav_path: Path) -> np.ndarray:
    """L2-normalized timbre fingerprint of one clip (log-mel + pitch/spectral stats).

    Deliberately lightweight (numpy only, no neural speaker model) so the
    offline test suite can exercise the whole pipeline, while real timbre
    differences (voice colour, pitch, formants) still separate speakers.
    """
    samples, sr = _read_mono(Path(wav_path))
    if samples.size == 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)

    peak = float(np.max(np.abs(samples)))
    if peak > 1e-9:
        samples = samples / peak

    mel_mean, mel_std = _log_mel_stats(samples, sr)
    centroid = _spectral_centroid(samples, sr)
    zcr = _zero_crossing_rate(samples)

    frame_len = max(1, int(round(sr * 0.04)))
    frames = [samples[i : i + frame_len] for i in range(0, samples.size - frame_len + 1, frame_len)]
    if frames:
        frame_db = np.array([_db(f) for f in frames], dtype=np.float64)
        voiced = frame_db > (np.max(frame_db) - 25.0)
        voiced_frames = [f for f, keep in zip(frames, voiced) if keep] or frames
        autocorr_pitch = []
        for frame in voiced_frames[:400]:
            win = frame - float(np.mean(frame))
            ac = np.correlate(win, win, mode='full')[win.size - 1 :]
            lo = max(1, int(sr / 400.0))
            hi = min(ac.size - 1, int(sr / 60.0))
            if hi <= lo:
                continue
            lag = lo + int(np.argmax(ac[lo:hi]))
            autocorr_pitch.append(sr / float(lag))
        pitch_mean = float(np.mean(autocorr_pitch)) if autocorr_pitch else 0.0
        pitch_std = float(np.std(autocorr_pitch)) if autocorr_pitch else 0.0
    else:
        pitch_mean = pitch_std = 0.0

    extras = np.array(
        [
            math.log1p(centroid) / 10.0,
            math.log1p(pitch_mean) / 10.0,
            pitch_std / 100.0,
            zcr,
            compute_rms_dbfs(samples) / 60.0,
            compute_peak_dbfs(samples) / 20.0,
        ],
        dtype=np.float64,
    )

    vector = np.concatenate(
        [
            mel_mean * 2.0,
            mel_std,
            extras,
        ]
    ).astype(np.float64)

    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _dominant_centroid(vectors: np.ndarray) -> np.ndarray:
    """Medoid-style centroid robust to guest-voice outliers.

    For each candidate, average the cosine similarity to its top 60% most
    similar neighbours; the highest-scoring candidate is the dense medoid.
    Guest voices never drag the centroid because they are a minority and
    therefore never appear in another speaker's top-neighbour window.
    """
    n = vectors.shape[0]
    if n == 0:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    if n == 1:
        return vectors[0]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    sims = normalized @ normalized.T

    k = max(1, int(math.ceil(n * CENTROID_NEIGHBOR_RATIO)))
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = np.sort(sims[i])[::-1][:k]
        scores[i] = float(np.mean(row))
    return vectors[int(np.argmax(scores))]


def filter_dominant_speaker(
    candidates: list[RawAudioCandidate],
    min_similarity: float = 0.78,
    embed_fn: Callable[[Path], np.ndarray] = compute_speaker_embedding,
) -> list[RawAudioCandidate]:
    """Keep only clips that match the dominant (channel-owner) speaker.

    Drops guest voices, interviews and inserted advertisements whose cosine
    similarity to the dominant cluster centroid is ``< min_similarity``.
    """
    if not candidates:
        return []

    vectors = np.stack([np.asarray(embed_fn(c.path), dtype=np.float64) for c in candidates])
    centroid = _dominant_centroid(vectors)

    kept: list[RawAudioCandidate] = []
    for candidate, vector in zip(candidates, vectors):
        similarity = _cosine(vector, centroid)
        if similarity >= min_similarity:
            kept.append(candidate)
        else:
            logger.info(f'🚫 [SPEAKER] Loại {candidate.path.name}: cosine={similarity:.3f} < {min_similarity}')
    logger.info(f'🗣️ [SPEAKER] Giữ {len(kept)}/{len(candidates)} câu thuộc giọng chủ kênh')
    return kept


# ---------------------------------------------------------------------------
# Step 5 — Word-level ASR gate, Golden 10s selection, dataset export
# ---------------------------------------------------------------------------


@dataclass
class TranscribedWord:
    word: str
    probability: float


@dataclass
class CuratedClip:
    clip_path: Path
    text: str
    duration_sec: float
    snr_db: float
    min_word_prob: float
    quality_score: float


@dataclass
class VoiceHarvestResult:
    voice_id: str
    voice_dir: Path
    total_clean_sec: float
    kept_clips: list[CuratedClip]
    golden_refs: list[CuratedClip]
    f5_metadata_path: Path
    gpt_sovits_list_path: Path
    profile_path: Path


def _strip_vietnamese_tone(word: str) -> str:
    decomposed = unicodedata.normalize('NFD', word.lower())
    return ''.join(ch for ch in decomposed if not unicodedata.combining(ch))


def _count_tone_marks(text: str) -> int:
    """Number of distinct Vietnamese tone marks present (0..5)."""
    decomposed = unicodedata.normalize('NFD', text.lower())
    present = {ch for ch in decomposed if ch in TONE_MARKS}
    return len(present)


def _is_word_like(word: str) -> bool:
    stripped = _strip_vietnamese_tone(word).strip()
    return any(ch.isalnum() for ch in stripped)


def _sanitize_transcript(words: Sequence[TranscribedWord]) -> str:
    """Join words into a single-line transcript, dropping annotation noise."""
    tokens = [w.word.strip() for w in words if w.word and w.word.strip()]
    tokens = [t for t in tokens if not (t.startswith('[') and t.endswith(']'))]
    text = ' '.join(tokens)
    return re.sub(r'\s+', ' ', text).strip()


def select_golden_references(clips: list[CuratedClip], top_k: int = 3) -> list[CuratedClip]:
    """Pick the best zero-shot reference clips.

    Prefers the ``7.0s - 11.0s`` window, Vietnamese tone diversity
    (``4/5`` distinct tone marks), then the highest combined
    SNR + word-confidence quality score. Falls back to the best remaining
    clips when fewer than ``top_k`` clips fall inside the ideal window.
    """
    if not clips or top_k <= 0:
        return []

    def rank(clip: CuratedClip) -> tuple:
        in_window = GOLDEN_MIN_SEC <= clip.duration_sec <= GOLDEN_MAX_SEC
        tone_marks = _count_tone_marks(clip.text)
        return (bool(in_window), int(tone_marks >= 4), tone_marks, clip.quality_score, clip.duration_sec)

    # Stable ordering: input order breaks ties deterministically.
    ordered = sorted(clips, key=rank, reverse=True)
    return ordered[:top_k]


def duplicate_token_ratio(words: Sequence[TranscribedWord]) -> float:
    """Fraction of words occupied by the single most repeated token.

    Whisper's classic failure mode is emitting one token hundreds of times
    ("một một một một ..."); such a clip is unusable for voice cloning.
    """
    tokens = [_strip_vietnamese_tone(w.word).strip().lower() for w in words if w.word and w.word.strip()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 1.0
    most_common = max(set(tokens), key=tokens.count)
    return tokens.count(most_common) / len(tokens)


def _quality_score(duration_sec: float, snr_db: float, min_word_prob: float) -> float:
    """Combined SNR + word-confidence score, biased toward 7-11s clips."""
    ideal_duration = 9.0
    duration_term = 1.0 - min(abs(duration_sec - ideal_duration) / ideal_duration, 1.0)
    snr_term = min(max(snr_db, 0.0) / 30.0, 1.0)
    prob_term = min(max(min_word_prob, 0.0), 1.0)
    return float(0.35 * duration_term + 0.30 * snr_term + 0.35 * prob_term)


def _load_profile(voice_dir: Path) -> dict:
    profile_path = voice_dir / 'profile.json'
    if not profile_path.exists():
        return {}
    try:
        return json.loads(profile_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        logger.warning(f'⚠️ [PROFILE] Không đọc được {profile_path}, tạo mới.')
        return {}


def _processed_video_ids(voice_dir: Path) -> list[str]:
    raw = _load_profile(voice_dir).get('processed_video_ids') or []
    return [str(v) for v in raw]


def _clean_transcript_for_export(text: str) -> str:
    """Transcript used in the dataset files: no trailing '.' (spec sample)."""
    return text.strip().rstrip('.').strip()


def curate_and_export_dataset(
    voice_id: str,
    candidates: list[RawAudioCandidate],
    voices_root: Path,
    transcribe_fn: Callable[[Path], Sequence[TranscribedWord]],
    processed_video_ids: list[str] | None = None,
    min_word_prob: float = MIN_WORD_PROB,
) -> VoiceHarvestResult:
    """Apply the word-confidence gate and export F5-TTS + GPT-SoVITS datasets.

    Any clip with a single word below ``min_word_prob``, or with an abnormal
    speaking rate (``8-28`` chars/sec), is discarded. Survivors are copied to
    ``voices/<voice_id>/wavs/clip_XXXX.wav``, indexed in
    ``metadata.csv`` (F5-TTS) and ``gpt_sovits.list``, and the pipeline state
    (including ``processed_video_ids``) is persisted to ``profile.json``.
    """
    voice_dir = Path(voices_root) / voice_id
    wavs_dir = voice_dir / 'wavs'
    golden_dir = voice_dir / 'golden_refs'
    wavs_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    kept: list[CuratedClip] = []
    for candidate in candidates:
        try:
            raw_words = transcribe_fn(Path(candidate.path))
        except Exception as exc:  # a single bad clip must not lose the whole dataset
            logger.error(f'❌ [ASR] Nhận dạng lỗi cho {Path(candidate.path).name}: {exc}')
            continue
        words = [TranscribedWord(str(w.word), float(w.probability)) for w in raw_words]
        words = [w for w in words if _is_word_like(w.word)]
        if not words:
            logger.info(f'🚫 [ASR] Loại {Path(candidate.path).name}: không nhận dạng được chữ')
            continue

        min_prob = min(w.probability for w in words)
        if min_prob < min_word_prob:
            logger.info(f'🚫 [ASR] Loại {Path(candidate.path).name}: word_prob={min_prob:.3f} < {min_word_prob}')
            continue

        text = _sanitize_transcript(words)
        if not text:
            continue

        export_text = _clean_transcript_for_export(text)
        if not export_text:
            continue

        duration_sec = float(candidate.duration_sec)
        # Whisper returns timestamped word tokens (whitespace pieces). The spec
        # writes "8-28 ký tự/giây", which is loose; 8 tokens/sec is exactly the
        # spec's worked constant 64/8, and the floor is what rejects a
        # transcription whose timings do not line up with the clip duration
        # (hallucination/merged-word artefact). Upper bound 28 words/sec.
        tokens_per_sec = len(words) / max(duration_sec, 1e-6)
        if tokens_per_sec > MAX_WORDS_PER_SEC:
            logger.info(
                f'🚫 [ASR] Loại {Path(candidate.path).name}: tốc độ {tokens_per_sec:.1f} từ/giây '
                f'> {MAX_WORDS_PER_SEC} (nghi ngờ lỗi nhận dạng)'
            )
            continue
        if duplicate_token_ratio(words) >= MAX_DUPLICATE_TOKEN_RATIO:
            logger.info(
                f'🚫 [ASR] Loại {Path(candidate.path).name}: lặp từ bất thường (hallucination)'
            )
            continue

        clip_name = f'clip_{len(kept) + 1:04d}.wav'
        clip_path = wavs_dir / clip_name
        shutil.copyfile(candidate.path, clip_path)

        kept.append(
            CuratedClip(
                clip_path=clip_path,
                text=export_text,
                duration_sec=duration_sec,
                snr_db=float(candidate.vocal_to_bgm_db),
                min_word_prob=float(min_prob),
                quality_score=_quality_score(duration_sec, float(candidate.vocal_to_bgm_db), float(min_prob)),
            )
        )

    golden_refs = select_golden_references(kept, top_k=3)
    for idx, golden in enumerate(golden_refs, start=1):
        destination = golden_dir / f'golden_{idx:02d}.wav'
        shutil.copyfile(golden.clip_path, destination)

    f5_metadata_path = voice_dir / 'metadata.csv'
    f5_metadata_path.write_text(
        ''.join(f'wavs/{clip.clip_path.name}|{clip.text}\n' for clip in kept),
        encoding='utf-8',
    )

    gpt_sovits_list_path = voice_dir / 'gpt_sovits.list'
    gpt_sovits_list_path.write_text(
        ''.join(f'wavs/{clip.clip_path.name}|{voice_id}|VI|{clip.text}\n' for clip in kept),
        encoding='utf-8',
    )

    total_clean_sec = float(sum(clip.duration_sec for clip in kept))
    best = golden_refs[0] if golden_refs else None
    history = list(processed_video_ids or [])
    for video_id in _processed_video_ids(voice_dir):
        if video_id not in history:
            history.append(video_id)

    profile = {
        'voice_id': voice_id,
        'processed_video_ids': history,
        'total_clean_sec': total_clean_sec,
        'total_clean_minutes': total_clean_sec / 60.0,
        'num_clips': len(kept),
        'best_golden_ref_wav': str(Path('golden_refs') / 'golden_01.wav') if best else '',
        'best_golden_ref_text': best.text if best else '',
        'golden_refs': [
            {
                'wav': str(Path('golden_refs') / f'golden_{i:02d}.wav'),
                'text': clip.text,
                'duration_sec': clip.duration_sec,
                'quality_score': clip.quality_score,
            }
            for i, clip in enumerate(golden_refs, start=1)
        ],
        'clips': [asdict(clip) | {'clip_path': str(clip.clip_path)} for clip in kept],
    }
    profile_path = voice_dir / 'profile.json'
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')

    logger.info(
        f'📦 [EXPORT] {voice_id}: giữ {len(kept)} câu / {total_clean_sec / 60.0:.1f} phút sạch, '
        f'{len(golden_refs)} golden refs -> {voice_dir}'
    )
    return VoiceHarvestResult(
        voice_id=voice_id,
        voice_dir=voice_dir,
        total_clean_sec=total_clean_sec,
        kept_clips=kept,
        golden_refs=golden_refs,
        f5_metadata_path=f5_metadata_path,
        gpt_sovits_list_path=gpt_sovits_list_path,
        profile_path=profile_path,
    )


# ---------------------------------------------------------------------------
# Step 1-2 (pipeline level) — YouTube discovery, download, separation, ASR
# ---------------------------------------------------------------------------

_YT_DLP_CACHE: list[str] | None = None
_WATCH_URL_RE = re.compile(r'(?:youtube\.com/watch\?[^ ]*?v=|youtu\.be/|/shorts/|/live/)([A-Za-z0-9_-]{11})')
_BARE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')


def _yt_dlp_cmd() -> list[str]:
    """Resolve the yt-dlp executable once (PATH, venv, then module fallback)."""
    global _YT_DLP_CACHE
    if _YT_DLP_CACHE is not None:
        return _YT_DLP_CACHE

    candidates: list[list[str]] = []
    found = shutil.which('yt-dlp')
    if found:
        candidates.append([found])

    venv_bin = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'yt-dlp'
    if venv_bin.exists():
        candidates.append([str(venv_bin)])

    for candidate in candidates:
        try:
            proc = subprocess.run([*candidate, '--version'], capture_output=True, text=True, timeout=30)
            if proc.returncode == 0:
                _YT_DLP_CACHE = candidate
                return candidate
        except (OSError, subprocess.SubprocessError):
            continue

    _YT_DLP_CACHE = [sys.executable, '-m', 'yt_dlp']
    return _YT_DLP_CACHE


def _ffmpeg_dir_candidates() -> list[Path]:
    """Directories that may hold ffmpeg/ffprobe binaries."""
    repo_root = Path(__file__).resolve().parent.parent
    dirs: list[Path] = []
    venv_bin = repo_root / '.venv' / 'bin'
    if venv_bin.exists():
        dirs.append(venv_bin)
    for site_packages in sorted(venv_bin.parent.glob('lib/python*/site-packages')):
        for rel in ('static_ffmpeg/bin/linux', 'static_ffmpeg/bin'):
            candidate = site_packages / rel
            if candidate.exists():
                dirs.append(candidate)
    return dirs


def _resolve_ffmpeg() -> str:
    venv_ffmpeg = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'ffmpeg'
    if venv_ffmpeg.exists():
        return str(venv_ffmpeg)
    return shutil.which('ffmpeg') or 'ffmpeg'


def _resolve_ffmpeg_location() -> str | None:
    """Directory containing ffmpeg/ffprobe to hand yt-dlp (or ``None``)."""
    for directory in _ffmpeg_dir_candidates():
        if (directory / 'ffmpeg').exists() and (directory / 'ffprobe').exists():
            return str(directory)
    return None


def _is_probably_url(value: str) -> bool:
    return value.startswith('http://') or value.startswith('https://')


def _extract_video_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in _WATCH_URL_RE.finditer(text or ''):
        if match.group(1) not in ids:
            ids.append(match.group(1))
    return ids


def _probe_duration_sec(url: str, timeout: float = 60.0) -> float:
    """Ask yt-dlp for a video duration; ``0.0`` means "unknown"."""
    cmd = [*_yt_dlp_cmd(), '--skip-download', '--print', '%(duration)s', '--no-warnings', url]
    ffmpeg_location = _resolve_ffmpeg_location()
    if ffmpeg_location:
        cmd += ['--ffmpeg-location', ffmpeg_location]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f'⚠️ [YT-DLP] Không dò được thời lượng {url}: {exc}')
        return 0.0
    if proc.returncode != 0:
        return 0.0
    for line in reversed(proc.stdout.strip().splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    return 0.0


def _search_youtube_ids(query: str, limit: int, timeout: float = 90.0) -> list[str]:
    """``ytsearchN:query`` -> ordered list of video ids."""
    cmd = [
        *_yt_dlp_cmd(),
        f'ytsearch{max(1, limit)}:{query}',
        '--flat-playlist',
        '--print',
        '%(id)s',
        '--no-warnings',
        '--skip-download',
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error(f'❌ [YT-DLP] Tìm kiếm thất bại cho "{query}": {exc}')
        return []
    if proc.returncode != 0:
        logger.error(f'❌ [YT-DLP] Tìm kiếm lỗi: {proc.stderr.strip()[:300]}')
        return []
    ids: list[str] = []
    for line in proc.stdout.strip().splitlines():
        candidate = line.strip()
        if candidate and candidate not in ids:
            ids.append(candidate)
    return ids


def _video_url(video_id: str) -> str:
    return f'https://www.youtube.com/watch?v={video_id}'


def _default_download(video_id: str, out_dir: Path, skip_intro_sec: float = INTRO_SKIP_SEC) -> Path:
    """Download one video's audio and crop intro/outro, yielding 24kHz mono WAV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_template = str(out_dir / f'{video_id}_raw.%(ext)s')
    dl_cmd = [
        *_yt_dlp_cmd(),
        '-x',
        '--audio-format',
        'wav',
        '--audio-quality',
        '0',
        '-o',
        raw_template,
        '--no-warnings',
    ]
    ffmpeg_location = _resolve_ffmpeg_location()
    if ffmpeg_location:
        dl_cmd += ['--ffmpeg-location', ffmpeg_location]
    dl_cmd.append(_video_url(video_id))
    logger.info(f'⬇️ [YT-DLP] Tải audio: {_video_url(video_id)}')
    proc = subprocess.run(dl_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'yt-dlp lỗi khi tải {video_id}: {(proc.stderr or proc.stdout).strip()[-500:]}')

    downloaded = sorted(out_dir.glob(f'{video_id}_raw.*'))
    if not downloaded:
        raise FileNotFoundError(f'yt-dlp không tạo file cho {video_id}')

    target = out_dir / f'{video_id}.wav'
    probe = _probe_duration_sec(_video_url(video_id))
    crop_end = max(0.0, probe - OUTRO_SKIP_SEC) if probe > 0 else 0.0
    ffmpeg = _resolve_ffmpeg()
    cmd = [ffmpeg, '-y', '-i', str(downloaded[0])]
    if skip_intro_sec > 0:
        cmd += ['-ss', f'{skip_intro_sec:.3f}']
    if crop_end > skip_intro_sec:
        cmd += ['-to', f'{crop_end:.3f}']
    cmd += ['-ac', str(CHANNELS), '-ar', str(SAMPLE_RATE), '-c:a', 'pcm_s16le', str(target)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    for stale in downloaded:
        stale.unlink(missing_ok=True)
    return target


def _default_separate(wav_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Split ``wav_path`` into ``(vocal.wav, inst.wav)``.

    Uses ``audio-separator`` (UVR5 MDX-Net) when installed; otherwise falls
    back to an FFmpeg high-pass "vocal band" filter plus mid-side instrumental
    subtraction so the pipeline still runs end-to-end offline.
    """
    wav_path = Path(wav_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocal_out = out_dir / f'{wav_path.stem}_vocal.wav'
    inst_out = out_dir / f'{wav_path.stem}_inst.wav'

    separator = shutil.which('audio-separator')
    if separator:
        cmd = [
            separator,
            str(wav_path),
            '--model_filename',
            'UVR-MDX-NET-Inst_HQ_3.onnx',
            '--output_dir',
            str(out_dir),
            '--output_format',
            'wav',
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            vocals = sorted(out_dir.glob('*vocal*.wav'))
            insts = sorted(out_dir.glob('*instrumental*.wav')) or sorted(out_dir.glob('*inst*.wav'))
            if vocals and insts:
                shutil.move(str(vocals[0]), str(vocal_out))
                shutil.move(str(insts[0]), str(inst_out))
                return vocal_out, inst_out
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(f'⚠️ [UVR] audio-separator lỗi ({exc}); dùng fallback FFmpeg.')

    logger.warning('⚠️ [UVR] Không có audio-separator; dùng fallback bộ lọc phổ FFmpeg (kém chính xác hơn).')
    ffmpeg = _resolve_ffmpeg()
    # "Vocal" = speech band. The instrumental estimate is *derived* from the
    # vocal band (voice-in and voice-out of the 200-4500 Hz speech band), not
    # from a separate band of the same recording: a bass-band "instrumental"
    # would still contain the voice and the Vocal-to-BGM gate would then always
    # measure ~0 dB and reject everything.
    subprocess.run(
        [
            ffmpeg,
            '-y',
            '-i',
            str(wav_path),
            '-af',
            'highpass=f=180,lowpass=f=4500',
            '-ac',
            str(CHANNELS),
            '-ar',
            str(SAMPLE_RATE),
            '-c:a',
            'pcm_s16le',
            str(vocal_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            ffmpeg,
            '-y',
            '-i',
            str(wav_path),
            '-af',
            'lowpass=f=200,highpass=f=60,highpass=f=4500',
            '-ac',
            str(CHANNELS),
            '-ar',
            str(SAMPLE_RATE),
            '-c:a',
            'pcm_s16le',
            str(inst_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return vocal_out, inst_out


_WHISPER_CACHE: dict[str, Any] = {}


def _load_whisper_model(model_size: str, device: str, compute_type: str):
    key = f'{model_size}|{device}|{compute_type}'
    if key in _WHISPER_CACHE:
        return _WHISPER_CACHE[key]
    from faster_whisper import WhisperModel  # imported lazily: heavy GPU dependency

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _WHISPER_CACHE[key] = model
    return model


def _default_transcribe(
    wav_path: Path,
    model_size: str = 'large-v3',
    device: str = 'auto',
    compute_type: str = 'float16',
    language: str = 'vi',
) -> list[TranscribedWord]:
    """faster-whisper word-level transcription (RTX 3060, FP16 by default)."""
    model = _load_whisper_model(model_size, device, compute_type)
    segments, _info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        vad_filter=False,
    )
    words: list[TranscribedWord] = []
    for segment in segments:
        for word in segment.words or []:
            words.append(TranscribedWord(word.word, float(word.probability)))
    return words


def discover_and_download_videos(
    voice_id: str,
    query_or_url: str,
    voices_root: Path,
    video_ids: list[str] | None = None,
    max_videos: int = 3,
    download_fn: Callable[[str, Path], Path] | None = None,
) -> list[Path]:
    """Find candidate YouTube videos and download the ones not yet harvested.

    ``video_ids`` already present in ``voices/<voice_id>/profile.json`` are skipped
    so incremental runs never duplicate clips. Videos shorter than ``120s``
    (Shorts) or longer than ``3600s`` (livestreams) are rejected.
    """
    voice_dir = Path(voices_root) / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = voice_dir / 'downloads'
    downloads_dir.mkdir(parents=True, exist_ok=True)
    processed = set(_processed_video_ids(voice_dir))

    resolved_ids = list(video_ids or [])
    if not resolved_ids:
        if _is_probably_url(query_or_url):
            resolved_ids = _extract_video_ids(query_or_url)
            if not resolved_ids and _BARE_ID_RE.match(query_or_url.strip()):
                resolved_ids = [query_or_url.strip()]
        else:
            resolved_ids = _search_youtube_ids(query_or_url, limit=max_videos * 2)

    download = download_fn or _default_download
    downloaded: list[Path] = []
    for video_id in resolved_ids:
        if len(downloaded) >= max_videos:
            break
        if not video_id or video_id in processed:
            if video_id in processed:
                logger.info(f'⏩ [DEDUP] Bỏ qua video đã cào trước đó: {video_id}')
            continue

        if download_fn is None:
            duration = _probe_duration_sec(_video_url(video_id))
            if duration and not (MIN_VIDEO_SEC <= duration <= MAX_VIDEO_SEC):
                logger.info(f'⏩ [FILTER] Bỏ qua {video_id}: thời lượng {duration:.0f}s ngoài khoảng cho phép')
                continue

        try:
            path = Path(download(video_id, downloads_dir))
        except Exception as exc:  # network/yt-dlp/ffmpeg errors must not kill the run
            logger.error(f'❌ [DOWNLOAD] Lỗi tải {video_id}: {exc}')
            continue

        processed.add(video_id)
        downloaded.append(path)
        logger.info(f'✔️ [DOWNLOAD] {video_id} -> {path}')

    return downloaded


def harvest_voice_from_youtube(
    voice_id: str,
    query_or_url: str,
    voices_root: Path,
    target_clean_minutes: float = 20.0,
    max_videos: int = 3,
    download_fn: Callable[[str, Path], Path] | None = None,
    separate_fn: Callable[[Path, Path], tuple[Path, Path]] | None = None,
    transcribe_fn: Callable[[Path], Sequence[TranscribedWord]] | None = None,
) -> VoiceHarvestResult:
    """Run the full 5-stage harvest pipeline for one voice.

    Every heavy stage is injectable (``download_fn``, ``separate_fn``,
    ``transcribe_fn``) so the whole pipeline is testable offline; the real
    defaults invoke ``yt-dlp``, ``audio-separator`` and ``faster-whisper``.
    """
    voice_dir = Path(voices_root) / voice_id
    voice_dir.mkdir(parents=True, exist_ok=True)
    work_dir = voice_dir / '_work'
    split_dir = work_dir / 'separated'
    clips_dir = work_dir / 'clips'

    target_clean_sec = max(0.0, float(target_clean_minutes) * 60.0)
    downloaded = discover_and_download_videos(
        voice_id=voice_id,
        query_or_url=query_or_url,
        voices_root=voices_root,
        max_videos=max_videos,
        download_fn=download_fn,
    )
    logger.info(f'🎬 [HARVEST] {voice_id}: {len(downloaded)} video mới cần xử lý')

    separate = separate_fn or _default_separate
    transcribe = transcribe_fn or _default_transcribe

    candidates: list[RawAudioCandidate] = []
    collected_sec = 0.0
    for path in downloaded:
        try:
            vocal_wav, inst_wav = separate(Path(path), split_dir)
        except Exception as exc:  # separation is best-effort per video
            logger.error(f'❌ [UVR] Tách nhạc nền lỗi cho {path}: {exc}')
            continue

        clip_out = clips_dir / Path(path).stem
        try:
            segments = slice_and_filter_segments(vocal_wav=vocal_wav, out_dir=clip_out, inst_wav=inst_wav)
        except Exception as exc:
            logger.error(f'❌ [SLICE] Cắt câu lỗi cho {path}: {exc}')
            continue
        segments = filter_dominant_speaker(segments)
        candidates.extend(segments)
        collected_sec += sum(seg.duration_sec for seg in segments)
        if collected_sec >= target_clean_sec:
            logger.info(f'🎯 [HARVEST] Đã đủ nguyên liệu thô ({collected_sec / 60.0:.1f} phút), dừng tải thêm video.')
            break

    result = curate_and_export_dataset(
        voice_id=voice_id,
        candidates=candidates,
        voices_root=voices_root,
        transcribe_fn=transcribe,
        processed_video_ids=[Path(p).stem for p in downloaded],
        min_word_prob=MIN_WORD_PROB,
    )
    logger.info(
        f'🎉 [HARVEST] {voice_id}: {len(result.kept_clips)} câu / '
        f'{result.total_clean_sec / 60.0:.1f} phút sạch sẵn sàng cho F5-TTS & GPT-SoVITS'
    )
    return result


def list_harvested_voices(voices_root: Path) -> list[dict]:
    """List harvested voice profiles under ``voices_root`` (newest first)."""
    root = Path(voices_root)
    if not root.exists():
        return []

    voices: list[dict] = []
    for profile_path in sorted(root.glob('*/profile.json')):
        try:
            profile = json.loads(profile_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f'⚠️ [PROFILE] Bỏ qua {profile_path}: {exc}')
            continue

        voice_dir = profile_path.parent
        golden_rel = profile.get('best_golden_ref_wav') or _first_golden_rel(voice_dir)
        golden_wav = voice_dir / golden_rel if golden_rel else None
        voices.append(
            {
                'voice_id': profile.get('voice_id') or voice_dir.name,
                'voice_dir': str(voice_dir),
                'profile_path': str(profile_path),
                'total_clean_minutes': float(
                    profile.get('total_clean_minutes')
                    or (float(profile.get('total_clean_sec') or 0.0) / 60.0)
                ),
                'num_clips': int(profile.get('num_clips') or len(profile.get('clips') or [])),
                'golden_ref_wav': str(golden_wav) if golden_wav and golden_wav.exists() else '',
                'golden_ref_text': profile.get('best_golden_ref_text') or '',
                'processed_video_ids': [str(v) for v in (profile.get('processed_video_ids') or [])],
                'modified_at': profile_path.stat().st_mtime,
            }
        )
    voices.sort(key=lambda v: v['modified_at'], reverse=True)
    return voices


def _first_golden_rel(voice_dir: Path) -> str:
    for candidate in sorted((voice_dir / 'golden_refs').glob('*.wav')):
        return str(Path('golden_refs') / candidate.name)
    return ''
