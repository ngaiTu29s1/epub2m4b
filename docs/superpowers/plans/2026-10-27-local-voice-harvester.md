# Local Autonomous Voice Harvester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained local automated pipeline (`core/voice_harvester.py` + Gradio Voice Harvester Studio tab in `app.py`) that searches/downloads YouTube audio, isolates vocals, filters segments via the 4 Emilia-Pipe quality gates (`Vocal-to-BGM >= 8dB`, `+200ms` padding, dominant speaker cosine similarity `>= 0.78`, and word-level ASR confidence `>= 0.92`), selects Golden 10s reference clips, and exports ready-to-use `F5-TTS` and `GPT-SoVITS` datasets.

**Architecture:** A modular pipeline in `core/voice_harvester.py` with pluggable stages (Downloader, VocalSeparator, AudioSlicer, SpeakerClusterer, Transcriber) so unit tests run deterministically offline in milliseconds while production calls execute `yt-dlp`, `audio-separator`/`ffmpeg`, acoustic embedding clustering, and `faster-whisper` on the local RTX 3060 12GB GPU. Results are persisted under `voices/<voice_id>/` (`profile.json`, `golden_refs/`, `wavs/`, `metadata.csv`, `gpt_sovits.list`) and exposed in `app.py` so harvested voices can immediately be selected to render `.m4b` audiobooks.

**Tech Stack:** Python 3.12, `numpy`, `soundfile`, `ffmpeg` (`.venv/bin/ffmpeg`), `yt-dlp` (CLI/subprocess), `faster-whisper` / pluggable ASR callable, `gradio`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-27-local-voice-harvester-design.md`

## Global Constraints

- Sample rate for all harvested clips: `24000` Hz, `1` channel (Mono), `PCM_16` (`s16`).
- Clip duration bounds: minimum `3.5` seconds, maximum `12.0` seconds.
- Mandatory silence padding at both ends of every sliced clip: `pad_ms = 200` (`+200ms` head and `+200ms` tail).
- Vocal-to-BGM energy gate: `vocal_to_bgm_db >= 8.0` dB.
- Clipping & silence gates: reject clips with `peak_dbfs >= -0.2` dBFS or `rms_dbfs < -42.0` dBFS.
- Dominant speaker similarity threshold: `cosine_similarity >= 0.78` against the dominant cluster centroid.
- Word-level transcription confidence gate: `min_word_prob >= 0.92` and speaking rate between `8.0` and `28.0` characters/second.
- All voice profiles and datasets live under `voices/<voice_id>/` (ignored by git via `.gitignore`).

## Review Focus

- Video already harvested in `profile.json`: `discover_and_download_videos` must skip any `video_id` already listed in `processed_video_ids` so incremental runs never duplicate clips.
- High background music segment (`vocal_to_bgm_db < 8.0` dB): `slice_and_filter_segments` must drop segments where background music energy is within 8 dB of vocal energy.
- Guest speaker / advertisement clip in the middle of a video: `filter_dominant_speaker` must drop outlier embeddings whose cosine similarity to the median/dominant speaker centroid is `< 0.78`.
- Whisper transcription with a single uncertain Vietnamese tone word (`word_prob < 0.92`): `curate_and_export_dataset` must discard any clip where even one word has probability `< 0.92`.
- Golden 10s reference selection when multiple valid clips exist: `select_golden_references` must prefer clips in the `7.0s - 11.0s` window that contain diverse Vietnamese diacritics (`àáảãạ...`) and highest combined SNR + word confidence.

---

### Task 1: Acoustic Slicer, `+200ms` Padding & `Vocal-to-BGM >= 8dB` Gate (`core/voice_harvester.py`)

**Files:**
- Create: `core/voice_harvester.py`
- Modify: `.gitignore`
- Test: `tests/test_voice_harvester.py`

**Interfaces:**
- Produces:
  - `@dataclass class RawAudioCandidate: path: Path; start_sec: float; end_sec: float; vocal_to_bgm_db: float; rms_dbfs: float; peak_dbfs: float`
  - `compute_rms_dbfs(samples: np.ndarray) -> float`
  - `slice_and_filter_segments(vocal_wav: Path, out_dir: Path, inst_wav: Path | None = None, min_sec: float = 3.5, max_sec: float = 12.0, pad_ms: int = 200, min_snr_db: float = 8.0) -> list[RawAudioCandidate]`

- [ ] **Step 1: Write the failing test in `tests/test_voice_harvester.py` for slicing with `+200ms` padding and rejecting low Vocal-to-BGM (`< 8.0 dB`) or clipped segments**

```python
from pathlib import Path
import numpy as np
import soundfile as sf
from core.voice_harvester import slice_and_filter_segments


def _write_test_wav(path: Path, segments_amp: list[tuple[float, float]], sr: int = 24000) -> Path:
    chunks = []
    for dur_sec, amp in segments_amp:
        n = int(dur_sec * sr)
        t = np.linspace(0, dur_sec, n, endpoint=False)
        wave = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32) if amp > 0 else np.zeros(n, dtype=np.float32)
        chunks.append(wave)
    audio = np.concatenate(chunks)
    sf.write(str(path), audio, sr, subtype='PCM_16')
    return path


def test_slice_and_filter_keeps_padded_clean_speech_and_drops_loud_bgm(tmp_path: Path):
    # Segment 1: 4.5s clean vocal (amp 0.3), 0.8s silence, Segment 2: 4.5s vocal with loud BGM
    vocal_path = _write_test_wav(
        tmp_path / 'vocal.wav',
        [(1.0, 0.0), (4.5, 0.3), (0.8, 0.0), (4.5, 0.3), (1.0, 0.0)],
    )
    inst_path = _write_test_wav(
        tmp_path / 'inst.wav',
        [(1.0, 0.0), (4.5, 0.02), (0.8, 0.0), (4.5, 0.22), (1.0, 0.0)],
    )
    kept = slice_and_filter_segments(
        vocal_wav=vocal_path,
        out_dir=tmp_path / 'clips',
        inst_wav=inst_path,
        min_sec=3.5,
        max_sec=12.0,
        pad_ms=200,
        min_snr_db=8.0,
    )
    assert len(kept) == 1
    assert kept[0].vocal_to_bgm_db >= 8.0
    # 4.5s speech + 0.2s head pad + 0.2s tail pad = ~4.9s
    assert 4.7 <= (kept[0].end_sec - kept[0].start_sec) <= 5.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_voice_harvester.py::test_slice_and_filter_keeps_padded_clean_speech_and_drops_loud_bgm -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.voice_harvester'`

- [ ] **Step 3: Implement `RawAudioCandidate`, `compute_rms_dbfs`, and `slice_and_filter_segments` in `core/voice_harvester.py` (and add `voices/` to `.gitignore`)**

Use `soundfile` and `numpy` on `50ms` frames (`1200` samples at `24000` Hz) to detect speech islands (`frame_rms_dbfs > -38.0`), merge islands separated by `< 0.35s` silence, split islands longer than `max_sec` at the quietest internal valley, add `pad_ms` (`200ms`) on both ends, and filter by `min_sec <= duration <= max_sec`, `vocal_to_bgm_db >= min_snr_db`, `peak_dbfs < -0.2`, and `rms_dbfs >= -42.0`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_voice_harvester.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .gitignore core/voice_harvester.py tests/test_voice_harvester.py
git commit -m "feat(harvester): add acoustic VAD slicer with 200ms padding and Vocal-to-BGM gate"
```

---

### Task 2: Dominant Speaker Clustering & Outlier Rejection (`core/voice_harvester.py`)

**Files:**
- Modify: `core/voice_harvester.py`
- Test: `tests/test_voice_harvester.py`

**Interfaces:**
- Consumes: `RawAudioCandidate`
- Produces:
  - `compute_speaker_embedding(wav_path: Path) -> np.ndarray` (L2-normalized spectral/pitch/formant feature vector of shape `(D,)`)
  - `filter_dominant_speaker(candidates: list[RawAudioCandidate], min_similarity: float = 0.78, embed_fn=compute_speaker_embedding) -> list[RawAudioCandidate]`

- [ ] **Step 1: Write the failing test in `tests/test_voice_harvester.py` for dominant speaker clustering**

```python
from core.voice_harvester import RawAudioCandidate, filter_dominant_speaker


def test_filter_dominant_speaker_drops_guest_voice_outliers(tmp_path: Path):
    c1 = RawAudioCandidate(tmp_path / '1.wav', 0, 5, 15.0, -18.0, -3.0)
    c2 = RawAudioCandidate(tmp_path / '2.wav', 5, 10, 15.0, -18.0, -3.0)
    c3 = RawAudioCandidate(tmp_path / '3.wav', 10, 15, 15.0, -18.0, -3.0)
    guest = RawAudioCandidate(tmp_path / 'guest.wav', 15, 20, 15.0, -18.0, -3.0)

    fake_vecs = {
        c1.path: np.array([0.98, 0.15, 0.05], dtype=np.float32),
        c2.path: np.array([0.97, 0.18, 0.04], dtype=np.float32),
        c3.path: np.array([0.99, 0.12, 0.06], dtype=np.float32),
        guest.path: np.array([0.10, 0.95, -0.25], dtype=np.float32),
    }
    kept = filter_dominant_speaker(
        [c1, c2, c3, guest],
        min_similarity=0.78,
        embed_fn=lambda p: fake_vecs[p] / np.linalg.norm(fake_vecs[p]),
    )
    assert [k.path.name for k in kept] == ['1.wav', '2.wav', '3.wav']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_voice_harvester.py::test_filter_dominant_speaker_drops_guest_voice_outliers -v`
Expected: FAIL with `ImportError: cannot import name 'filter_dominant_speaker'`

- [ ] **Step 3: Implement `compute_speaker_embedding` and `filter_dominant_speaker` in `core/voice_harvester.py`**

Compute L2-normalized log-mel / harmonic-spectral feature statistics per clip, find the dense medoid/median centroid (the vector with highest average cosine similarity to its top 60% neighbors so guest voices never skew the mean), and retain only candidates with `float(np.dot(vec, centroid)) >= min_similarity`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_voice_harvester.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/voice_harvester.py tests/test_voice_harvester.py
git commit -m "feat(harvester): add dominant speaker clustering to reject guest voices"
```

---

### Task 3: Word-Level Whisper Gate (`>= 0.92`), Golden 10s Selector & Multi-Format Dataset Exporter (`core/voice_harvester.py`)

**Files:**
- Modify: `core/voice_harvester.py`
- Test: `tests/test_voice_harvester.py`

**Interfaces:**
- Consumes: `RawAudioCandidate`
- Produces:
  - `@dataclass class TranscribedWord: word: str; probability: float`
  - `@dataclass class CuratedClip: clip_path: Path; text: str; duration_sec: float; snr_db: float; min_word_prob: float; quality_score: float`
  - `@dataclass class VoiceHarvestResult: voice_id: str; voice_dir: Path; total_clean_sec: float; kept_clips: list[CuratedClip]; golden_refs: list[CuratedClip]; f5_metadata_path: Path; gpt_sovits_list_path: Path; profile_path: Path`
  - `select_golden_references(clips: list[CuratedClip], top_k: int = 3) -> list[CuratedClip]`
  - `curate_and_export_dataset(voice_id: str, candidates: list[RawAudioCandidate], voices_root: Path, transcribe_fn, processed_video_ids: list[str] | None = None, min_word_prob: float = 0.92) -> VoiceHarvestResult`

- [ ] **Step 1: Write the failing test in `tests/test_voice_harvester.py` for word-confidence filtering (`< 0.92` dropped), Golden 10s selection, and F5-TTS/GPT-SoVITS export**

```python
from core.voice_harvester import (
    RawAudioCandidate,
    TranscribedWord,
    curate_and_export_dataset,
)


def test_curate_and_export_drops_low_word_prob_and_exports_golden_refs(tmp_path: Path):
    w1 = _write_test_wav(tmp_path / 'c1.wav', [(8.5, 0.25)])
    w2 = _write_test_wav(tmp_path / 'c2.wav', [(5.0, 0.25)])
    c1 = RawAudioCandidate(w1, 0.0, 8.5, 18.0, -16.0, -2.0)
    c2 = RawAudioCandidate(w2, 8.5, 13.5, 14.0, -18.0, -3.0)

    def fake_transcribe(wav_path: Path) -> list[TranscribedWord]:
        if wav_path == w1:
            return [
                TranscribedWord('Xin', 0.98),
                TranscribedWord('chào', 0.96),
                TranscribedWord('mọi', 0.95),
                TranscribedWord('người', 0.97),
                TranscribedWord('hôm', 0.99),
                TranscribedWord('nay', 0.96),
                TranscribedWord('trời', 0.95),
                TranscribedWord('rất', 0.97),
                TranscribedWord('đẹp', 0.98),
                TranscribedWord('và', 0.99),
                TranscribedWord('nắng', 0.94),
                TranscribedWord('ấm.', 0.95),
            ]
        return [
            TranscribedWord('Câu', 0.99),
            TranscribedWord('này', 0.81),  # < 0.92 -> must be dropped!
            TranscribedWord('bị', 0.95),
            TranscribedWord('ngọng.', 0.95),
        ]

    res = curate_and_export_dataset(
        voice_id='vui_ve',
        candidates=[c1, c2],
        voices_root=tmp_path / 'voices',
        transcribe_fn=fake_transcribe,
        processed_video_ids=['vid_001'],
        min_word_prob=0.92,
    )
    assert len(res.kept_clips) == 1
    assert len(res.golden_refs) == 1
    assert res.f5_metadata_path.exists()
    assert 'wavs/clip_0001.wav|Xin chào mọi người' in res.f5_metadata_path.read_text(encoding='utf-8')
    assert '|vui_ve|VI|Xin chào mọi người' in res.gpt_sovits_list_path.read_text(encoding='utf-8')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_voice_harvester.py::test_curate_and_export_drops_low_word_prob_and_exports_golden_refs -v`
Expected: FAIL with `ImportError: cannot import name 'curate_and_export_dataset'`

- [ ] **Step 3: Implement `TranscribedWord`, `CuratedClip`, `VoiceHarvestResult`, `select_golden_references`, `curate_and_export_dataset`, and `harvest_voice_from_youtube` (with `processed_video_ids` deduplication) in `core/voice_harvester.py`**

 Also add a test `test_discover_and_download_skips_already_processed_video_ids` in `tests/test_voice_harvester.py` verifying deduplication against `profile.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_voice_harvester.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/voice_harvester.py tests/test_voice_harvester.py
git commit -m "feat(harvester): add word-level Whisper confidence gate, Golden 10s selector, and F5/SoVITS exporter"
```

---

### Task 4: Integrate "🎙️ Voice Harvester Studio" into Gradio Web UI (`app.py` & `tests/test_app_smoke.py`)

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `harvest_voice_from_youtube`, `list_harvested_voices(voices_root: Path) -> list[dict]` from `core/voice_harvester.py`
- Produces:
  - `handle_harvest_voice(voice_id: str, query_or_url: str, target_minutes: float, max_videos: int) -> tuple[str, str | None, str, list[list[str]]]` in `app.py`
  - Auto-populates `F5-TTS` reference audio & transcript when user selects a harvested voice profile in `app.py`.

- [ ] **Step 1: Write the failing test in `tests/test_app_smoke.py` verifying `handle_harvest_voice` and voice profile loading into `F5-TTS`**

```python
def test_app_harvest_voice_and_apply_golden_ref(tmp_path: Path, monkeypatch):
    # Verify handle_harvest_voice returns summary, golden_ref_wav, golden_ref_text, and table rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_app_smoke.py -v`
Expected: FAIL with `AttributeError: module 'app' has no attribute 'handle_harvest_voice'`

- [ ] **Step 3: Implement `handle_harvest_voice`, `handle_select_harvested_voice`, and the "🎙️ Cào & Tuyển chọn Giọng Tự động (Auto Voice Harvester)" section in `app.py`**

- [ ] **Step 4: Run full test suite to verify 100% pass**

Run: `.venv/bin/pytest -v`
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "feat(ui): add Auto Voice Harvester Studio to Gradio Web UI"
```
