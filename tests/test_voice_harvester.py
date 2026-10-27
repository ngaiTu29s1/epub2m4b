from pathlib import Path

import json

import numpy as np
import soundfile as sf

from core.voice_harvester import (
    RawAudioCandidate,
    TranscribedWord,
    curate_and_export_dataset,
    discover_and_download_videos,
    filter_dominant_speaker,
    load_sliced_candidates,
    slice_and_filter_segments,
)


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


def test_slice_and_filter_keeps_hot_mastered_speech_but_drops_true_clipping(tmp_path: Path):
    """Hot-mastered audio (peak ~0 dBFS in a full mix) is usable, pinned audio is not.

    YouTube uploads are mastered to full scale, so a peak `>= -0.2 dBFS` is
    normal and only real clipping (energy pinned at full scale) is rejected.
    Without `inst_wav` there is no BGM reference, so the SNR gate is skipped and
    the clipping gate is what is under test here.
    """
    hot_path = _write_test_wav(tmp_path / 'hot.wav', [(6.0, 0.9999)])
    pinned_path = _write_test_wav(tmp_path / 'pinned.wav', [(6.0, 0.25)])
    pinned, sr = sf.read(str(pinned_path), dtype='float32')
    pinned = np.clip(pinned * 40.0, -1.0, 1.0).astype(np.float32)
    sf.write(str(pinned_path), pinned, sr, subtype='PCM_16')

    kept = slice_and_filter_segments(
        vocal_wav=hot_path,
        out_dir=tmp_path / 'hot_clips',
        min_sec=3.5,
        max_sec=12.0,
        pad_ms=200,
    )
    assert len(kept) == 1
    assert kept[0].peak_dbfs >= -0.2

    rejected = slice_and_filter_segments(
        vocal_wav=pinned_path,
        out_dir=tmp_path / 'pinned_clips',
        min_sec=3.5,
        max_sec=12.0,
        pad_ms=200,
    )
    assert rejected == []


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


def test_curate_keeps_real_clean_clip_with_one_low_confidence_word(tmp_path: Path):
    """Calibration: a real clean Vietnamese clip must survive the ASR gate.

    Measured with faster-whisper large-v3-turbo on real YouTube speech: mean word
    probability 0.948-0.951 with one word as low as 0.739 (1/20 below 0.92).
    A per-word minimum gate rejects every such clip (observed 0/85 kept).
    """
    candidate = RawAudioCandidate(
        path=_write_test_wav(tmp_path / 'real_clip.wav', [(5.0, 0.3)]),
        start_sec=0.0,
        end_sec=5.0,
        vocal_to_bgm_db=14.0,
        rms_dbfs=-20.0,
        peak_dbfs=-0.5,
    )
    measured = [1.0, 0.99, 0.98, 0.95, 0.99, 0.96, 0.94, 0.95, 0.97, 0.94,
                0.96, 0.95, 0.98, 0.97, 0.94, 0.96, 0.99, 0.95, 0.97, 0.739]

    def fake_transcribe(_path: Path):
        return [TranscribedWord(f'word{i}', p) for i, p in enumerate(measured)]

    res = curate_and_export_dataset(
        voice_id='vui_ve',
        candidates=[candidate],
        voices_root=tmp_path / 'voices',
        transcribe_fn=fake_transcribe,
    )
    assert len(res.kept_clips) == 1
    assert len(res.golden_refs) == 1


def test_load_sliced_candidates_round_trips_sidecar_metadata(tmp_path: Path):
    """A crash during transcription must not lose already-sliced audio."""
    vocal_path = _write_test_wav(tmp_path / 'vocal.wav', [(6.0, 0.3)])
    out_dir = tmp_path / 'clips'
    written = slice_and_filter_segments(vocal_path, out_dir, pad_ms=200)
    assert len(written) == 1

    reloaded = load_sliced_candidates(out_dir)
    assert len(reloaded) == 1
    assert reloaded[0].path == written[0].path
    assert reloaded[0].vocal_to_bgm_db == written[0].vocal_to_bgm_db
    assert round(reloaded[0].duration_sec, 3) == round(written[0].duration_sec, 3)


def test_discover_and_download_skips_already_processed_video_ids(tmp_path: Path):
    profile_dir = tmp_path / 'voices' / 'vui_ve'
    profile_dir.mkdir(parents=True)
    (profile_dir / 'profile.json').write_text(
        json.dumps({'voice_id': 'vui_ve', 'processed_video_ids': ['vid_old']}),
        encoding='utf-8',
    )

    calls: list[str] = []

    def fake_download(video_id: str, out_dir: Path) -> Path:
        calls.append(video_id)
        out = Path(out_dir) / f'{video_id}.wav'
        _write_test_wav(out, [(150.0, 0.3)])
        return out

    downloads = discover_and_download_videos(
        voice_id='vui_ve',
        query_or_url='https://www.youtube.com/watch?v=vid_old',
        voices_root=tmp_path / 'voices',
        video_ids=['vid_old', 'vid_new'],
        download_fn=fake_download,
    )
    assert calls == ['vid_new']
    assert [Path(d).stem for d in downloads] == ['vid_new']
