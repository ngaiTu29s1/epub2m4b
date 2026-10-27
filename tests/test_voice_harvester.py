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
