"""Tests for the Python 3.12-compatible RVC v2 voice conversion engine.

These exercise the *contract* of ``core.rvc_engine`` without downloading the
multi-hundred-megabyte pretrained RVC checkpoints: the heavy model wiring is
faked at the module boundary, exactly like the F5-TTS tests do in
``tests/test_tts_engine.py``.
"""
import wave
from pathlib import Path

import numpy as np
import pytest


def _write_wav(path: Path, seconds: float = 0.5, sr: int = 24000) -> None:
    n = int(sr * seconds)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b'\x00\x00' * n)


def test_rvc_engine_constants_point_at_v2_40k_assets():
    """The engine must target the RVC v2 40k generator + ContentVec 768-dim."""
    from core import rvc_engine

    assert rvc_engine.RVC_SAMPLE_RATE == 40000
    assert rvc_engine.RVC_VERSION == 'v2'
    assert rvc_engine.CONTENTVEC_REPO == 'lengyue233/content-vec-best'
    assert rvc_engine.HUBERT_FEATURE_DIM == 768
    assert 'f0G40k.pth' in rvc_engine.RVC_GENERATOR_FILE


def test_rvc_converter_resamples_to_40k_and_returns_wav(tmp_path: Path, monkeypatch):
    """`convert_file` reads Edge-TTS 24k audio, runs the net, writes 40k PCM16."""
    from core import rvc_engine

    in_wav = tmp_path / 'in.wav'
    out_wav = tmp_path / 'out.wav'
    _write_wav(in_wav, seconds=0.5, sr=24000)

    converter = rvc_engine.RVCVoiceConverter.__new__(rvc_engine.RVCVoiceConverter)
    converter.model_path = tmp_path / 'fake.pth'
    converter.index_path = None
    converter.device = 'cpu'
    converter.index_rate = 0.5
    converter.index = None
    converter.f0_method = 'rmvpe'
    converter.speaker_id = 0

    # Fake the heavy stages so we exercise only the I/O + resample wiring.
    monkeypatch.setattr(
        converter, '_extract_features',
        lambda audio, sr: np.zeros((1, rvc_engine.HUBERT_FEATURE_DIM, 10), dtype=np.float32),
    )
    monkeypatch.setattr(
        converter, '_extract_f0',
        lambda audio, sr: (np.full(10, 200, dtype=np.int64),
                           np.full(10, 200.0, dtype=np.float32)),
    )
    monkeypatch.setattr(
        converter, '_run_net',
        lambda feats, pitch, pitchf, sid: np.sin(
            np.arange(rvc_engine.RVC_SAMPLE_RATE // 2) * 0.05
        ).astype(np.float32) * 0.1,
    )

    converter.convert_file(in_wav, out_wav)

    assert out_wav.exists()
    with wave.open(str(out_wav), 'rb') as wf:
        assert wf.getframerate() == rvc_engine.RVC_SAMPLE_RATE
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_rvc_converter_resample_preserves_duration(tmp_path: Path, monkeypatch):
    """A 0.5s @24k input stays ~0.5s of audio after the 40k conversion path."""
    from core import rvc_engine

    in_wav = tmp_path / 'in.wav'
    out_wav = tmp_path / 'out.wav'
    _write_wav(in_wav, seconds=0.5, sr=24000)

    converter = rvc_engine.RVCVoiceConverter.__new__(rvc_engine.RVCVoiceConverter)
    converter.model_path = tmp_path / 'fake.pth'
    converter.index_path = None
    converter.device = 'cpu'
    converter.index_rate = 0.5
    converter.index = None
    converter.f0_method = 'rmvpe'
    converter.speaker_id = 0

    seen = {}

    def fake_run_net(feats, pitch, pitchf, sid):
        # Emit exactly as many samples as the 0.5 s input implies.
        seen['f0_len'] = len(pitchf)
        return np.zeros(rvc_engine.RVC_SAMPLE_RATE // 2, dtype=np.float32)

    monkeypatch.setattr(converter, '_extract_features', lambda a, s: np.zeros((1, 768, 10), np.float32))
    monkeypatch.setattr(
        converter, '_extract_f0',
        lambda a, s: (np.full(50, 200, np.int64), np.full(50, 200.0, np.float32)),
    )
    monkeypatch.setattr(converter, '_run_net', fake_run_net)

    duration = converter.convert_file(in_wav, out_wav)
    assert seen['f0_len'] == 50
    assert abs(duration - 500) < 60


def test_reference_index_paths_are_discoverable(tmp_path: Path):
    """The dataset index/model paths follow the conventional layout."""
    from core import rvc_engine

    assert rvc_engine.DEFAULT_MODEL_DIR_NAME == 'models/rvc_vui_ve'
    out = rvc_engine.expected_artifacts(Path('models/rvc_vui_ve'), 'vui_ve')
    assert out['model'].name == 'vui_ve.pth'
    assert out['index'].name == 'vui_ve.index'


def test_rvc_engine_is_importable_without_loading_models():
    """Importing the module must not eagerly pull torch/cuda or hit the Hub."""
    import importlib

    import core.rvc_engine as m

    importlib.reload(m)
    # No model is loaded merely by importing.
    assert m.RVCVoiceConverter is not None


def test_feature_index_builder_writes_a_faiss_index(tmp_path: Path, monkeypatch):
    """`build_feature_index` consumes a folder of wavs and writes a .index."""
    import faiss

    from core import rvc_engine

    wav_dir = tmp_path / 'wavs'
    wav_dir.mkdir()
    for i in range(3):
        _write_wav(wav_dir / f'clip_{i}.wav', seconds=0.4, sr=24000)
    out_index = tmp_path / 'vui_ve.index'

    converter = rvc_engine.RVCVoiceConverter.__new__(rvc_engine.RVCVoiceConverter)
    converter.device = 'cpu'
    converter.index = None
    converter.index_rate = 0.0

    def fake_extract(self, audio, sr):
        return np.random.rand(1, rvc_engine.HUBERT_FEATURE_DIM, 7).astype(np.float32)

    monkeypatch.setattr(rvc_engine.RVCVoiceConverter, '_extract_features',
                        fake_extract)
    monkeypatch.setattr(rvc_engine.RVCVoiceConverter, '_load_index',
                        lambda self: None)

    n = rvc_engine.build_feature_index(converter, wav_dir, out_index)

    assert out_index.exists()
    assert n > 0
    idx = faiss.read_index(str(out_index))
    assert idx.ntotal == n
    assert idx.d == rvc_engine.HUBERT_FEATURE_DIM

