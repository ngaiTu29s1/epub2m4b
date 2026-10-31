"""Tests for the Chatterbox / Viterbox TTS & voice-conversion engines.

Like ``tests/test_rvc_engine.py``, these exercise the *contract* of
``core.chatterbox_engine`` without downloading the multi-gigabyte Viterbox
checkpoints or loading torch models: the heavy model wiring is faked at the
module boundary (``get_viterbox`` / ``get_s3gen``) and the Edge-TTS call is
replaced with a fake engine.
"""
import wave
from pathlib import Path

import numpy as np
import pytest


class _FakeViterboxModel:
    """Stand-in for ``viterbox.Viterbox`` exposing only what the engine uses."""

    def __init__(self, sr: int = 24000, seconds: float = 0.5):
        self.sr = sr
        self.seconds = seconds
        self.calls: list[dict] = []

    def generate(self, text: str, **kwargs):
        import torch

        self.calls.append({'text': text, **kwargs})
        n = int(self.sr * self.seconds)
        return torch.zeros(1, n)


class _FakeS3Gen:
    """Stand-in for ``viterbox.models.s3gen.S3Gen`` voice-conversion decoder."""

    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.embedded = 0
        self.tokenized = 0
        self.inferred = 0

    def embed_ref(self, ref_wav, ref_sr, device='auto', **kwargs):
        self.embedded += 1
        return {'prompt_token': None, 'embedding': None}

    def tokenizer(self, wav):
        import torch

        self.tokenized += 1
        return torch.zeros(1, 25), torch.tensor([25])

    def inference(self, speech_tokens, ref_dict, **kwargs):
        import torch

        self.inferred += 1
        n = int(self.sr * 0.5)
        return torch.zeros(1, n), None


class _FakeEdgeEngine:
    """Stand-in for ``EdgeTTSEngine`` that writes a silent 24 kHz WAV."""

    def __init__(self, sr: int = 24000, seconds: float = 0.5):
        self.sr = sr
        self.seconds = seconds
        self.calls: list[str] = []

    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        self.calls.append(text)
        n = int(self.sr * self.seconds)
        with wave.open(str(out_wav), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(b'\x00\x00' * n)
        return int(self.seconds * 1000)


def _read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype='<i2'), sr, wf.getnchannels()


# --------------------------------------------------------------------------
# Contract / configuration
# --------------------------------------------------------------------------

def test_engine_module_defaults_point_at_vui_ve_assets():
    from core import chatterbox_engine as ce

    assert ce.DEFAULT_CKPT_DIR.name == 'chatterbox_vui_ve'
    assert ce.DEFAULT_GOLDEN_REF.name == 'golden_01.wav'
    assert 'vui_ve' in str(ce.DEFAULT_GOLDEN_REF)
    assert ce.SAMPLE_RATE == 24000


def test_engines_are_tts_engine_subclasses():
    from core.chatterbox_engine import ChatterboxVCEngine, ViterboxTTSEngine
    from core.tts_engine import TTSEngine

    assert issubclass(ViterboxTTSEngine, TTSEngine)
    assert issubclass(ChatterboxVCEngine, TTSEngine)


# --------------------------------------------------------------------------
# Silence handling (no model load)
# --------------------------------------------------------------------------

def test_viterbox_engine_nonword_segment_returns_silence(tmp_path: Path):
    from core.chatterbox_engine import ViterboxTTSEngine

    engine = ViterboxTTSEngine()
    out = tmp_path / 'silence.wav'
    dur = engine.synthesize_segment('...', out)
    assert dur == 200
    data, sr, ch = _read_wav(out)
    assert sr == 24000 and ch == 1
    assert len(data) == 24000 // 5  # 200 ms


def test_vc_engine_nonword_segment_returns_silence(tmp_path: Path):
    from core.chatterbox_engine import ChatterboxVCEngine

    engine = ChatterboxVCEngine()
    out = tmp_path / 'silence.wav'
    dur = engine.synthesize_segment('   ', out)
    assert dur == 200
    assert out.exists()


# --------------------------------------------------------------------------
# Viterbox TTS engine (model faked)
# --------------------------------------------------------------------------

def test_viterbox_engine_writes_24k_pcm16_mono(tmp_path: Path, monkeypatch):
    from core import chatterbox_engine as ce

    fake = _FakeViterboxModel(sr=24000, seconds=0.75)
    monkeypatch.setattr(ce, 'get_viterbox', lambda *a, **k: fake)

    engine = ce.ViterboxTTSEngine()
    out = tmp_path / 'seg.wav'
    dur = engine.synthesize_segment('Xin chào thế giới', out)

    assert dur == 750
    data, sr, ch = _read_wav(out)
    assert sr == 24000 and ch == 1
    assert len(data) == int(24000 * 0.75)
    assert fake.calls and fake.calls[0]['text'] == 'Xin chào thế giới'
    assert fake.calls[0]['language'] == 'vi'


def test_viterbox_engine_resamples_non_24k_output(tmp_path: Path, monkeypatch):
    from core import chatterbox_engine as ce

    # Model claims 16 kHz; engine must resample to 24 kHz on write.
    fake = _FakeViterboxModel(sr=16000, seconds=0.5)
    monkeypatch.setattr(ce, 'get_viterbox', lambda *a, **k: fake)

    engine = ce.ViterboxTTSEngine()
    out = tmp_path / 'seg.wav'
    engine.synthesize_segment('kiểm tra', out)
    _, sr, _ = _read_wav(out)
    assert sr == 24000


# --------------------------------------------------------------------------
# Chatterbox VC engine (Edge + S3Gen faked)
# --------------------------------------------------------------------------

def test_vc_engine_runs_edge_then_s3gen(tmp_path: Path, monkeypatch):
    from core import chatterbox_engine as ce

    s3gen = _FakeS3Gen()
    edge = _FakeEdgeEngine()
    monkeypatch.setattr(ce, 'get_s3gen', lambda *a, **k: s3gen)

    engine = ce.ChatterboxVCEngine(edge_engine=edge)
    out = tmp_path / 'vc.wav'
    dur = engine.synthesize_segment('Đây là câu thử nghiệm.', out)

    assert edge.calls == ['Đây là câu thử nghiệm.']
    assert s3gen.embedded == 1
    assert s3gen.tokenized == 1
    assert s3gen.inferred == 1
    assert dur == 500
    data, sr, ch = _read_wav(out)
    assert sr == 24000 and ch == 1
    assert len(data) == int(24000 * 0.5)


def test_vc_engine_caches_ref_dict(tmp_path: Path, monkeypatch):
    from core import chatterbox_engine as ce

    s3gen = _FakeS3Gen()
    edge = _FakeEdgeEngine()
    monkeypatch.setattr(ce, 'get_s3gen', lambda *a, **k: s3gen)

    engine = ce.ChatterboxVCEngine(edge_engine=edge)
    engine.synthesize_segment('câu một', tmp_path / 'a.wav')
    engine.synthesize_segment('câu hai', tmp_path / 'b.wav')

    # embed_ref must run only once across segments.
    assert s3gen.embedded == 1
    assert s3gen.inferred == 2
