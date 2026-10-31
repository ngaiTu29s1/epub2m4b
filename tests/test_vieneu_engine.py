"""Unit tests for the VieNeu-TTS engine wrapper (no GPU / model download)."""
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from core import vieneu_engine
from core.vieneu_engine import (
    DEFAULT_PRESET_VOICE,
    PRESET_ALIASES,
    VieNeuEngine,
    _resolve_alias,
    build_voice_catalogue,
    _write_wav_resampled,
)


def _write_silence_wav(path: Path, seconds: float = 1.0, sr: int = 24000) -> None:
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b'\x00\x00' * int(sr * seconds))


def test_resolve_alias_maps_friendly_names():
    assert _resolve_alias('Tuyên') == 'Phạm Tuyên'
    assert _resolve_alias('Bình') == 'Thanh Bình'
    assert _resolve_alias('Sơn') == 'Quang Sơn'
    assert _resolve_alias('không-tồn-tại') == 'không-tồn-tại'


def test_engine_rejects_unknown_mode():
    with pytest.raises(ValueError):
        VieNeuEngine(mode='not-a-mode')


def test_engine_describe_for_each_mode(tmp_path: Path):
    preset = VieNeuEngine(mode='preset', voice='Bình')
    assert preset.describe() == 'preset:Thanh Bình'

    classic = VieNeuEngine(mode='classic')
    assert classic.describe() == 'classic:Nguyễn Ngọc Ngạn'

    ref = tmp_path / 'ref.wav'
    _write_silence_wav(ref)
    clone = VieNeuEngine(mode='clone', ref_audio=ref)
    assert clone.describe().startswith('clone:')


def test_engine_defaults_to_default_voice():
    engine = VieNeuEngine()
    assert engine.voice == DEFAULT_PRESET_VOICE


def test_write_wav_resampled_downmixes_to_project_rate(tmp_path: Path):
    sr_in = 48000
    samples = np.zeros(sr_in, dtype=np.float32)  # 1s of silence
    out = tmp_path / 'out.wav'
    dur_ms = _write_wav_resampled(samples, sr_in, out)
    assert out.exists()
    with wave.open(str(out), 'rb') as wf:
        assert wf.getframerate() == 24000
        assert wf.getnchannels() == 1
    assert 950 <= dur_ms <= 1050


def test_write_wav_resampled_empty_returns_silence(tmp_path: Path):
    out = tmp_path / 'empty.wav'
    dur_ms = _write_wav_resampled(np.array([], dtype=np.float32), 48000, out)
    assert dur_ms == 200
    assert out.exists()


def test_classic_reference_falls_back_to_vui_ve(tmp_path: Path, monkeypatch):
    """When no NNN WAV exists, classic mode falls back to the Vui Vẻ golden ref."""
    fallback = tmp_path / 'golden_01.wav'
    _write_silence_wav(fallback)

    monkeypatch.setattr(vieneu_engine, 'DEFAULT_CLASSIC_NGAN_REF_WAV', tmp_path / 'missing.wav')
    monkeypatch.setattr(vieneu_engine, 'VUI_VE_GOLDEN_REF', fallback)
    monkeypatch.setattr(vieneu_engine, 'VUI_VE_GOLDEN_TEXT', 'Transcript Vui Vẻ')

    engine = VieNeuEngine(mode='classic')
    ref_audio, ref_text = engine._reference_for_classic()
    assert ref_audio == fallback
    assert ref_text == 'Transcript Vui Vẻ'


def test_classic_reference_reads_sidecar_transcript(tmp_path: Path, monkeypatch):
    ref = tmp_path / 'ngan_01.wav'
    _write_silence_wav(ref)
    ref.with_suffix('.txt').write_text('Câu mẫu Nguyễn Ngọc Ngạn', encoding='utf-8')

    monkeypatch.setattr(vieneu_engine, 'DEFAULT_CLASSIC_NGAN_REF_WAV', ref)
    engine = VieNeuEngine(mode='classic')
    ref_audio, ref_text = engine._reference_for_classic()
    assert ref_audio == ref
    assert ref_text == 'Câu mẫu Nguyễn Ngọc Ngạn'


def test_classic_explicit_ref_text_wins_over_sidecar(tmp_path: Path, monkeypatch):
    ref = tmp_path / 'ngan_01.wav'
    _write_silence_wav(ref)
    ref.with_suffix('.txt').write_text('sidecar', encoding='utf-8')

    engine = VieNeuEngine(mode='classic', ref_audio=ref, ref_text='explicit')
    _, ref_text = engine._reference_for_classic()
    assert ref_text == 'explicit'


def test_catalogue_contains_classic_and_clone_entries(monkeypatch):
    def boom(*_a, **_kw):
        raise RuntimeError('no model in unit test')

    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', boom)
    labels = [o.label for o in build_voice_catalogue()]
    assert any('Nguyễn Ngọc Ngạn' in label for label in labels)
    assert any('Clone · WAV' in label for label in labels)
    assert any('Clone · Vui Vẻ' in label for label in labels)


def test_build_voice_catalogue_without_model_has_no_hardcoded_presets(monkeypatch):
    """Không còn danh sách giọng hardcode: offline → chỉ classic + clone."""
    def boom(*_a, **_kw):
        raise RuntimeError('no model')

    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', boom)
    options = build_voice_catalogue()
    assert all(o.mode != 'preset' for o in options)
    labels = [o.label for o in options]
    assert any('Nguyễn Ngọc Ngạn' in label for label in labels)
    assert any('Clone · Vui Vẻ' in label for label in labels)
    assert not hasattr(vieneu_engine, 'FALLBACK_PRESET_VOICES')


def test_list_available_preset_voices_empty_offline(monkeypatch):
    def boom(*_a, **_kw):
        raise RuntimeError('offline')

    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', boom)
    assert vieneu_engine.list_available_preset_voices() == []


def test_batch_size_default_is_adaptive(monkeypatch):
    """Mặc định lấy từ ``auto_detect_batch_size`` (VRAM/env), không hardcode."""
    monkeypatch.setattr(vieneu_engine, 'auto_detect_batch_size', lambda: 8)
    engine = VieNeuEngine(device='cuda')
    assert engine.batch_size == 8


def test_batch_size_explicit_overrides_default():
    engine = VieNeuEngine(device='cuda', batch_size=6)
    assert engine.batch_size == 6


def test_batch_size_never_below_one():
    engine = VieNeuEngine(device='cuda', batch_size=0)
    assert engine.batch_size == 1


def test_batch_size_forwarded_to_infer(monkeypatch, tmp_path: Path):
    """``batch_size`` phải được truyền vào ``tts.infer``."""
    captured: dict = {}

    class _FakeTTS:
        sample_rate = 24000

        def infer(self, text, **kwargs):
            captured.update(kwargs)
            return np.zeros(2400, dtype=np.float32)

    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', lambda *a, **k: _FakeTTS())
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=4)
    engine.synthesize_segment('Xin chào', tmp_path / 'out.wav')
    assert captured.get('batch_size') == 4

