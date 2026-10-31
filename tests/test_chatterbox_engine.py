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
        self.prepare_calls: list[dict] = []

    def prepare_conditionals(self, audio_prompt, exaggeration=0.5, prompt_time_stretch=1.0):
        self.prepare_calls.append({
            'audio_prompt': audio_prompt,
            'exaggeration': exaggeration,
            'prompt_time_stretch': prompt_time_stretch,
        })

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
    # Text reaching the model is normalized (trailing full stop added) and the
    # conditionals are prepared once, not per call.
    assert fake.calls and fake.calls[0]['text'] == 'Xin chào thế giới.'
    assert fake.calls[0]['language'] == 'vi'
    assert fake.calls[0]['audio_prompt'] is None
    assert len(fake.prepare_calls) == 1
    assert fake.prepare_calls[0]['prompt_time_stretch'] == 0.82


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


# --------------------------------------------------------------------------
# Step 1 — Vietnamese text normalization (ALL-CAPS -> sentence case)
# --------------------------------------------------------------------------

def test_deuppercase_folds_all_caps_heading():
    from core.chatterbox_engine import deuppercase_vietnamese

    assert deuppercase_vietnamese('PHẦN DẪN') == 'Phần dẫn'
    assert deuppercase_vietnamese('CHƯƠNG 1') == 'Chương 1'
    assert deuppercase_vietnamese('GIỚI THIỆU') == 'Giới thiệu'


def test_deuppercase_preserves_single_letter_initials():
    from core.chatterbox_engine import deuppercase_vietnamese

    # One-letter initials are legitimate, do not fold them.
    assert deuppercase_vietnamese('A. T.') == 'A. T.'
    # Already-normal text is left as-is.
    assert deuppercase_vietnamese('Năm thứ tư') == 'Năm thứ tư'


def test_normalize_viterbox_text_adds_sentence_punctuation():
    from core.chatterbox_engine import normalize_viterbox_text

    assert normalize_viterbox_text('PHẦN DẪN') == 'Phần dẫn.'
    assert normalize_viterbox_text('Xin chào') == 'Xin chào.'
    # Existing punctuation is preserved.
    assert normalize_viterbox_text('Phần dẫn.') == 'Phần dẫn.'


def test_heading_tokenizes_to_lowercase_chars():
    """The root-cause check: after normalization the tokenizer must see the
    lowercase syllables, not the spell-out capital characters."""
    import sys

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / 'vendor' / 'viterbox_src'))
    from viterbox.models.tokenizers import MTLTokenizer

    from core.chatterbox_engine import plan_viterbox_segments

    tok = MTLTokenizer(str(repo / 'models' / 'chatterbox_vui_ve' / 'tokenizer_vi_expanded.json'))
    (clause, _pause), = plan_viterbox_segments('PHẦN DẪN')
    ids = tok.text_to_tokens(clause, language_id='vi').squeeze(0).tolist()
    # The lowercase syllable IDs (drops the leading [vi] + [SPACE] prefix).
    assert ids[2:] == [292, 21, 2460, 27, 2, 17, 2463, 27, 9]


# --------------------------------------------------------------------------
# Step 1 — Clause chunking for long sentences
# --------------------------------------------------------------------------

def test_chunk_long_sentence_splits_on_commas():
    from core.chatterbox_engine import chunk_into_clauses

    long = (
        'Trương Đông Thăng vốn là người đàn ông ở rể, trước khi cưới đã làm '
        'giấy chứng minh tài sản, nếu ly hôn thì ra đi tay trắng.'
    )
    clauses = chunk_into_clauses(long)
    assert len(clauses) >= 2
    assert all(c.endswith('.') for c in clauses)
    # Every clause stays within the friendly word budget.
    assert all(len(c.split()) <= 21 for c in clauses)


def test_chunk_short_sentence_is_untouched():
    from core.chatterbox_engine import chunk_into_clauses

    assert chunk_into_clauses('Phần dẫn.') == ['Phần dẫn.']


def test_plan_viterbox_segments_pauses():
    from core.chatterbox_engine import plan_viterbox_segments

    planned = plan_viterbox_segments('PHẦN DẪN')
    assert planned == [('Phần dẫn.', 500)]

    long = (
        'Trương Đông Thăng vốn là người đàn ông ở rể, trước khi cưới đã làm '
        'giấy chứng minh tài sản, nếu ly hôn thì ra đi tay trắng.'
    )
    planned = plan_viterbox_segments(long)
    # Comma-split clauses get the short breath, the sentence end gets 500 ms.
    assert planned[-1][1] == 500
    assert any(p == 250 for _c, p in planned)


# --------------------------------------------------------------------------
# Step 2 — Tuned defaults & conditional caching
# --------------------------------------------------------------------------

def test_viterbox_engine_tuned_defaults():
    from core.chatterbox_engine import ViterboxTTSEngine

    engine = ViterboxTTSEngine()
    assert engine.repetition_penalty == 1.25
    assert engine.cfg_weight == 0.35
    assert engine.exaggeration == 0.40
    assert engine.temperature == 0.65
    assert engine.top_p == 0.90
    assert engine.prompt_time_stretch == 0.82
    assert engine.cond_margin_s >= 0.12
    # A fixed seed keeps the fragile short-title tones reproducible.
    assert engine.seed == 3


def test_viterbox_engine_caches_conditionals(tmp_path: Path, monkeypatch):
    from core import chatterbox_engine as ce

    fake = _FakeViterboxModel(sr=24000, seconds=0.1)
    monkeypatch.setattr(ce, 'get_viterbox', lambda *a, **k: fake)

    engine = ce.ViterboxTTSEngine()
    for i in range(3):
        engine.synthesize_segment('Xin chào thế giới', tmp_path / f'{i}.wav')

    # prepare_conditionals runs exactly once for the whole chapter.
    assert len(fake.prepare_calls) == 1
    assert engine.conds_cached is True
    assert fake.calls[0]['split_sentences'] is False


def test_viterbox_engine_synthesizes_each_clause(tmp_path: Path, monkeypatch):
    from core import chatterbox_engine as ce

    fake = _FakeViterboxModel(sr=24000, seconds=0.1)
    monkeypatch.setattr(ce, 'get_viterbox', lambda *a, **k: fake)

    engine = ce.ViterboxTTSEngine()
    long = (
        'Trương Đông Thăng vốn là người đàn ông ở rể, trước khi cưới đã làm '
        'giấy chứng minh tài sản, nếu ly hôn thì ra đi tay trắng.'
    )
    engine.synthesize_segment(long, tmp_path / 'seg.wav')

    # One generate() call per clause.
    assert len(fake.calls) >= 2
    assert all(c['split_sentences'] is False for c in fake.calls)
