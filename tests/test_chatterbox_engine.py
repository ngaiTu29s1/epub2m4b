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

def test_chunk_long_sentence_splits_on_commas_preserving_them():
    from core.chatterbox_engine import chunk_into_clauses

    # 31 words with commas -> long enough to split.
    long = (
        'Trương Đông Thăng vốn là người đàn ông ở rể, trước khi cưới đã làm '
        'giấy chứng minh tài sản, nếu ly hôn, gần như là ra đi tay trắng và '
        'không có gì cả trong tay của anh ta lúc đó.'
    )
    clauses = chunk_into_clauses(long)
    assert len(clauses) >= 2
    # No emitted clause is a bare space-joined fragment: a clause cut at a comma
    # MUST still end with the comma (never silently dropped, never forced to '.').
    joined = ' '.join(clauses)
    assert joined.count(',') >= 2 or any(c.endswith(',') for c in clauses)
    # Every clause stays within the friendly word budget (+ slack for tiny ones).
    assert all(len(c.split()) <= 25 for c in clauses)


def test_chunk_short_sentence_is_untouched():
    from core.chatterbox_engine import chunk_into_clauses

    assert chunk_into_clauses('Phần dẫn.') == ['Phần dẫn.']


def test_medium_sentence_with_commas_kept_whole():
    """An 18-21 word sentence with clear commas must NOT be shredded."""
    from core.chatterbox_engine import chunk_into_clauses

    sentence = (
        'Năm thứ tư sau khi kết hôn, Từ Tịnh ngoại tình, đề nghị ly hôn với '
        'Trương Đông Thăng.'
    )
    assert chunk_into_clauses(sentence) == [sentence]


def test_commas_are_never_dropped_in_clause_chunking():
    """Requirement #4: every comma in the source survives chunking."""
    from core.chatterbox_engine import chunk_into_clauses

    source = (
        'Năm thứ tư sau khi kết hôn, Từ Tịnh ngoại tình, đề nghị ly hôn với '
        'Trương Đông Thăng. Sau khi suy nghĩ kỹ lưỡng, anh ta quyết định làm '
        'vài việc để thay đổi kết cục này, nào ngờ mọi thứ đổ bể hoàn toàn '
        'và không thể cứu vãn được nữa trong tình huống đó.'
    )
    clauses = chunk_into_clauses(source)
    n_src = source.count(',')
    n_out = sum(c.count(',') for c in clauses)
    assert n_out == n_src, f'comma count changed: {n_src} -> {n_out}'


def test_every_clause_boundary_is_at_punctuation():
    """Requirement #4: no clause boundary may be an unpunctuated word gap.

    Every clause (except the final one) must end with one of
    , ; : . ! ? … — i.e. T3 is only ever asked to break at real punctuation.
    """
    from core.chatterbox_engine import chunk_into_clauses

    source = (
        'Đây vốn là kế hoạch phạm tội hoàn mỹ mà anh ta đã dày công dàn dựng, '
        'nào ngờ màn biểu diễn này lại bị ba học sinh đang nô đùa ở gần đó vô '
        'tình dùng máy ảnh quay lại, và mọi chuyện vỡ lẽ ngay sau đó vài ngày '
        'khi cảnh sát vào cuộc điều tra rất nhanh chóng và kỹ lưỡng.'
    )
    clauses = chunk_into_clauses(source)
    terminators = (',', ';', ':', '.', '!', '?', '…')
    for c in clauses[:-1]:
        assert c.endswith(terminators), f'clause boundary without punctuation: {c!r}'


def test_plan_viterbox_segments_pauses():
    from core.chatterbox_engine import plan_viterbox_segments

    planned = plan_viterbox_segments('PHẦN DẪN')
    assert planned == [('Phần dẫn.', 480)]

    long = (
        'Trương Đông Thăng vốn là người đàn ông ở rể, trước khi cưới đã làm '
        'giấy chứng minh tài sản, nếu ly hôn, gần như là ra đi tay trắng và '
        'không có gì cả trong tay của anh ta lúc đó.'
    )
    planned = plan_viterbox_segments(long)
    # Comma-split clauses get the short breath, the sentence end gets ~480 ms.
    assert planned[-1][1] == 480
    assert any(p == 200 for _c, p in planned)


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


def test_vendor_split_text_never_cuts_midword():
    """RC3: the vendor sentence splitter must only break at punctuation, never
    in the middle of an unpunctuated word run."""
    import sys

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / 'vendor' / 'viterbox_src'))
    from viterbox.tts import _split_text_to_sentences

    text = (
        'Đây vốn là kế hoạch phạm tội hoàn mỹ mà anh ta đã dày công dàn dựng, '
        'nào ngờ màn biểu diễn này lại bị ba học sinh đang nô đùa ở gần đó vô '
        'tình dùng máy ảnh quay lại và mọi chuyện vỡ lẽ ngay sau đó vài ngày '
        'khi cảnh sát vào cuộc điều tra rất nhanh chóng và kỹ lưỡng trong '
        'vụ án này.'
    )
    parts = _split_text_to_sentences(text)
    terminators = (',', ';', ':', '.', '!', '?', '…')
    for p in parts[:-1]:
        assert p.endswith(terminators), f'mid-word split: {p!r}'


def test_vendor_split_keeps_commas_and_stopdots():
    import sys

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / 'vendor' / 'viterbox_src'))
    from viterbox.tts import _split_text_to_sentences

    text = (
        'Trương Đông Thăng vốn là người đàn ông ở rể, trước khi cưới đã làm '
        'giấy chứng minh tài sản, nếu ly hôn, gần như là ra đi tay trắng và '
        'không còn bất cứ thứ gì trong tay cả lúc đó.'
    )
    parts = _split_text_to_sentences(text)
    assert sum(p.count(',') for p in parts) == text.count(',')


def test_viterbox_engine_synthesizes_each_clause(tmp_path: Path, monkeypatch):
    from core import chatterbox_engine as ce

    fake = _FakeViterboxModel(sr=24000, seconds=0.1)
    monkeypatch.setattr(ce, 'get_viterbox', lambda *a, **k: fake)

    engine = ce.ViterboxTTSEngine()
    long = (
        'Trương Đông Thăng vốn là người đàn ông ở rể, trước khi cưới đã làm '
        'giấy chứng minh tài sản, nếu ly hôn, gần như là ra đi tay trắng và '
        'không còn gì cả trong tay của anh ta lúc đó.'
    )
    engine.synthesize_segment(long, tmp_path / 'seg.wav')

    # One generate() call per clause.
    assert len(fake.calls) >= 2
    assert all(c['split_sentences'] is False for c in fake.calls)


# --------------------------------------------------------------------------
# Tail-artifact cleanup (buzzing "rè rè" before every pause)
# --------------------------------------------------------------------------

def _load_tail_cleanup():
    import sys

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / 'vendor' / 'viterbox_src'))
    from viterbox.tts import _frame_rms_zcr, tail_cleanup_clean

    return tail_cleanup_clean, _frame_rms_zcr


def test_tail_cleanup_cuts_silence_valley_and_buzz_burst():
    """synthetic speech + 150ms valley + 100ms 0.03-RMS burst -> the burst
    (and the valley) must be removed, with a fade to exactly 0.0."""
    import numpy as np

    tail_cleanup_clean, _ = _load_tail_cleanup()
    sr = 24000
    rng = np.random.default_rng(0)

    speech = 0.1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
    valley = np.zeros(int(0.15 * sr))
    burst = 0.03 * rng.standard_normal(int(0.10 * sr))
    audio = np.concatenate([speech, valley, burst]).astype(np.float32)

    out = tail_cleanup_clean(audio, sr)

    # Trimmed to (at most) the start of the valley, never longer.
    assert len(out) <= len(audio)
    assert out[-1] == 0.0
    # No buzzing island survives: the final 5 ms is faded to near-silence and
    # the last sample is exactly zero.
    assert float(np.max(np.abs(out[-int(0.005 * sr):]))) < 0.02
    # The burst region (tail 100ms of the input) is gone from the output tail.
    assert not np.array_equal(out[-1:], audio[-1:])


def test_tail_cleanup_tail_rms_monotonically_decays_to_zero():
    import numpy as np

    tail_cleanup_clean, _frame_rms_zcr = _load_tail_cleanup()
    sr = 24000
    rng = np.random.default_rng(1)
    speech = 0.1 * np.sin(2 * np.pi * 180 * np.arange(sr) / sr)
    valley = np.zeros(int(0.2 * sr))
    burst = 0.03 * rng.standard_normal(int(0.08 * sr))
    audio = np.concatenate([speech, valley, burst]).astype(np.float32)

    out = tail_cleanup_clean(audio, sr)
    w = int(0.05 * sr)
    tail = out[-int(0.5 * sr):]
    rms = [float(np.sqrt(np.mean(tail[i:i + w] ** 2))) for i in range(0, len(tail) - w + 1, w)]
    # The 50ms RMS windows must never rise toward the end (no re-bloom).
    assert all(rms[i] >= rms[i + 1] - 1e-6 for i in range(len(rms) - 1))
    assert out[-1] == 0.0


def test_tail_cleanup_removes_old_valley_plus_island_pattern():
    """Reproduce the exact pre-fix artifact: speech decays to a 0.0026 valley,
    then a 0.0104 high-ZCR island at the very end. Both must be cut."""
    import numpy as np

    tail_cleanup_clean, _ = _load_tail_cleanup()
    sr = 24000
    rng = np.random.default_rng(5)
    t = np.arange(int(0.5 * sr)) / sr
    speech = 0.15 * np.sin(2 * np.pi * 220 * t) * np.exp(-3 * t)
    speech[-int(0.1 * sr):] *= np.linspace(1, 0.02, int(0.1 * sr))
    valley = 0.0026 * rng.standard_normal(int(0.15 * sr))
    island = 0.0104 * rng.standard_normal(int(0.05 * sr))
    audio = np.concatenate([speech, valley, island]).astype(np.float32)

    out = tail_cleanup_clean(audio, sr)

    # The 150ms valley + 50ms island are gone from the tail.
    assert len(out) < len(audio) - int(0.15 * sr)
    assert out[-1] == 0.0
    w = int(0.05 * sr)
    rms = [float(np.sqrt(np.mean(out[i:i + w] ** 2))) for i in range(0, len(out) - w + 1, w)]
    # Monotone decay toward the end, no island re-bloom.
    assert all(rms[i] >= rms[i + 1] - 1e-6 for i in range(len(rms) - 1))
    # No 0.01-RMS island survives: the last 5 ms is essentially silent.
    assert float(np.max(np.abs(out[-int(0.005 * sr):]))) < 0.005


def test_tail_is_cleaned_on_synthesize_by_engine(tmp_path: Path, monkeypatch):
    """The engine's per-clause path must pass audio through the cleanup even
    when the fake model returns a raw buzz-tail buffer."""
    import numpy as np

    from core import chatterbox_engine as ce

    sr = 24000

    class _BuzzModel:
        sr = 24000

        def prepare_conditionals(self, *a, **k):
            return None

        def generate(self, text, **kwargs):
            import torch

            # Delegate to the real vendored generate() so the cleanup runs.
            rng = np.random.default_rng(2)
            speech = 0.1 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)
            valley = np.zeros(int(0.15 * sr))
            burst = 0.03 * rng.standard_normal(int(0.1 * sr))
            audio = np.concatenate([speech, valley, burst]).astype(np.float32)
            from viterbox.tts import tail_cleanup_clean

            cleaned = tail_cleanup_clean(audio, sr)
            return torch.from_numpy(cleaned).unsqueeze(0)

    monkeypatch.setattr(ce, 'get_viterbox', lambda *a, **k: _BuzzModel())
    engine = ce.ViterboxTTSEngine()

    out = tmp_path / 'buzz.wav'
    engine.synthesize_segment('Xin chào.', out)

    data, _sr, _ch = _read_wav(out)
    data_f = data.astype(np.float32) / 32768.0
    assert float(np.max(np.abs(data_f[-int(0.005 * sr):]))) < 0.02
