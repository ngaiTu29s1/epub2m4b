"""Tests cho chapter-level batching & in-memory resampling.

Spec: docs/superpowers/specs/2026-10-31-chapter-batching-inmemory-resample-design.md
"""
import wave
from pathlib import Path

import numpy as np
import pytest

from core import vieneu_engine
from core.models import Chapter
from core.tts_engine import TTSEngine, render_chapter
from core.vieneu_engine import (
    DEFAULT_SAMPLING,
    SAMPLE_RATE,
    VieNeuEngine,
    _concat_waveforms,
    _resample_waveform_in_memory,
    _write_wav_resampled,
)


# ---------------------------------------------------------------------------
# 2.1 In-memory resampling
# ---------------------------------------------------------------------------
def test_resample_same_rate_returns_input():
    x = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)
    out = _resample_waveform_in_memory(x, 24000, 24000)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x)


def test_resample_48k_to_24k_halves_length():
    x = np.zeros(48000, dtype=np.float32)
    out = _resample_waveform_in_memory(x, 48000, 24000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(len(out) - 24000) <= 2


def test_resample_preserves_sine_frequency():
    sr, freq = 48000, 440.0
    t = np.arange(sr, dtype=np.float32) / sr
    sine = np.sin(2 * np.pi * freq * t).astype(np.float32)
    out = _resample_waveform_in_memory(sine, sr, 24000)
    # Đếm zero-crossing trong 1 giây → ~2*freq.
    crossings = np.sum(np.diff(np.signbit(out)) != 0)
    assert abs(crossings - 2 * freq) < 40


def test_resample_flattens_stereo():
    x = np.zeros((2, 1000), dtype=np.float32)
    out = _resample_waveform_in_memory(x, 48000, 24000)
    assert out.ndim == 1


def test_resample_does_not_invoke_ffmpeg(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError('không được gọi ffmpeg/subprocess để resample trong RAM')

    monkeypatch.setattr(vieneu_engine.subprocess, 'run', boom)
    monkeypatch.setattr(vieneu_engine, '_get_ffmpeg_bin', boom)
    out_wav = Path('/tmp/opencode/resample_test.wav')
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _write_wav_resampled(np.zeros(48000, dtype=np.float32), 48000, out_wav)
    with wave.open(str(out_wav), 'rb') as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnchannels() == 1


def test_write_wav_resampled_empty_is_silence(tmp_path: Path):
    out = tmp_path / 'empty.wav'
    dur = _write_wav_resampled(np.zeros(0, dtype=np.float32), 48000, out)
    assert dur == 200
    assert out.exists()


def test_concat_waveforms_handles_empties():
    a = np.ones(10, dtype=np.float32)
    assert len(_concat_waveforms([a, None, a])) == 20
    assert len(_concat_waveforms([])) == 0


# ---------------------------------------------------------------------------
# 2.2 synthesize_segments_batch
# ---------------------------------------------------------------------------
class _FakeVieNeuTTS:
    """Model giả: ``_infer_chunks`` trả 48 kHz với độ dài tỉ lệ độ dài phoneme."""

    sample_rate = 48000
    backend = 'pytorch'

    def __init__(self):
        self.calls: list[dict] = []

    def _resolve_ref(self, voice, ref_audio, denoise, use_ref_codes):
        return np.zeros(4, dtype=np.float32), None

    def _infer_chunks(self, phonemes, speaker_emb, ref_codes, use_ref_codes,
                      batch_size, sampling):
        self.calls.append({
            'phonemes': list(phonemes),
            'batch_size': batch_size,
            'sampling': dict(sampling),
            'use_ref_codes': use_ref_codes,
        })
        return [np.zeros(48000 // 2, dtype=np.float32) for _ in phonemes]


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeVieNeuTTS()
    monkeypatch.setattr(vieneu_engine, 'load_vieneu_model', lambda *a, **k: model)
    return model


def test_batch_returns_one_result_per_text(fake_model):
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=4)
    texts = ['Câu một.', 'Câu hai.', 'Câu ba.']
    results = engine.synthesize_segments_batch(texts)
    assert len(results) == len(texts)
    for wav, dur_ms in results:
        assert isinstance(wav, np.ndarray)
        assert wav.ndim == 1
        assert len(wav) == 24000 // 2  # 48 kHz 0.5s → 24 kHz 0.5s
        assert dur_ms == 500


def test_batch_duration_matches_waveform_length(fake_model):
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=4)
    for wav, dur_ms in engine.synthesize_segments_batch(['Câu một.', 'Câu hai.']):
        assert dur_ms == int(round(len(wav) / SAMPLE_RATE * 1000))


def test_batch_issues_single_infer_chunks_call(fake_model):
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=4)
    engine.synthesize_segments_batch(['Câu một.', 'Câu hai.', 'Câu ba.'])
    assert len(fake_model.calls) == 1


def test_batch_forwards_batch_size_and_full_sampling(fake_model):
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=6)
    engine.synthesize_segments_batch(['Câu một.'])
    call = fake_model.calls[0]
    assert call['batch_size'] == 6
    # sampling={} sẽ làm `_cap_frames` KeyError → phải có max_new_frames.
    assert call['sampling']['max_new_frames'] == DEFAULT_SAMPLING['max_new_frames']
    assert call['sampling']['temperature'] == DEFAULT_SAMPLING['temperature']
    assert call['sampling'] is not DEFAULT_SAMPLING  # không mutate hằng số
    assert call['use_ref_codes'] is True


def test_batch_preserves_order_and_concatenates_chunks(fake_model, monkeypatch):
    # Câu đầu 2 chunk, câu hai 1 chunk → phải nối đúng thứ tự, không lẫn.
    monkeypatch.setattr(
        vieneu_engine, '_upstream_chunker',
        lambda: (lambda text, max_chars=256: (
            (['a', 'b'], [0.1, 0.1]) if text == 'Dài.' else (['c'], [0.1])
        )),
    )
    monkeypatch.setattr(vieneu_engine, '_upstream_phonemize', lambda: (lambda c: c))
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=4)
    results = engine.synthesize_segments_batch(['Dài.', 'Ngắn.'])
    assert len(results) == 2
    # 2 chunk (mỗi chunk 0.5s) + silence nội bộ → dài hơn hẳn câu 1 chunk.
    assert results[0][1] > results[1][1]


def test_batch_empty_input_returns_empty(fake_model):
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=4)
    assert engine.synthesize_segments_batch([]) == []
    assert fake_model.calls == []


def test_batch_blank_text_yields_silence_not_crash(fake_model, monkeypatch):
    # Chunker thật bỏ qua chuỗi toàn khoảng trắng; giả lập đúng hành vi đó.
    monkeypatch.setattr(
        vieneu_engine, '_upstream_chunker',
        lambda: (lambda text, max_chars=256: ([], []) if text.strip() == 'blank' else (['x'], [])),
    )
    monkeypatch.setattr(vieneu_engine, '_upstream_phonemize', lambda: (lambda c: c))
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=4)
    results = engine.synthesize_segments_batch(['blank', 'có chữ'])
    assert len(results) == 2
    blank_wav, blank_ms = results[0]
    assert blank_ms == 200
    assert len(blank_wav) == int(SAMPLE_RATE * 0.2)
    assert results[1][1] == 500  # câu có chữ: 0.5 s từ model giả


def test_resolve_reference_single_call_for_batch(fake_model):
    calls = []
    original = fake_model._resolve_ref

    def counting(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)

    fake_model._resolve_ref = counting
    engine = VieNeuEngine(mode='preset', voice='Bình', device='cuda', batch_size=4)
    engine.synthesize_segments_batch(['Câu một.'])
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 2.3 render_chapter batching
# ---------------------------------------------------------------------------
class BatchTTSEngine(TTSEngine):
    """Engine giả hỗ trợ batch: mỗi câu 0.5 s im lặng."""

    def __init__(self, dur_ms: int = 500):
        self.dur_ms = dur_ms
        self.batch_calls: list[list[str]] = []
        self.segment_calls = 0

    def synthesize_segments_batch(self, texts: list[str]):
        self.batch_calls.append(list(texts))
        n = int(SAMPLE_RATE * self.dur_ms / 1000)
        return [(np.zeros(n, dtype=np.float32), self.dur_ms) for _ in texts]

    def synthesize_segment(self, text: str, out_wav: Path) -> int:  # pragma: no cover
        self.segment_calls += 1
        raise AssertionError('không được gọi tuần tự khi engine hỗ trợ batch')


def test_render_chapter_uses_batch_once(tmp_path: Path):
    chap = Chapter(index=2, title='Chương 2', markdown='# Chương 2\n\nCâu một.\n\nCâu hai.\n\nCâu ba.')
    engine = BatchTTSEngine()
    res = render_chapter(chap, engine, tmp_path)
    assert len(engine.batch_calls) == 1
    assert engine.segment_calls == 0
    assert len(engine.batch_calls[0]) == len(res.cues)
    with wave.open(str(res.audio_path), 'rb') as wf:
        assert wf.getframerate() == SAMPLE_RATE
        assert wf.getnchannels() == 1


def test_render_chapter_batch_cues_are_contiguous(tmp_path: Path):
    from core.tts_engine import markdown_to_segments

    md = '# Chương 3\n\nCâu một.\n\nCâu hai.\n\nCâu ba.'
    chap = Chapter(index=3, title='Chương 3', markdown=md)
    engine = BatchTTSEngine(dur_ms=400)
    res = render_chapter(chap, engine, tmp_path)
    specs = markdown_to_segments(md)
    # Cue khớp từng phân đoạn: start = tổng (audio + pause) của các đoạn trước.
    expected_start = 0
    for cue, spec in zip(res.cues, specs):
        assert cue.start_ms == expected_start
        assert cue.end_ms == expected_start + 400
        assert cue.text == spec.text
        expected_start += 400 + spec.pause_after_ms
    assert res.cues[0].start_ms == 0
    assert res.cues[-1].end_ms <= res.duration_ms
    assert res.duration_ms == expected_start


def test_render_chapter_cue_durations_match_audio(tmp_path: Path):
    chap = Chapter(index=4, title='Chương 4', markdown='# Chương 4\n\nMột câu duy nhất thôi.')
    engine = BatchTTSEngine(dur_ms=1234)
    res = render_chapter(chap, engine, tmp_path)
    total = sum(c.end_ms - c.start_ms for c in res.cues)
    assert total == 1234 * len(res.cues)
    assert res.duration_ms >= total


def test_render_chapter_falls_back_without_batch_support(tmp_path: Path):
    from tests.test_tts_engine import DummyTTSEngine

    chap = Chapter(index=5, title='Chương 5', markdown='# Chương 5\n\nCâu đầu tiên.\n\nCâu thứ hai.')
    engine = DummyTTSEngine()
    res = render_chapter(chap, engine, tmp_path)
    assert engine.calls == len(res.cues)
    assert all(c.end_ms > c.start_ms for c in res.cues)


def test_render_chapter_batch_mismatch_raises(tmp_path: Path):
    class BadBatch(BatchTTSEngine):
        def synthesize_segments_batch(self, texts):
            return []  # cố tình sai số lượng

    chap = Chapter(index=6, title='Chương 6', markdown='Chương 6\n\nMột câu.')
    with pytest.raises(RuntimeError, match='synthesize_segments_batch'):
        render_chapter(chap, BadBatch(), tmp_path)


def test_render_chapter_batch_pause_is_written(tmp_path: Path):
    chap = Chapter(
        index=7,
        title='Chương 7',
        markdown='Chương 7\n\nCâu một.\n\n\n\nCâu hai.',
    )
    engine = BatchTTSEngine(dur_ms=300)
    res = render_chapter(chap, engine, tmp_path)
    gaps = [nxt.start_ms - prev.end_ms for prev, nxt in zip(res.cues, res.cues[1:])]
    assert any(g > 0 for g in gaps), f'kỳ vọng có khoảng lặng, nhận {gaps}'
