import wave
from pathlib import Path
from core.models import Chapter
from core.tts_engine import TTSEngine, render_chapter


class DummyTTSEngine(TTSEngine):
    def __init__(self):
        self.calls = 0

    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        self.calls += 1
        sr = 24000
        n_frames = sr // 2  # 500ms
        with wave.open(str(out_wav), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b'\x00\x00' * n_frames)
        return 500


def test_render_chapter_stitches_pauses_and_caches_checkpoint(tmp_path: Path):
    chap = Chapter(index=1, title='Chương 1', markdown='# Chương 1\n\nCâu đầu tiên.\n\nCâu thứ hai.')
    engine = DummyTTSEngine()
    res1 = render_chapter(chap, engine, tmp_path)
    assert res1.audio_path.exists()
    assert res1.duration_ms >= 3500
    assert len(res1.cues) == 3
    calls_after_first = engine.calls

    res2 = render_chapter(chap, engine, tmp_path, force=False)
    assert engine.calls == calls_after_first
    assert res2.duration_ms == res1.duration_ms


def test_edge_tts_non_word_segment_returns_silence(tmp_path: Path):
    from core.tts_engine import EdgeTTSEngine
    engine = EdgeTTSEngine()
    out_wav = tmp_path / 'silence.wav'
    dur = engine.synthesize_segment('...', out_wav)
    assert dur == 200
    assert out_wav.exists()


def test_f5_vietnamese_checkpoint_constants_are_configured():
    """The F5 engine must target a Vietnamese checkpoint with its own vocab.txt.

    The base SWivid/F5-TTS model is EN-ZH: its vocab has no Vietnamese
    syllables, so it produces garbled/lisping speech for Vietnamese text. The
    Vietnamese fine-tune (toandev/F5-TTS-Vietnamese) ships a char-level vocab
    covering all 47 Vietnamese diacritics.
    """
    from core import tts_engine

    assert 'Vietnamese' in tts_engine.F5_VIETNAMESE_REPO
    assert tts_engine.F5_VIETNAMESE_CKPT_FILE.endswith('.safetensors')
    assert tts_engine.F5_VIETNAMESE_VOCAB_FILE == 'vocab.txt'


def test_f5_resources_loads_vietnamese_vocab(tmp_path: Path, monkeypatch):
    """`load_model` must be given the Vietnamese vocab.txt, not the EN-ZH default."""
    from core import tts_engine

    captured = {}

    def fake_load_model(model_cls, model_cfg, ckpt_path, **kwargs):
        captured['model_cfg'] = model_cfg
        captured['ckpt_path'] = ckpt_path
        captured.update(kwargs)
        return object()

    # Fake the heavy deps so we exercise only our wiring.
    monkeypatch.setattr(tts_engine, '_load_f5_vocab', lambda: tmp_path / 'vocab.txt')
    (tmp_path / 'vocab.txt').write_text(' a\nă\n', encoding='utf-8')
    monkeypatch.setattr(tts_engine, '_resolve_f5_ckpt', lambda: tmp_path / 'model.safetensors')
    monkeypatch.setattr(
        tts_engine, '_import_f5_infer_deps',
        lambda: (fake_load_model, lambda is_local: object()),
    )

    tts_engine._F5_CACHE.clear()
    model, vocoder, device = tts_engine._get_f5_resources()
    assert model is not None
    assert captured['vocab_file'] == str(tmp_path / 'vocab.txt')
    assert captured['ckpt_path'] == str(tmp_path / 'model.safetensors')


