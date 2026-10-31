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


def test_legacy_engines_are_removed():
    """F5-TTS / RVC / Viterbox must no longer exist after the VieNeu migration."""
    from core import tts_engine

    assert not hasattr(tts_engine, 'F5TTSEngine')
    assert not hasattr(tts_engine, 'RvcEngine')
    assert not hasattr(tts_engine, 'ViterboxTTSEngine')


def test_vieneu_engine_is_lazily_exported():
    """``from core.tts_engine import VieNeuEngine`` resolves via __getattr__."""
    from core.tts_engine import VieNeuEngine as EngineA
    from core.vieneu_engine import VieNeuEngine as EngineB

    assert EngineA is EngineB


