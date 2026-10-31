"""Voice showcase generator (spec 2026-10-31 §2.3) — không render model thật."""
import json
import wave
from pathlib import Path

import pytest

from scripts import generate_voice_showcase as showcase


def _write_wav(path: Path, seconds: float = 0.5, sr: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b'\x00\x00' * int(sr * seconds))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    out = tmp_path / 'golden_samples'
    monkeypatch.setattr(showcase, 'OUT_DIR', out)
    monkeypatch.setattr(showcase, 'INDEX_PATH', out / 'index.json')
    monkeypatch.setattr(showcase, '_REPO_ROOT', tmp_path)
    return out


def test_slug_is_ascii_and_safe():
    assert showcase._slug('Phạm Tuyên') == 'Pham_Tuyen'
    assert showcase._slug('Thiền Tâm Đức') == 'Thien_Tam_Duc'
    assert showcase._slug('Adam bựa') == 'Adam_bua'
    assert showcase._slug('!!!') == 'voice'
    assert showcase._voice_id('Ngọc Trân') == 'Ngoc_Tran'


def test_golden_script_reuses_benchmark():
    from scripts.benchmark_stress_test import _build_raw_script

    assert showcase._golden_script() == _build_raw_script()
    normalized = showcase._normalized_script()
    assert len(normalized) > 500
    # Đã qua normalizer: không còn ký hiệu thô khó đọc.
    assert 'km/h' not in normalized


def test_generate_writes_wav_and_index(isolated, monkeypatch):
    calls: list[str] = []

    class FakeEngine:
        def __init__(self, mode, voice):
            self.voice = voice

        def synthesize_segment(self, text, out_wav):
            calls.append(self.voice)
            _write_wav(Path(out_wav), seconds=1.0)
            return 1000

    monkeypatch.setattr('core.vieneu_engine.VieNeuEngine', FakeEngine)
    monkeypatch.setattr(showcase, '_wav_to_opus', lambda w, o: False)

    voices = [('⭐ Phạm Tuyên — Nam', 'Phạm Tuyên'), ('⭐ Trúc Ly — Nữ', 'Trúc Ly')]
    entries = showcase.generate(voices=voices, make_ogg=True, force=True)

    assert calls == ['Phạm Tuyên', 'Trúc Ly']
    assert len(entries) == 2
    assert (isolated / 'Pham_Tuyen.wav').exists()
    assert (isolated / 'Truc_Ly.wav').exists()

    index = json.loads((isolated / 'index.json').read_text(encoding='utf-8'))
    assert index['voice_count'] == 2
    assert [v['id'] for v in index['voices']] == ['Pham_Tuyen', 'Truc_Ly']
    first = index['voices'][0]
    assert first['name'] == 'Phạm Tuyên'
    assert first['label'].startswith('⭐')
    assert first['wav_path'] == 'golden_samples/Pham_Tuyen.wav'
    assert first['ogg_path'] is None  # encoder giả lập thất bại
    assert first['duration_sec'] == pytest.approx(1.0, abs=0.05)


def test_generate_skips_existing_without_force(isolated, monkeypatch):
    calls: list[str] = []

    class FakeEngine:
        def __init__(self, mode, voice):
            pass

        def synthesize_segment(self, text, out_wav):
            calls.append(text)
            _write_wav(Path(out_wav), seconds=0.5)
            return 500

    monkeypatch.setattr('core.vieneu_engine.VieNeuEngine', FakeEngine)
    monkeypatch.setattr(showcase, '_wav_to_opus', lambda w, o: False)

    voices = [('L', 'Phạm Tuyên')]
    showcase.generate(voices=voices, make_ogg=False, force=True)
    assert len(calls) == 1
    showcase.generate(voices=voices, make_ogg=False, force=False)
    assert len(calls) == 1, 'không được render lại khi chưa --force'


def test_generate_ogg_entry_when_encoder_works(isolated, monkeypatch):
    class FakeEngine:
        def __init__(self, mode, voice):
            pass

        def synthesize_segment(self, text, out_wav):
            _write_wav(Path(out_wav), seconds=0.5)
            return 500

    def fake_opus(wav, ogg):
        Path(ogg).write_bytes(b'OggS')
        return True

    monkeypatch.setattr('core.vieneu_engine.VieNeuEngine', FakeEngine)
    monkeypatch.setattr(showcase, '_wav_to_opus', fake_opus)

    entries = showcase.generate(voices=[('L', 'Phạm Tuyên')], make_ogg=True, force=True)
    assert entries[0]['ogg_path'] == 'golden_samples/Pham_Tuyen.ogg'
    assert (isolated / 'Pham_Tuyen.ogg').exists()


def test_generate_requires_voices(isolated, monkeypatch):
    monkeypatch.setattr(showcase, 'get_available_preset_voices', lambda: [])
    with pytest.raises(RuntimeError, match='giọng'):
        showcase.generate(voices=None)


def test_generate_uses_live_voice_list(isolated, monkeypatch):
    seen: list[str] = []

    class FakeEngine:
        def __init__(self, mode, voice):
            seen.append(voice)

        def synthesize_segment(self, text, out_wav):
            _write_wav(Path(out_wav), seconds=0.2)
            return 200

    monkeypatch.setattr('core.vieneu_engine.VieNeuEngine', FakeEngine)
    monkeypatch.setattr(showcase, '_wav_to_opus', lambda w, o: False)
    monkeypatch.setattr(
        showcase, 'get_available_preset_voices',
        lambda: [('⭐ A — x', 'Giọng Một'), ('B — y', 'Giọng Hai')],
    )
    entries = showcase.generate(voices=None, make_ogg=False, force=True)
    assert seen == ['Giọng Một', 'Giọng Hai']
    assert [e['id'] for e in entries] == ['Giong_Mot', 'Giong_Hai']


def test_select_voices_filters_and_reports_missing(monkeypatch):
    monkeypatch.setattr(
        showcase, 'get_available_preset_voices',
        lambda: [('L1', 'Phạm Tuyên'), ('L2', 'Trúc Ly')],
    )
    assert showcase._select_voices('') is None
    chosen = showcase._select_voices('phạm tuyên')
    assert [(n) for _l, n in chosen] == ['Phạm Tuyên']
    with pytest.raises(SystemExit):
        showcase._select_voices('Không Có')


def test_ffmpeg_missing_disables_ogg(monkeypatch, tmp_path):
    monkeypatch.setattr(showcase.shutil, 'which', lambda _n: None)

    def boom():
        raise RuntimeError('no ffmpeg')

    monkeypatch.setattr(showcase, '_get_ffmpeg_bin', boom)
    wav = tmp_path / 'a.wav'
    _write_wav(wav)
    assert showcase._wav_to_opus(wav, tmp_path / 'a.ogg') is False


def test_duration_helper(tmp_path):
    wav = tmp_path / 'd.wav'
    _write_wav(wav, seconds=1.5)
    assert showcase._duration_sec(wav) == pytest.approx(1.5, abs=0.01)
    assert showcase._duration_sec(tmp_path / 'missing.wav') == 0.0
