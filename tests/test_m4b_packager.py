import json
import subprocess
import wave
from pathlib import Path
from core.models import BookMetadata, Chapter
from core.tts_engine import TTSEngine, render_chapter
from core.m4b_packager import (
    build_ffmetadata,
    encode_wav_to_m4a,
    export_webvtt,
    package_m4b,
    resolve_package_sources,
)


class DummyTTSEngine(TTSEngine):
    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        sr = 24000
        n_frames = sr // 2  # 500ms
        with wave.open(str(out_wav), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b'\x00\x00' * n_frames)
        return 500


def test_build_ffmetadata_and_package_m4b_with_chapters_and_vtt(tmp_path: Path):
    c1 = Chapter(index=1, title='Chương 1', markdown='# Chương 1\n\nCâu mở đầu chương một.')
    c2 = Chapter(index=2, title='Chương 2', markdown='# Chương 2\n\nCâu mở đầu chương hai.')
    engine = DummyTTSEngine()
    r1 = render_chapter(c1, engine, tmp_path / 'chaps')
    r2 = render_chapter(c2, engine, tmp_path / 'chaps')

    meta = BookMetadata(
        title='Sách Thử Nghiệm',
        author='Tác Giả Việt',
        cover_path=None,
        chapters=[c1, c2],
    )

    ffmeta = build_ffmetadata(meta, [r1, r2])
    assert ';FFMETADATA1' in ffmeta
    assert 'title=Sách Thử Nghiệm' in ffmeta
    assert 'artist=Tác Giả Việt' in ffmeta
    assert 'title=Chương 1' in ffmeta
    assert 'title=Chương 2' in ffmeta

    vtt_path = export_webvtt([r1, r2], tmp_path / 'out.vtt')
    vtt_text = vtt_path.read_text(encoding='utf-8')
    assert vtt_text.startswith('WEBVTT')
    assert 'Câu mở đầu chương một.' in vtt_text
    assert 'Câu mở đầu chương hai.' in vtt_text

    m4b_path = package_m4b(meta, [r1, r2], tmp_path / 'out.m4b')
    assert m4b_path.exists()
    assert m4b_path.stat().st_size > 0

    ffprobe_bin = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'ffprobe'
    probe_out = subprocess.check_output(
        [str(ffprobe_bin), '-v', 'error', '-show_chapters', '-of', 'json', str(m4b_path)],
        text=True,
    )
    probe_data = json.loads(probe_out)
    chapters = probe_data.get('chapters', [])
    assert len(chapters) == 2
    titles = [ch.get('tags', {}).get('title') for ch in chapters]
    assert titles == ['Chương 1', 'Chương 2']


# ---------------------------------------------------------------------------
# encode_wav_to_m4a
# ---------------------------------------------------------------------------
def _make_wav(path: Path, ms: int = 500) -> Path:
    sr = 24000
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b'\x00\x00' * (sr * ms // 1000))
    return path


def test_encode_wav_to_m4a_creates_file(tmp_path: Path):
    wav = _make_wav(tmp_path / 'chapter_001.wav')
    out = encode_wav_to_m4a(wav)
    assert out == tmp_path / 'chapter_001.m4a'
    assert out.exists() and out.stat().st_size > 0


def test_encode_wav_to_m4a_custom_output_path(tmp_path: Path):
    wav = _make_wav(tmp_path / 'a.wav')
    target = tmp_path / 'sub' / 'custom.m4a'
    out = encode_wav_to_m4a(wav, target)
    assert out == target
    assert target.exists()


def test_encode_wav_to_m4a_uses_cache(tmp_path: Path, monkeypatch):
    import core.m4b_packager as pkg
    wav = _make_wav(tmp_path / 'c.wav')
    out = encode_wav_to_m4a(wav)

    called = {'n': 0}

    def spy(*a, **k):
        called['n'] += 1
        raise AssertionError('không được encode lại khi còn cache')

    monkeypatch.setattr(pkg.subprocess, 'run', spy)
    assert encode_wav_to_m4a(wav) == out
    assert called['n'] == 0


def test_encode_wav_to_m4a_force_reencodes(tmp_path: Path, monkeypatch):
    import core.m4b_packager as pkg
    wav = _make_wav(tmp_path / 'd.wav')
    encode_wav_to_m4a(wav)

    real_run = pkg.subprocess.run
    calls = {'n': 0}

    def counting(cmd, **kw):
        calls['n'] += 1
        return real_run(cmd, **kw)

    monkeypatch.setattr(pkg.subprocess, 'run', counting)
    encode_wav_to_m4a(wav, force=True)
    assert calls['n'] == 1


def test_encode_wav_to_m4a_reenables_when_wav_newer(tmp_path: Path):
    import os
    wav = _make_wav(tmp_path / 'e.wav')
    out = encode_wav_to_m4a(wav)
    # WAV mới hơn .m4a → cache cũ không còn hợp lệ.
    os.utime(out, (out.stat().st_atime, out.stat().st_mtime - 100))
    stat_new = encode_wav_to_m4a(wav)
    assert stat_new.stat().st_mtime >= wav.stat().st_mtime


def test_encode_wav_to_m4a_ignores_empty_output(tmp_path: Path):
    wav = _make_wav(tmp_path / 'f.wav')
    empty = tmp_path / 'f.m4a'
    empty.write_bytes(b'')
    out = encode_wav_to_m4a(wav)
    assert out.stat().st_size > 0


def test_encode_wav_to_m4a_command_has_aac_flags(tmp_path: Path, monkeypatch):
    import core.m4b_packager as pkg
    captured = {}

    def capture(cmd, **kw):
        captured['cmd'] = cmd
        Path(cmd[-1]).write_bytes(b'FAKE')

    monkeypatch.setattr(pkg.subprocess, 'run', capture)
    wav = _make_wav(tmp_path / 'g.wav')
    encode_wav_to_m4a(wav)
    cmd = captured['cmd']
    assert 'aac' in cmd
    assert '-c:a' in cmd and cmd[cmd.index('-c:a') + 1] == 'aac'
    assert '-b:a' in cmd and cmd[cmd.index('-b:a') + 1] == '64k'
    assert '-ac' in cmd and cmd[cmd.index('-ac') + 1] == '1'


# ---------------------------------------------------------------------------
# resolve_package_sources / package_m4b: stream copy vs fallback
# ---------------------------------------------------------------------------
def _chapters_and_meta(tmp_path: Path):
    c1 = Chapter(index=1, title='Chương 1', markdown='# Chương 1\n\nNội dung một.')
    c2 = Chapter(index=2, title='Chương 2', markdown='# Chương 2\n\nNội dung hai.')
    engine = DummyTTSEngine()
    chaps = tmp_path / 'chaps'
    r1 = render_chapter(c1, engine, chaps)
    r2 = render_chapter(c2, engine, chaps)
    meta = BookMetadata(title='Sách Pipeline', author='Tác Giả', cover_path=None, chapters=[c1, c2])
    return [r1, r2], meta


def test_resolve_package_sources_stream_copy_when_all_m4a(tmp_path: Path):
    results, _meta = _chapters_and_meta(tmp_path)
    for r in results:
        encode_wav_to_m4a(r.audio_path)
    sources, stream_copy = resolve_package_sources(results)
    assert stream_copy is True
    assert [s.suffix for s in sources] == ['.m4a', '.m4a']


def test_resolve_package_sources_fallback_when_wav_only(tmp_path: Path):
    results, _meta = _chapters_and_meta(tmp_path)
    sources, stream_copy = resolve_package_sources(results)
    assert stream_copy is False
    assert [s.suffix for s in sources] == ['.wav', '.wav']


def test_resolve_package_sources_partial_m4a_falls_back(tmp_path: Path):
    """Chỉ 1 chương có .m4a → phải fallback để tránh trộn WAV/M4A."""
    results, _meta = _chapters_and_meta(tmp_path)
    encode_wav_to_m4a(results[0].audio_path)
    sources, stream_copy = resolve_package_sources(results)
    assert stream_copy is False
    assert [s.suffix for s in sources] == ['.wav', '.wav']


def test_resolve_package_sources_ignores_empty_m4a(tmp_path: Path):
    results, _meta = _chapters_and_meta(tmp_path)
    for r in results:
        Path(r.audio_path).with_suffix('.m4a').write_bytes(b'')
    sources, stream_copy = resolve_package_sources(results)
    assert stream_copy is False


def test_package_uses_stream_copy_when_m4a_present(tmp_path: Path, monkeypatch):
    import core.m4b_packager as pkg
    results, meta = _chapters_and_meta(tmp_path)
    for r in results:
        encode_wav_to_m4a(r.audio_path)

    cmds = []
    real_run = pkg.subprocess.run

    def spy(cmd, **kw):
        cmds.append(list(cmd))
        return real_run(cmd, **kw)

    monkeypatch.setattr(pkg.subprocess, 'run', spy)
    out = package_m4b(meta, results, tmp_path / 'copy.m4b')
    assert out.exists() and out.stat().st_size > 0
    ffmpeg_cmd = cmds[-1]
    assert '-c:a' in ffmpeg_cmd
    assert ffmpeg_cmd[ffmpeg_cmd.index('-c:a') + 1] == 'copy'


def test_package_falls_back_to_reencode_when_wav_only(tmp_path: Path, monkeypatch):
    import core.m4b_packager as pkg
    results, meta = _chapters_and_meta(tmp_path)

    cmds = []
    real_run = pkg.subprocess.run

    def spy(cmd, **kw):
        cmds.append(list(cmd))
        return real_run(cmd, **kw)

    monkeypatch.setattr(pkg.subprocess, 'run', spy)
    out = package_m4b(meta, results, tmp_path / 'reenc.m4b')
    assert out.exists() and out.stat().st_size > 0
    ffmpeg_cmd = cmds[-1]
    assert ffmpeg_cmd[ffmpeg_cmd.index('-c:a') + 1] == 'aac'
    assert ffmpeg_cmd[ffmpeg_cmd.index('-b:a') + 1] == '64k'


def test_package_stream_copy_keeps_chapters(tmp_path: Path):
    results, meta = _chapters_and_meta(tmp_path)
    for r in results:
        encode_wav_to_m4a(r.audio_path)
    out = package_m4b(meta, results, tmp_path / 'chapters.m4b')

    ffprobe_bin = Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'ffprobe'
    probe = json.loads(subprocess.check_output(
        [str(ffprobe_bin), '-v', 'error', '-show_chapters', '-of', 'json', str(out)]
    ))
    assert len(probe['chapters']) == 2
    titles = [c['tags']['title'] for c in probe['chapters']]
    assert titles == ['Chương 1', 'Chương 2']
