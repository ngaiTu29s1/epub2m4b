import json
import subprocess
import wave
from pathlib import Path
from core.models import BookMetadata, Chapter
from core.tts_engine import TTSEngine, render_chapter
from core.m4b_packager import build_ffmetadata, export_webvtt, package_m4b


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
