import json
import wave
from pathlib import Path
import gradio as gr
from ebooklib import epub
from core.tts_engine import TTSEngine
import app


class DummyTTSEngine(TTSEngine):
    def synthesize_segment(self, text: str, out_wav: Path) -> int:
        sr = 24000
        n_frames = sr // 4  # 250ms
        with wave.open(str(out_wav), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(b'\x00\x00' * n_frames)
        return 250


def _create_harvested_voice(voices_root: Path, voice_id: str = 'vui_ve') -> Path:
    """Create a minimal harvested-voice profile on disk (as the harvester writes it)."""
    voice_dir = voices_root / voice_id
    wavs_dir = voice_dir / 'wavs'
    golden_dir = voice_dir / 'golden_refs'
    wavs_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    golden_wav = golden_dir / 'golden_01.wav'
    _write_silent_wav(golden_wav, seconds=10.0)

    (voice_dir / 'metadata.csv').write_text(
        'wavs/clip_0001.wav|Câu mẫu trong dataset\n', encoding='utf-8'
    )
    (voice_dir / 'profile.json').write_text(
        json.dumps(
            {
                'voice_id': voice_id,
                'num_clips': 27,
                'total_clean_minutes': 6.25,
                'total_clean_sec': 375.0,
                'processed_video_ids': ['vid_001', 'vid_002'],
                'best_golden_ref_wav': 'golden_refs/golden_01.wav',
                'best_golden_ref_text': 'Đàn ông và phụ nữ ở bẩn thì sẽ có đôi chút khác biệt',
                'clips': [],
                'golden_refs': [],
            },
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    return voice_dir


class _FakeHarvestResult:
    """Minimal stand-in for VoiceHarvestResult so the UI handler stays offline."""

    def __init__(self, voice_dir: Path):
        self.voice_id = voice_dir.name
        self.voice_dir = voice_dir
        self.total_clean_sec = 375.0
        self.kept_clips = [object()] * 27
        self.golden_refs = [object()]
        self.f5_metadata_path = voice_dir / 'metadata.csv'
        self.gpt_sovits_list_path = voice_dir / 'gpt_sovits.list'
        self.profile_path = voice_dir / 'profile.json'


def _write_silent_wav(path: Path, seconds: float, sample_rate: int = 24000) -> Path:
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b'\x00\x00' * int(sample_rate * seconds))
    return path


def _create_test_epub(path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier('smoke-epub-1')
    book.set_title('Sách Nói Việt Nam')
    book.set_language('vi')
    book.add_author('Tác Giả Demo')

    c1 = epub.EpubHtml(title='Chương 1', file_name='c1.xhtml', lang='vi')
    c1.content = (
        '<h1>Chương I: Khởi Đầu</h1>'
        '<p>Ngày 19/05/1990 tại TP.HCM, câu chuyện bắt đầu với những kỷ niệm đẹp đẽ.</p>'
    )
    c2 = epub.EpubHtml(title='Mục lục', file_name='toc.xhtml', lang='vi')
    c2.content = '<h1>Mục lục</h1><p>Chương 1</p>'

    book.add_item(c1)
    book.add_item(c2)
    book.spine = ['nav', c1, c2]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


def test_gradio_app_builds_and_handles_pipeline(tmp_path: Path, monkeypatch):
    demo = app.build_app()
    assert isinstance(demo, gr.Blocks)

    # The Auto Voice Harvester tab and its handlers must be wired into the UI.
    config = demo.get_config_file()
    api_names = {d.get('api_name') for d in config.get('dependencies', [])}
    assert 'handle_harvest_voice' in api_names
    assert '_load_harvested_voices' in api_names
    assert '_refresh_all_voices' in api_names

    labels = [
        ' '.join(str(component.get('props', {}).get(key, '')) for key in ('label', 'value'))
        for component in config.get('components', [])
    ]
    assert any('Auto Voice Harvester' in label for label in labels)
    assert any('Chọn giọng đọc' in label for label in labels)

    epub_path = _create_test_epub(tmp_path / 'demo.epub')
    state, title, author, cover, checkbox_update, md_text = app.handle_epub_upload(
        str(epub_path), work_root=tmp_path / 'work'
    )
    assert title == 'Sách Nói Việt Nam'
    assert author == 'Tác Giả Demo'
    assert '# Chương I: Khởi Đầu' in md_text
    assert 'Mục lục' not in md_text

    monkeypatch.setattr(app, '_create_tts_engine', lambda *a, **kw: DummyTTSEngine())

    m4b_out, m4b_file, vtt_file, timeline_rows, status_msg = app.handle_generate_m4b(
        state=state,
        book_title=title,
        book_author=author,
        md_text=md_text,
        engine_choice='Edge-TTS (Siêu nhanh)',
        voice_choice='vi-VN-NamMinhNeural',
        rate_pct=0,
        clone_ref_audio=None,
        clone_ref_text='',
        force_rerender=True,
    )
    assert Path(m4b_out).exists()
    assert Path(vtt_file).exists()
    assert len(timeline_rows) == 1
    assert timeline_rows[0][0] == 'Chương I: Khởi Đầu'
    assert 'Hoàn tất' in status_msg


def test_select_and_deselect_all_chapters():
    state = {
        'chapters': [
            {'index': 1, 'label': '[01] Chương 1', 'markdown': '# Chương 1\nNội dung 1', 'enabled': False},
            {'index': 2, 'label': '[02] Chương 2', 'markdown': '# Chương 2\nNội dung 2', 'enabled': True},
        ]
    }
    # Test Deselect All
    update_desel, md_desel = app.handle_deselect_all_chapters(state)
    assert update_desel['value'] == []
    assert md_desel == ''
    assert all(not ch['enabled'] for ch in state['chapters'])

    # Test Select All
    update_sel, md_sel = app.handle_select_all_chapters(state)
    assert update_sel['value'] == ['[01] Chương 1', '[02] Chương 2']
    assert '# Chương 1' in md_sel and '# Chương 2' in md_sel
    assert all(ch['enabled'] for ch in state['chapters'])


def test_app_harvest_voice_and_apply_golden_ref(tmp_path: Path, monkeypatch):
    """handle_harvest_voice runs the pipeline and returns summary/golden/table rows."""
    voices_root = tmp_path / 'voices'
    monkeypatch.setattr(app, 'VOICES_ROOT', voices_root)

    def fake_harvest(voice_id, query_or_url, voices_root, target_clean_minutes, max_videos, **_kw):
        voice_dir = _create_harvested_voice(Path(voices_root), voice_id)
        return _FakeHarvestResult(voice_dir)

    monkeypatch.setattr(app, 'harvest_voice_from_youtube', fake_harvest)

    summary, rows = app.handle_harvest_voice(
        voice_id='vui_ve',
        query_or_url='https://www.youtube.com/watch?v=3kqJ6u2LrNU',
        target_minutes=6.0,
        max_videos=1,
    )

    assert 'vui_ve' in summary
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == 'vui_ve'
    assert '27' in str(row[1])
    assert '6.2' in str(row[2])
    assert 'golden_01.wav' in str(row[3])

    # The harvested profile must be visible to the clone-voice resolver.
    ref_audio, ref_text = app._clone_reference_for_label('Clone · vui_ve')
    assert Path(ref_audio).exists()
    assert Path(ref_audio).name == 'golden_01.wav'
    assert ref_text == 'Đàn ông và phụ nữ ở bẩn thì sẽ có đôi chút khác biệt'


