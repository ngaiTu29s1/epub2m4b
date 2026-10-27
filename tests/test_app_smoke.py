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
        rvc_pth='',
        rvc_index='',
        f5_ref_audio=None,
        f5_ref_text='',
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

