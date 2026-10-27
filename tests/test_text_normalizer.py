from core.text_normalizer import clean_html_to_markdown, normalize_vi_text, markdown_to_segments


def test_clean_html_strips_formatting_and_keeps_scene_breaks():
    html = '''
    <html><body>
      <h1>Chương 1: Khởi đầu</h1>
      <p>Đây là <b>in đậm</b> và <i>in nghiêng</i> cùng chú thích<sup>[1]</sup>.</p>
      <hr/>
      <p>Đoạn sau chuyển cảnh tại https://example.com.</p>
    </body></html>
    '''
    title, md = clean_html_to_markdown(html)
    assert title == 'Chương 1: Khởi đầu'
    assert '**' not in md and '[1]' not in md and 'https://' not in md
    assert '---' in md
    assert md.startswith('# Chương 1: Khởi đầu')


def test_normalize_vi_text_converts_roman_and_dates():
    assert 'Chương 4' in normalize_vi_text('Chương IV: Bí mật')
    assert '19 tháng 5 năm 1990' in normalize_vi_text('Sinh ngày 19/05/1990.')


def test_markdown_to_segments_assigns_pauses_and_splits_long_sentences():
    md = '# Chương 1\n\nCâu một ngắn. Câu hai ngắn.\n\n---\n\nĐoạn mới.'
    segs = markdown_to_segments(md, max_chars=20)
    assert segs[0].is_chapter_title is True
    assert segs[0].pause_after_ms == 2000
    assert any(s.pause_after_ms == 1500 for s in segs)
    assert all(len(s.text) <= 25 for s in segs)
