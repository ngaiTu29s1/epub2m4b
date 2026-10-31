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
    assert 'Chương bốn' in normalize_vi_text('Chương IV: Bí mật')
    assert 'mười chín tháng năm năm một nghìn chín trăm chín mươi' in normalize_vi_text(
        'Sinh ngày 19/05/1990.'
    )


def test_markdown_to_segments_assigns_pauses_and_splits_long_sentences():
    md = '# Chương 1\n\nCâu một ngắn. Câu hai ngắn.\n\n---\n\nĐoạn mới.'
    segs = markdown_to_segments(md, max_chars=20)
    assert segs[0].is_chapter_title is True
    assert segs[0].pause_after_ms == 2000
    assert any(s.pause_after_ms == 1500 for s in segs)
    assert all(len(s.text) <= 25 for s in segs)


def test_clean_html_converts_vni_encoding_and_heals_broken_pdf_lines():
    html = '''
    <html><body>
      <h1>Chương 1</h1>
      <p>Ñöùa Treû Hö - Töû Kim Traàn.</p>
      <p>Bây giờ cậu đã tiếp</p>
      <p>xúc với hai người này một khoảng thời gian rồi.</p>
    </body></html>
    '''
    _, md = clean_html_to_markdown(html)
    assert 'Đứa Trẻ Hư - Tử Kim Trần.' in md
    assert 'Bây giờ cậu đã tiếp xúc với hai người này một khoảng thời gian rồi.' in md

    _, md_uni = clean_html_to_markdown('<p>Cảnh sát hoàn toàn không nghi ngờ toàn trường.</p>')
    assert 'hoàn toàn không nghi ngờ toàn trường' in md_uni


def test_normalizer_produces_no_arabic_digits():
    """Chốt chặn cuối: KHÔNG được còn ký tự số 0-9 nào trước khi vào TTS."""
    import re

    samples = [
        'Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi.',
        'Năm 2024 có 365 ngày, 12 tháng, 24 giờ.',
        'Giá 1.500.000 đồng, giảm 25%, còn 1,5 kg.',
        'Cao 1m76 nặng 68kg, nhiệt độ 36,5°C, tốc độ 60km/h.',
        'SĐT 0912345678, mã 007, ngày 2/9/1945.',
        'CHƯƠNG 1: 100% robot AI.',
    ]
    for s in samples:
        out = normalize_vi_text(s)
        assert not re.search(r'\d', out), f'còn chữ số trong: {out!r}'


def test_normalizer_context_aware_rules():
    assert normalize_vi_text('Chương IV') == 'Chương bốn'
    assert 'hai mươi mốt' in normalize_vi_text('thế kỷ XXI')
    assert normalize_vi_text('tháng 4') == 'tháng tư'
    assert normalize_vi_text('thứ 4') == 'thứ tư'
    assert 'một mét bảy mươi sáu' in normalize_vi_text('1m76')
    assert 'xăng-ti-mét' in normalize_vi_text('60cm')
    assert 'mười bốn giờ ba mươi phút' in normalize_vi_text('14h30')
    assert 'hai mươi lăm phần trăm' in normalize_vi_text('25%')
    assert 'một phẩy năm' in normalize_vi_text('1,5')
    assert 'sáu đến bảy mét' in normalize_vi_text('6-7 mét')
    assert 'không không bảy' in normalize_vi_text('mã 007')
    # Số thuần Bắc
    assert 'linh năm' in normalize_vi_text('105')
    assert 'không trăm linh tư' in normalize_vi_text('2004')
    assert 'hai mươi mốt' in normalize_vi_text('21')
    assert 'hai mươi tư' in normalize_vi_text('24')
    assert 'mười lăm' in normalize_vi_text('15')


def test_normalizer_abbr_and_loanwords():
    assert 'Thành phố Hồ Chí Minh' in normalize_vi_text('TP.HCM')
    assert 'Hà Nội' in normalize_vi_text('HN')
    assert 'Nhà xuất bản' in normalize_vi_text('NXB')
    assert 'in-tơ-nét' in normalize_vi_text('Internet')          # noqa: RUF001
    assert 'rô-bốt' in normalize_vi_text('robot')
    assert 'ca-mê-ra' in normalize_vi_text('camera')
    assert 'Niu Óoc' in normalize_vi_text('New York')
    assert 'Luân Đôn' in normalize_vi_text('London')

