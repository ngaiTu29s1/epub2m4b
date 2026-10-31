"""Vietnamised number/date/abbreviation normalisation for TTS."""
import pytest

from core.text_normalizer import (
    normalize_vi_text,
    number_to_words_vi,
    roman_to_int,
    roman_to_words_vi,
)


@pytest.mark.parametrize(
    'value,expected',
    [
        (0, 'không'),
        (5, 'năm'),
        (10, 'mười'),
        (15, 'mười lăm'),
        (21, 'hai mươi mốt'),
        (24, 'hai mươi tư'),
        (25, 'hai mươi lăm'),
        (100, 'một trăm'),
        (105, 'một trăm linh năm'),
        (110, 'một trăm mười'),
        (115, 'một trăm mười lăm'),
        (1990, 'một nghìn chín trăm chín mươi'),
        (1945, 'một nghìn chín trăm bốn mươi lăm'),
        (2000, 'hai nghìn'),
        (1005, 'một nghìn không trăm linh năm'),
        (1_000_000, 'một triệu'),
        (2_000_000_000, 'hai tỷ'),
        (-5, 'âm năm'),
    ],
)
def test_number_to_words_vi(value, expected):
    assert number_to_words_vi(value) == expected


@pytest.mark.parametrize(
    'roman,value',
    [('I', 1), ('IV', 4), ('IX', 9), ('XIV', 14), ('XXI', 21), ('MCMXC', 1990)],
)
def test_roman_to_int(roman, value):
    assert roman_to_int(roman) == value


def test_roman_to_int_rejects_invalid():
    assert roman_to_int('IC') is None
    assert roman_to_int('ABC') is None


def test_roman_to_words():
    assert roman_to_words_vi('XXI') == 'hai mươi mốt'
    assert roman_to_words_vi('IV') == 'bốn'


def test_chapter_roman_numerals_after_nouns():
    assert normalize_vi_text('Chương IV: Bí mật') == 'Chương bốn: Bí mật'
    assert 'thế kỷ hai mươi mốt' in normalize_vi_text('thế kỷ XXI')


def test_roman_heading_prefix():
    assert normalize_vi_text('IV. Khởi đầu') == 'bốn. Khởi đầu'


def test_full_dates():
    out = normalize_vi_text('Sinh ngày 19/05/1990.')
    assert 'mười chín tháng năm năm một nghìn chín trăm chín mươi' in out
    assert normalize_vi_text('2/9/1945') .startswith('hai tháng chín năm một nghìn chín')


def test_month_and_year_references():
    assert 'tháng tư năm một nghìn chín trăm bảy mươi lăm' in normalize_vi_text(
        'tháng 4 năm 1975'
    )


def test_ordinal_weekday():
    assert normalize_vi_text('thứ 4') == 'thứ tư'
    assert normalize_vi_text('thứ 7') == 'thứ bảy'


def test_lunar_day():
    assert normalize_vi_text('mùng 1') == 'mùng một'
    assert normalize_vi_text('mồng 1 tháng Giêng') == 'mồng một tháng Giêng'


@pytest.mark.parametrize(
    'raw,expected',
    [
        ('Anh cao 1m76.', 'Anh cao một mét bảy mươi sáu.'),
        ('Nặng 70kg.', 'Nặng bảy mươi ki-lô-gam.'),
        ('Dài 5km.', 'Dài năm ki-lô-mét.'),
        ('Cao 1,5m.', 'Cao một phẩy năm mét.'),
        ('Diện tích 20m2.', 'Diện tích hai mươi mét vuông.'),
        ('Thể tích 2m3.', 'Thể tích hai mét khối.'),
        ('Tỉ lệ 25%.', 'Tỉ lệ hai mươi lăm phần trăm.'),
        ('Nhiệt độ 37°C.', 'Nhiệt độ ba mươi bảy độ xê.'),
        ('Giá 1000 đồng.', 'Giá một nghìn đồng.'),
    ],
)
def test_measurements_and_units(raw, expected):
    assert normalize_vi_text(raw) == expected


def test_abbreviations():
    out = normalize_vi_text('TP.HCM và HN')
    assert 'Thành phố Hồ Chí Minh' in out
    assert 'Hà Nội' in out
    assert normalize_vi_text('GS. TS. Nguyễn Văn A') == 'Giáo sư Tiến sĩ Nguyễn Văn A'
    assert normalize_vi_text('PGS.TS. Trần B') == 'Phó giáo sư Tiến sĩ Trần B'


def test_etc_abbreviation():
    assert normalize_vi_text('a, b, v.v...') == 'a, b, vân vân'


def test_decimal_and_grouped_numbers():
    assert 'một phẩy năm' in normalize_vi_text('1,5')
    assert 'một nghìn hai trăm ba mươi tư' in normalize_vi_text('1.234')


def test_markdown_and_urls_are_stripped():
    out = normalize_vi_text('**Đậm** và https://example.com xong')
    assert '*' not in out
    assert 'https' not in out
    assert 'Đậm' in out


def test_plain_vietnamese_text_is_untouched():
    text = 'Trời hôm nay đẹp và trong xanh.'
    assert normalize_vi_text(text) == text
