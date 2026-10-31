"""Golden-dataset rules: keep English words, read maths, chemistry, units, orgs.

Bổ sung cho spec ``2026-10-31-text-normalizer-v3-golden-dataset-speedup.md``.
"""
import re

import pytest

from core import llm_normalizer as ln
from core.text_normalizer import normalize_vi_text


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Không để test ghi vào ``workspace/dictionary_cache.json`` của repo."""
    monkeypatch.setattr(ln, '_RUNTIME_CACHE_PATH', tmp_path / 'dict.json')
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    ln.reload_cache()
    yield
    ln.reload_cache()


# ---------------------------------------------------------------------------
# 1. Từ mượn tiếng Anh giữ nguyên cho engine song ngữ
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'word',
    ['video', 'camera', 'online', 'robot', 'internet', 'website', 'email',
     'data', 'radio', 'logic', 'computer', 'software', 'hardware', 'smartphone'],
)
def test_loanwords_are_not_transliterated(word):
    assert normalize_vi_text(word) == word
    assert '-' not in normalize_vi_text(word) or word in {'off-line'}


def test_bilingual_sentence_keeps_english_and_sino_vietnamese_places():
    out = normalize_vi_text('Machine Learning và video call tại Bắc Kinh, Luân Đôn.')
    assert 'Machine Learning' in out
    assert 'video call' in out
    assert 'Bắc Kinh' in out
    assert 'Luân Đôn' in out


# ---------------------------------------------------------------------------
# 2. Viết tắt chính thức vẫn được chuẩn hoá
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'raw,expected',
    [
        ('TP.HCM', 'Thành phố Hồ Chí Minh'),
        ('UBND', 'Uỷ ban nhân dân'),
        ('THPT', 'Trung học phổ thông'),
        ('NXB', 'Nhà xuất bản'),
    ],
)
def test_official_abbreviations(raw, expected):
    assert expected in normalize_vi_text(raw)


# ---------------------------------------------------------------------------
# 3. Công thức hoá học
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'raw,expected',
    [
        ('H2SO4', 'H hai S O bốn'),
        ('CO2', 'C O hai'),
        ('NaCl', 'N A C L'),
        ('KMnO4', 'K M N O bốn'),
    ],
)
def test_chemical_formulas(raw, expected):
    assert expected in normalize_vi_text(f'Axit {raw} trong thí nghiệm.')


# ---------------------------------------------------------------------------
# 4. Toán học
# ---------------------------------------------------------------------------
def test_equation_from_cache():
    out = normalize_vi_text('Theo E = mc^2, khối lượng và năng lượng tương đương.')
    assert 'E bằng m c bình phương' in out


def test_unknown_equation_degrades_to_rule_based():
    out = normalize_vi_text('Phương trình x^2 + 2x + 1 = 0 có nghiệm kép.')
    assert 'bình phương' in out
    assert '^' not in out
    assert not re.search(r'\d', out)


def test_generic_exponent_and_symbols():
    assert 'x mũ năm' in normalize_vi_text('x^5')
    assert 'an-pha' in normalize_vi_text('góc α')
    assert 'bê-ta' in normalize_vi_text('hệ số β')
    assert 'căn bậc hai' in normalize_vi_text('√9')


def test_fraction_is_not_a_date():
    assert 'ba phần tư' in normalize_vi_text('tỉ lệ 3/4')


def test_dates_still_win_over_fractions():
    assert 'ba mươi tháng tư' in normalize_vi_text('30/04')
    assert 'mười chín tháng năm' in normalize_vi_text('19/05')


# ---------------------------------------------------------------------------
# 5. Đơn vị đo lường
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'raw,expected',
    [
        ('3.5 GHz', 'ba phẩy năm giga héc'),
        ('2.4 GHz', 'hai phẩy bốn giga héc'),
        ('60 km/h', 'sáu mươi ki-lô-mét trên giờ'),
        ('37.5 độ C', 'ba mươi bảy phẩy năm độ xê'),
        ('37.5°C', 'ba mươi bảy phẩy năm độ xê'),
        ('25%', 'hai mươi lăm phần trăm'),
    ],
)
def test_units(raw, expected):
    assert expected in normalize_vi_text(f'Chỉ số {raw} hôm nay.')


# ---------------------------------------------------------------------------
# 6. Roman numerals & dates
# ---------------------------------------------------------------------------
def test_roman_and_date():
    assert 'hai mươi mốt' in normalize_vi_text('Thế kỷ XXI')
    assert 'ba mươi tháng tư năm một nghìn chín trăm bảy mươi lăm' in normalize_vi_text(
        'ngày 30/04/1975'
    )


def test_no_digits_anywhere_in_golden_sentence():
    raw = (
        'Machine Learning, H2SO4, CO2, E = mc^2, x^2 + 2x + 1 = 0, TP.HCM, '
        'UNESCO, COVID-19, WHO, 3.5 GHz, 60 km/h, 37.5 độ C, 25%, '
        'Thế kỷ XXI, ngày 30/04/1975.'
    )
    out = normalize_vi_text(raw)
    assert not re.search(r'\d', out), out
    assert not re.search(r'[=^]', out), out


# ---------------------------------------------------------------------------
# 7. LLM fallback + auto-enrich (transport giả, không gọi mạng)
# ---------------------------------------------------------------------------
def test_llm_writes_to_cache_and_reuses_it(monkeypatch):
    calls: list[str] = []

    def fake_call(text, *, timeout=30.0):
        calls.append(text)
        return text.replace('Zz7', 'gi-ét').replace(' = ', ' bằng ')

    monkeypatch.setenv('LLM_API_KEY', 'test-key')
    monkeypatch.setattr(ln, '_call_llm', fake_call)

    first = ln.enrich_and_normalize('Biểu thức Zz7 = 9 rất lạ.')
    assert 'gi-ét bằng' in first
    assert len(calls) == 1
    assert ln.lookup('Zz7') == 'gi-ét'

    # Lần hai: cache đã học, không gọi LLM nữa.
    second = ln.enrich_and_normalize('Biểu thức Zz7 = 9 rất lạ.')
    assert 'gi-ét bằng' in second
    assert len(calls) == 1


def test_missing_api_key_never_calls_llm(monkeypatch):
    def boom(*_a, **_kw):
        raise AssertionError('không được gọi LLM khi thiếu key')

    monkeypatch.setattr(ln, '_call_llm', boom)
    out = ln.enrich_and_normalize('Phương trình q^2 = 4.')
    assert isinstance(out, str) and out


def test_llm_http_failure_degrades(monkeypatch):
    monkeypatch.setenv('LLM_API_KEY', 'k')
    monkeypatch.setenv('LLM_BASE_URL', 'http://127.0.0.1:9/v1')
    monkeypatch.setenv('LLM_TIMEOUT', '1')
    out = ln.enrich_and_normalize('E = mc^2')
    assert isinstance(out, str) and out


def test_seed_cache_entries():
    assert ln.lookup('H2SO4') == 'H hai S O bốn'
    assert ln.lookup('CO2') == 'C O hai'
    assert ln.lookup('NaCl') == 'N A C L'
    assert ln.lookup('E = mc^2') == 'E bằng m c bình phương'
    assert ln.lookup('x^2 + y^2 = z^2') == (
        'x bình phương cộng y bình phương bằng z bình phương'
    )


def test_seed_cache_file_is_valid_json_with_required_seeds():
    import json

    from core.llm_normalizer import _CACHE_SEED_PATH

    data = json.loads(_CACHE_SEED_PATH.read_text(encoding='utf-8'))
    entries = data.get('entries', data)
    for key in ('h2so4', 'co2', 'nacl', 'e = mc^2', 'x^2 + 2x + 1 = 0'):
        assert key in entries, f'thiếu seed {key!r}'
