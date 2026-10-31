"""Uncertainty scanner: ``detect_uncertain_spans`` (spec 2026-10-31 §2.4)."""
from core.text_normalizer import (
    UncertainSpan,
    detect_uncertain_spans,
    uncertain_spans_to_review,
)


def _reasons(text: str) -> list[str]:
    return [s['reason'] for s in detect_uncertain_spans(text)]


def _tokens(text: str) -> list[str]:
    return [s['token'] for s in detect_uncertain_spans(text)]


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
def test_empty_text_returns_empty_list():
    assert detect_uncertain_spans('') == []
    assert detect_uncertain_spans('   \n\t ') == []


def test_span_shape_and_types():
    spans = detect_uncertain_spans('Mã Zz7 xuất hiện.')
    assert len(spans) == 1
    span = spans[0]
    assert set(span) == {
        'token', 'sentence', 'reason', 'start', 'end', 'suggested_reading',
    }
    assert isinstance(span['token'], str)
    assert isinstance(span['start'], int) and isinstance(span['end'], int)
    assert span['start'] < span['end']


def test_span_offsets_point_at_token():
    text = 'Câu đầu. Mã Zz7 ở đây.'
    span = detect_uncertain_spans(text)[0]
    assert text[span['start']:span['end']] == span['token']


def test_span_carries_full_sentence_context():
    text = 'Đây là câu một. Máy bay F-35 bay qua. Câu ba.'
    span = next(s for s in detect_uncertain_spans(text) if s['token'] == 'F-35')
    assert span['sentence'] == 'Máy bay F-35 bay qua.'


def test_spans_sorted_by_position():
    spans = detect_uncertain_spans('F-35 rồi Zz7 rồi FAO.')
    starts = [s['start'] for s in spans]
    assert starts == sorted(starts)


def test_uncertain_span_dataclass_to_dict():
    span = UncertainSpan(
        token='x', sentence='x', reason='math_equation', start=0, end=1,
        suggested_reading='ích',
    )
    d = span.to_dict()
    assert d['token'] == 'x' and d['suggested_reading'] == 'ích'


def test_review_helper_drops_offsets():
    review = uncertain_spans_to_review('Mã Zz7.')
    assert set(review[0]) == {'token', 'sentence', 'reason'}


# ---------------------------------------------------------------------------
# math_equation
# ---------------------------------------------------------------------------
def test_detects_equation_with_exponent():
    spans = detect_uncertain_spans('Ta có E = mc^2 trong vật lý.')
    assert 'math_equation' in _reasons('Ta có E = mc^2 trong vật lý.')
    eq = next(s for s in spans if s['reason'] == 'math_equation')
    assert '=' in eq['token'] and '^' in eq['token']


def test_detects_quadratic_expression():
    tokens = _tokens('Phương trình x^2 + 2x + 1 = 0 có nghiệm kép.')
    assert any('=' in t and '^' in t for t in tokens)


def test_detects_sqrt_and_integral_symbols():
    assert 'math_equation' in _reasons('Tính √9 và ∫x dx.')
    assert 'math_equation' in _reasons('Dùng ký hiệu ∫ ở đây.')


def test_math_suggestion_is_spoken_form():
    span = next(
        s for s in detect_uncertain_spans('Ta có E = mc^2.') if s['reason'] == 'math_equation'
    )
    assert span['suggested_reading']
    assert '=' not in span['suggested_reading']
    assert '^' not in span['suggested_reading']


def test_no_duplicate_math_spans_for_same_expression():
    spans = [s for s in detect_uncertain_spans('x^2 + 2x + 1 = 0') if s['reason'] == 'math_equation']
    assert len(spans) == 1


# ---------------------------------------------------------------------------
# chem_formula
# ---------------------------------------------------------------------------
def test_flags_unknown_chemical_formula():
    spans = detect_uncertain_spans('Hợp chất XY3Z9 chưa rõ.')
    assert [s['reason'] for s in spans] == ['chem_formula']
    assert spans[0]['token'] == 'XY3Z9'


def test_known_cached_formulas_are_not_flagged():
    """H2SO4/CO2/NaCl đã có trong từ điển → không phải điểm nghi vấn."""
    assert detect_uncertain_spans('Có H2SO4, CO2 và NaCl.') == []


def test_short_simple_chemic_is_not_flagged():
    # H2O/CO2/NaCl là công thức hoá học rõ ràng; O2 (đơn chất + chỉ số) vẫn
    # được coi là công thức.
    assert detect_uncertain_spans('Khí O2 và H2O.') == []


def test_uppercase_acronym_is_not_chem_formula():
    """UNESCO khớp khuôn ký tự hoá học nhưng là viết tắt đã biết."""
    reasons = _reasons('UNESCO và WHO làm việc.')
    assert 'chem_formula' not in reasons


def test_unknown_acronym_is_not_chem_formula():
    """FAO là viết tắt lạ → unknown_abbr, không phải chem_formula."""
    spans = detect_uncertain_spans('FAO họp hôm nay.')
    assert [s['reason'] for s in spans] == ['unknown_abbr']


# ---------------------------------------------------------------------------
# mixed_alphanumeric
# ---------------------------------------------------------------------------
def test_flags_f35_style_token():
    assert 'mixed_alphanumeric' in _reasons('Máy bay F-35 hạ cánh.')
    assert 'F-35' in _tokens('Máy bay F-35 hạ cánh.')


def test_flags_zz7_style_token():
    assert 'Zz7' in _tokens('Mã Zz7 là gì?')


def test_flags_covid19_style_token():
    assert 'COVID-19' in _tokens('Đại dịch COVID-19 bùng phát.')


def test_known_cache_tokens_not_flagged_as_mixed():
    assert detect_uncertain_spans('Có H2SO4 ở đây.') == []


# ---------------------------------------------------------------------------
# unknown_abbr
# ---------------------------------------------------------------------------
def test_flags_unknown_abbreviation():
    spans = detect_uncertain_spans('FAO và BOLA họp.')
    tokens = [s['token'] for s in spans]
    assert 'FAO' in tokens and 'BOLA' in tokens
    assert all(s['reason'] == 'unknown_abbr' for s in spans)


def test_known_official_abbreviations_not_flagged():
    text = 'UBND TP.HCM và HĐND, THPT, NXB, GS, TS đều ổn.'
    assert detect_uncertain_spans(text) == []


def test_sentence_case_words_not_flagged_as_abbr():
    """Chữ hoa đầu câu ("Trời", "Chúng") không phải viết tắt."""
    assert detect_uncertain_spans('Trời đẹp. Chúng ta đi. Hôm nay vui.') == []


def test_normal_prose_has_no_false_alarms():
    """Văn xuôi bình thường phải im lặng hoàn toàn."""
    prose = (
        'Buổi sáng hôm đó, ông lão ra đồng từ rất sớm. Trời còn se lạnh và sương '
        'phủ trắng những ngọn cỏ. Ông nhìn về phía chân trời, nơi mặt trời đang '
        'dần nhô lên sau rặng tre. Một ngày mới lại bắt đầu trên làng quê nghèo.'
    )
    assert detect_uncertain_spans(prose) == []


def test_prose_with_numbers_is_quiet():
    text = 'Giá 37,5 độ C, độ ẩm 25%, tốc độ 60 km/h, ngày 30/04/1975.'
    assert detect_uncertain_spans(text) == []


# ---------------------------------------------------------------------------
# fraction
# ---------------------------------------------------------------------------
def test_flags_ambiguous_fraction():
    spans = detect_uncertain_spans('Tỷ lệ 3/4 và 1/2.')
    fractions = [s for s in spans if s['reason'] == 'fraction']
    assert [s['token'] for s in fractions] == ['3/4', '1/2']


def test_fraction_suggestion_is_readable():
    span = next(
        s for s in detect_uncertain_spans('Tỷ lệ 3/4.') if s['reason'] == 'fraction'
    )
    assert span['suggested_reading'] == 'ba phần tư'


def test_dates_are_not_flagged_as_fraction():
    assert detect_uncertain_spans('Ngày 30/04/1975 rất quan trọng.') == []


def test_decimal_and_percent_not_flagged_as_fraction():
    assert detect_uncertain_spans('Nhiệt độ 37,5 độ, tăng 25%.') == []


# ---------------------------------------------------------------------------
# Combined / UI use
# ---------------------------------------------------------------------------
def test_mixed_document_reports_all_categories():
    text = (
        'WHO cảnh báo đại dịch COVID-19. Phòng thí nghiệm có XY3Z9 lạ. '
        'Máy bay F-35 xuất hiện. Ta có E = mc^2. Tỷ lệ 3/4 khá cao. '
        'Đại diện FAO phát biểu.'
    )
    reasons = set(_reasons(text))
    assert {'mixed_alphanumeric', 'chem_formula', 'unknown_abbr',
            'math_equation', 'fraction'} <= reasons


def test_scanner_handles_vietnamese_diacritics():
    assert detect_uncertain_spans('Tỷ lệ ½ không nằm trong danh sách.') == []
    assert 'Zz7' in _tokens('Tôi thấy mã Zz7 ở đây.')
