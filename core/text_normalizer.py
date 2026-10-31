"""Chuẩn hoá văn bản tiếng Việt cho TTS (miền Bắc).

Chuyển số, số La Mã, ngày tháng, đơn vị và viết tắt sang **chữ tiếng Việt
thuần** để giọng đọc không bao giờ đọc sai (ví dụ ``Chương IV`` đọc thành
"chương bốn" chứ không phải "chương i vê").
"""
import re
import unicodedata
import warnings
from dataclasses import dataclass
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from core.models import AudioSegmentSpec
from core.llm_normalizer import enrich_and_normalize, lookup

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
# ---------------------------------------------------------------------------
# Số → chữ tiếng Việt (chuẩn miền Bắc)
# ---------------------------------------------------------------------------
_DIGIT_WORDS = {
    0: 'không', 1: 'một', 2: 'hai', 3: 'ba', 4: 'bốn', 5: 'năm',
    6: 'sáu', 7: 'bảy', 8: 'tám', 9: 'chín',
}
_SCALE_WORDS = {3: 'nghìn', 6: 'triệu', 9: 'tỷ', 12: 'nghìn tỷ', 15: 'triệu tỷ'}


def _read_group_of_three(n: int, *, full: bool) -> str:
    """Đọc một nhóm 0..999.

    ``full=True`` luôn đọc đủ, kể cả nhóm đứng giữa câu (``một trăm linh năm``).
    ``full=False`` dùng cho nhóm cao nhất (không có tiền tố, không có "linh").
    """
    words: list[str] = []
    hundreds = n // 100
    tens = (n // 10) % 10
    units = n % 10

    if hundreds > 0 or full:
        words.append(f'{_DIGIT_WORDS[hundreds]} trăm')
        if tens == 0 and units > 0:
            words.append('linh')

    if tens > 1:
        words.append(f'{_DIGIT_WORDS[tens]} mươi')
        if units == 1:
            words.append('mốt')
        elif units == 4:
            words.append('tư')
        elif units == 5:
            words.append('lăm')
        elif units > 0:
            words.append(_DIGIT_WORDS[units])
    elif tens == 1:
        words.append('mười')
        if units == 1:
            words.append('một')
        elif units == 5:
            words.append('lăm')
        elif units > 0:
            words.append(_DIGIT_WORDS[units])
    elif units > 0:
        # Sau "linh" (hàng chục = 0), tiếng Việt chuẩn đọc là "linh một",
        # "linh tư", "linh năm" (không đọc "linh mốt" hay "linh lăm").
        if hundreds > 0 or full:
            if units == 4:
                words.append('tư')
            else:
                words.append(_DIGIT_WORDS[units])
        else:
            words.append(_DIGIT_WORDS[units])

    return ' '.join(w for w in words if w)


def number_to_words_vi(number: int) -> str:
    """Đọc số nguyên thành chữ tiếng Việt miền Bắc chuẩn.

    >>> number_to_words_vi(1990)
    'một nghìn chín trăm chín mươi'
    >>> number_to_words_vi(105)
    'một trăm linh năm'
    >>> number_to_words_vi(21)
    'hai mươi mốt'
    """
    number = int(number)
    if number == 0:
        return 'không'
    if number < 0:
        return 'âm ' + number_to_words_vi(-number)

    groups: list[int] = []
    while number > 0:
        groups.append(number % 1000)
        number //= 1000

    parts: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        group = groups[idx]
        if group == 0:
            continue
        is_most_significant = not parts
        text = _read_group_of_three(group, full=not is_most_significant)
        scale = _SCALE_WORDS.get(idx * 3) if idx > 0 else None
        parts.append(f'{text} {scale}' if scale else text)
    return ' '.join(parts)


def _read_decimal(digits: str) -> str:
    """Đọc phần thập phân theo từng chữ số (``1,5`` → "một phẩy năm")."""
    return ' '.join(_DIGIT_WORDS[int(d)] for d in digits if d.isdigit())


# ---------------------------------------------------------------------------
# Số La Mã
# ---------------------------------------------------------------------------
_ROMAN_VALUES = (
    ('M', 1000), ('CM', 900), ('D', 500), ('CD', 400),
    ('C', 100), ('XC', 90), ('L', 50), ('XL', 40),
    ('X', 10), ('IX', 9), ('V', 5), ('IV', 4), ('I', 1),
)
_ROMAN_VALID_RE = re.compile(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$')


def roman_to_int(roman: str) -> int | None:
    """Chuyển số La Mã chuẩn tắc thành int, trả ``None`` nếu không hợp lệ."""
    text = roman.upper().strip()
    if not text or not _ROMAN_VALID_RE.match(text):
        return None
    value = 0
    idx = 0
    for sym, val in _ROMAN_VALUES:
        while text.startswith(sym, idx):
            value += val
            idx += len(sym)
    return value if idx == len(text) else None


def roman_to_words_vi(roman: str) -> str | None:
    value = roman_to_int(roman)
    return number_to_words_vi(value) if value else None


# Chỉ các danh từ đứng trước mới được phép coi token La Mã là số thứ tự.
_ROMAN_PREFIX = (
    r'chương|phần|quyển|tập|hồi|mục|tiết|chương trình|điều|khoản|'
    r'thế kỷ|thiên niên kỷ|vương triều|đời|triều đại|tập phim|kỳ|kì|'
    r'quân đoàn|chiến dịch|liên đoàn|olympic|thế vận hội'
)
_ROMAN_AFTER_PREFIX_RE = re.compile(
    rf'\b((?:{_ROMAN_PREFIX})\s+)([IVXLCDM]{{1,6}})\b',
    re.IGNORECASE,
)
# Đề mục đứng ở đầu dòng/dấu câu: "I. Khởi đầu", "IV - Kết thúc"
_ROMAN_HEADING_RE = re.compile(
    r'(?:(?<=^)|(?<=[\n\.;:!?]\s)|(?<=\n))([IVXLCDM]{1,6})(?=[\.\)\-–—:]\s)',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Ngày tháng năm
# ---------------------------------------------------------------------------
def _day_to_words(day: int) -> str:
    """``19`` → "mười chín", ``21`` → "hai mươi mốt", ``15`` → "mười lăm"."""
    if day == 1:
        return 'mồng một'
    return number_to_words_vi(day)


_MONTH_WORDS = {
    1: 'giêng', 2: 'hai', 3: 'ba', 4: 'tư', 5: 'năm', 6: 'sáu',
    7: 'bảy', 8: 'tám', 9: 'chín', 10: 'mười', 11: 'mười một', 12: 'mười hai',
}
_ORDINAL_MONTH_WORDS = {
    1: 'nhất', 2: 'hai', 3: 'ba', 4: 'tư', 5: 'năm', 6: 'sáu',
    7: 'bảy', 8: 'tám', 9: 'chín', 10: 'mười', 11: 'mười một', 12: 'mười hai',
}
_ORDINAL_WEEKDAY = {
    2: 'hai', 3: 'ba', 4: 'tư', 5: 'năm', 6: 'sáu', 7: 'bảy', 8: 'chủ nhật',
}


def _month_words(month: int) -> str:
    return _MONTH_WORDS.get(month, number_to_words_vi(month))


def _normalize_full_date(text: str) -> str:
    """``19/05/1990`` → "mười chín tháng năm năm một nghìn chín trăm chín mươi"."""

    def repl(m: re.Match) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return m.group(0)
        return (
            f'{_day_to_words(day)} tháng {_month_words(month)} '
            f'năm {number_to_words_vi(year)}'
        )

    return re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', repl, text)


def _normalize_short_date(text: str) -> str:
    """``19/05`` → "mười chín tháng năm" (chỉ khi hợp lệ như một ngày)."""

    def repl(m: re.Match) -> str:
        day, month = int(m.group(1)), int(m.group(2))
        # Phân số đơn giản "3/4", "1/2" là phân số, không phải ngày tháng.
        if m.group(1).isdigit() and m.group(2).isdigit() and day < month <= 12:
            return m.group(0)
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f'{_day_to_words(day)} tháng {_month_words(month)}'
        return m.group(0)

    return re.sub(r'(?<![\d/])\b(\d{1,2})/(\d{1,2})\b(?![\d/])', repl, text)


def _normalize_standalone_year(text: str) -> str:
    """Năm đứng một mình (``năm 1990``, ``năm 1945``) đọc đủ theo nhóm nghìn."""

    def repl(m: re.Match) -> str:
        return f'năm {number_to_words_vi(int(m.group(1)))}'

    return re.sub(r'\bnăm\s+(\d{4})\b', repl, text, flags=re.IGNORECASE)


def _normalize_month_ref(text: str) -> str:
    """``tháng 4`` → "tháng tư"; ``tháng 12`` → "tháng mười hai"."""

    def repl(m: re.Match) -> str:
        month = int(m.group(1))
        return f'tháng {_month_words(month)}' if 1 <= month <= 12 else m.group(0)

    text = re.sub(r'\btháng\s+(\d{1,2})\b', repl, text, flags=re.IGNORECASE)

    def repl_year(m: re.Match) -> str:
        return f'tháng {_month_words(int(m.group(1)))} năm {number_to_words_vi(int(m.group(2)))}'

    return re.sub(
        r'\btháng\s+(\d{1,2})\s*[/-]\s*(\d{4})\b', repl_year, text, flags=re.IGNORECASE
    )


def _normalize_ordinal_day(text: str) -> str:
    """``thứ 4`` → "thứ tư"; ``ngày 2/9`` đã được xử lý ở nhánh ngày tháng."""

    def repl(m: re.Match) -> str:
        day = int(m.group(1))
        return f'thứ {_ORDINAL_WEEKDAY[day]}' if day in _ORDINAL_WEEKDAY else m.group(0)

    return re.sub(r'\bthứ\s+(\d{1,2})\b', repl, text, flags=re.IGNORECASE)


def _normalize_lunar(text: str) -> str:
    """``mùng 1`` / ``mồng 1 tháng Giêng`` → chữ."""

    def repl(m: re.Match) -> str:
        return f'{m.group(1)} {number_to_words_vi(int(m.group(2)))}'

    return re.sub(r'\b(mùng|mồng)\s+(\d{1,2})\b', repl, text, flags=re.IGNORECASE)


def _normalize_clock(text: str) -> str:
    """``14h30`` / ``14:30`` / ``7h`` / ``7 giờ 30`` → "mười bốn giờ ba mươi phút".

    Chỉ áp dụng cho giờ hợp lệ 0-23; ``07:05`` đọc "bảy giờ không mươi lăm phút".
    Dấu ``:`` bắt buộc phải có phút (tránh ``Chương 1:`` bị đọc thành "giờ").
    """

    def repl(m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return m.group(0)
        out = f'{number_to_words_vi(hour)} giờ'
        if m.group(2) is not None:
            out += f' {number_to_words_vi(minute)} phút'
        return out

    # `h`/`g` không cần phút; dấu `:` bắt buộc phải có 2 chữ số phút.
    text = re.sub(r'\b(\d{1,2})\s*[hg]\s*(\d{2})?\b', repl, text, flags=re.IGNORECASE)
    text = re.sub(r'(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])', repl, text)
    return text


_FRACTION_DEN_WORDS = {2: 'hai', 3: 'ba', 4: 'tư', 5: 'năm', 6: 'sáu', 8: 'tám', 10: 'mười'}
_FRACTION_RE = re.compile(r'(?<![\d/.,])\b(\d{1,3})/(\d{1,3})\b(?![\d/])')


def _normalize_fraction(text: str) -> str:
    """``3/4`` → "ba phần tư"; ``1/2`` → "một phần hai"."""

    def repl(m: re.Match) -> str:
        num, den = int(m.group(1)), int(m.group(2))
        if den < 2 or num < 1:
            return m.group(0)
        return f'{number_to_words_vi(num)} phần {_FRACTION_DEN_WORDS.get(den, number_to_words_vi(den))}'

    return _FRACTION_RE.sub(repl, text)


def _normalize_number_range(text: str) -> str:
    """``6-7 mét`` / ``10 - 20`` → "sáu đến bảy mét" (giữ nguyên đơn vị đi kèm)."""

    def repl(m: re.Match) -> str:
        lo, hi = int(m.group(1)), int(m.group(2))
        if hi <= lo:
            return m.group(0)
        return f'{number_to_words_vi(lo)} đến {number_to_words_vi(hi)}'

    return re.sub(r'\b(\d{1,4})\s*[-–—]\s*(\d{1,4})\b', repl, text)


def _normalize_digit_code(text: str) -> str:
    """Mã số nhiều chữ số đọc từng chữ: ``007`` → "không không bảy".

    Chỉ áp dụng cho chuỗi >= 3 chữ số có số 0 đứng đầu (mã/mã hiệu), không áp
    dụng cho số thông thường hay năm.
    """

    def repl(m: re.Match) -> str:
        digits = m.group(0)
        return ' '.join(number_to_words_vi(int(d)) for d in digits)

    return re.sub(r'(?<![\d\w.,])0\d{2,}(?!\d)', repl, text)



# ---------------------------------------------------------------------------
# Đơn vị & đo lường
# ---------------------------------------------------------------------------
# 1m76 -> một mét bảy mươi sáu   |   1,76m -> một phẩy bảy mươi sáu mét
_METERS_RE = re.compile(r'\b(\d+)\s*m\s*(\d{1,2})\b', re.IGNORECASE)
_DECIMAL_METERS_RE = re.compile(r'\b(\d+[.,]\d+)\s*m\b', re.IGNORECASE)
_CENTIMETERS_RE = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*(?:cm|centimet)\b', re.IGNORECASE)
_KILOGRAMS_RE = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*(?:kg|kí|kilôgam|kilogram)\b', re.IGNORECASE)
_GRAMS_RE = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*(?:g|gram|gam)\b', re.IGNORECASE)
_KILOMETERS_RE = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*(?:km|kilômét|kilomet)\b', re.IGNORECASE)
_SQUARE_METERS_RE = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*m\s*2\b', re.IGNORECASE)
_CUBIC_METERS_RE = re.compile(r'\b(\d+(?:[.,]\d+)?)\s*m\s*3\b', re.IGNORECASE)
_PERCENT_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*%')
_CURRENCY_RE = re.compile(r'\b(\d[\d.,]*)\s*(?:đ|vnđ|vnd|đồng)\b', re.IGNORECASE)
_CELSIUS_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:°\s*c|độ\s*c)\b', re.IGNORECASE)
_CELSIUS_WORD_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*độ\s+[cCkK]\b')
_SIMPLE_CELSIUS_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*°')
_GHZ_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*GHz\b', re.IGNORECASE)
_MHZ_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*MHz\b', re.IGNORECASE)
_KM_PER_H_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*km\s*/\s*h\b', re.IGNORECASE)
_M_PER_S_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*m\s*/\s*s\b', re.IGNORECASE)
_KWH_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*kWh\b', re.IGNORECASE)
_KM_PER_H_WORD_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*ki-lô-mét\s*/\s*giờ\b', re.IGNORECASE)


def _read_number_literal(literal: str) -> str:
    """Đọc một literal số: ``123``, ``1,5``, ``1.5``, ``1.234`` (dấu phân cách)."""
    literal = literal.strip()
    if not literal:
        return literal
    grouped = re.fullmatch(r'(\d{1,3}(?:[.,]\d{3})+)(?:[.,](\d+))?', literal)
    if grouped:
        int_part = re.sub(r'[.,]', '', grouped.group(1))
        words = number_to_words_vi(int(int_part))
        frac = grouped.group(2)
        if frac:
            return f'{words} phẩy {_read_decimal(frac)}'
        return words
    if re.fullmatch(r'\d+', literal):
        return number_to_words_vi(int(literal))
    decimal = re.fullmatch(r'(\d+)[.,](\d+)', literal)
    if decimal:
        return (
            f'{number_to_words_vi(int(decimal.group(1)))} '
            f'phẩy {_read_decimal(decimal.group(2))}'
        )
    return literal


def _normalize_measurements(text: str) -> str:
    def meter_repl(m: re.Match) -> str:
        metres = int(m.group(1))
        centimetres = int(m.group(2))
        words = f'{number_to_words_vi(metres)} mét'
        if centimetres:
            words += f' {number_to_words_vi(centimetres)}'
        return words

    # Square/cubic metres must run before the generic ``m`` patterns so that
    # "20m2" is not split into "20 mét 2".
    text = _SQUARE_METERS_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} mét vuông', text
    )
    text = _CUBIC_METERS_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} mét khối', text
    )
    text = _METERS_RE.sub(meter_repl, text)
    text = _DECIMAL_METERS_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} mét', text
    )
    # Đơn vị tốc độ / tần số / năng lượng trước các đơn vị "trần" bên dưới.
    text = _KM_PER_H_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} ki-lô-mét trên giờ', text
    )
    text = _KM_PER_H_WORD_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} ki-lô-mét trên giờ', text
    )
    text = _M_PER_S_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} mét trên giây', text
    )
    text = _GHZ_RE.sub(lambda m: f'{_read_number_literal(m.group(1))} giga héc', text)
    text = _MHZ_RE.sub(lambda m: f'{_read_number_literal(m.group(1))} mê-ga héc', text)
    text = _KWH_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} ki-lô-oát giờ', text
    )
    text = _CENTIMETERS_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} xăng-ti-mét', text
    )
    text = _KILOMETERS_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} ki-lô-mét', text
    )
    text = _KILOGRAMS_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} ki-lô-gam', text
    )
    text = _GRAMS_RE.sub(lambda m: f'{_read_number_literal(m.group(1))} gam', text)
    text = _CURRENCY_RE.sub(lambda m: f'{_read_number_literal(m.group(1))} đồng', text)
    text = _PERCENT_RE.sub(lambda m: f'{_read_number_literal(m.group(1))} phần trăm', text)
    text = _CELSIUS_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} độ xê', text
    )
    text = _CELSIUS_WORD_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} độ xê', text
    )
    text = _SIMPLE_CELSIUS_RE.sub(
        lambda m: f'{_read_number_literal(m.group(1))} độ', text
    )
    return text


# ---------------------------------------------------------------------------
# Viết tắt
# ---------------------------------------------------------------------------
_ABBR_MAP = [
    (re.compile(r'\bTP\.?\s*HCM\b', re.IGNORECASE), 'Thành phố Hồ Chí Minh'),
    (re.compile(r'\bTp\.?\s*HCM\b'), 'Thành phố Hồ Chí Minh'),
    # "TP. HCM" / "TP.HCM": xử lý trước mọi luật "TP." để không còn "H" mồ côi.
    (re.compile(r'\bTP\.\s*HCM\b', re.IGNORECASE), 'Thành phố Hồ Chí Minh'),
    (re.compile(r'\bTP\.?\s+CN\b'), 'Thành phố Cần Thơ'),
    (re.compile(r'(?<![\w.])HCM\b'), 'Hồ Chí Minh'),
    (re.compile(r'\bHN\b'), 'Hà Nội'),
    (re.compile(r'\bĐH\b'), 'Đại học'),
    (re.compile(r'\bNXB\b'), 'Nhà xuất bản'),
    (re.compile(r'\bPGS\.?\s*TS\.'), 'Phó giáo sư Tiến sĩ '),
    (re.compile(r'\bGS\.?\s*TS\.'), 'Giáo sư Tiến sĩ '),
    (re.compile(r'\bPGS\.\s'), 'Phó giáo sư '),
    (re.compile(r'\bGS\.\s'), 'Giáo sư '),
    (re.compile(r'\bTS\.\s'), 'Tiến sĩ '),
    (re.compile(r'\bThS\.\s'), 'Thạc sĩ '),
    (re.compile(r'\bBS\.\s'), 'Bác sĩ '),
    # Tổ chức / cơ quan chính thức của Việt Nam (chữ viết tắt đọc liền, không
    # đánh vần từng chữ như engine G2P sẽ làm với một chuỗi in hoa).
    (re.compile(r'\bUBND\b'), 'Uỷ ban nhân dân'),
    (re.compile(r'\bHĐND\b'), 'Hội đồng nhân dân'),
    (re.compile(r'\bTHPT\b'), 'Trung học phổ thông'),
    (re.compile(r'\bTHCS\b'), 'Trung học cơ sở'),
    (re.compile(r'\bDNNN\b'), 'Doanh nghiệp nhà nước'),
]
# Viết tắt tổ chức quốc tế đọc quen thuộc trong tiếng Việt.
_ORG_ABBR_MAP = [
    (re.compile(r'\bUNESCO\b'), 'U N E S C O'),
    (re.compile(r'\bWHO\b'), 'W H O'),
    (re.compile(r'\bCOVID\s*-?\s*19\b', re.IGNORECASE), 'Cô-vít mười chín'),
    (re.compile(r'\bADN\b'), 'A D N'),
    (re.compile(r'\bGDP\b'), 'G D P'),
]
# Viết tắt cần khoảng trắng/ranh giới phía sau để không cắn vào tên riêng.
_ABBR_TAIL = [
    (re.compile(r'\bv\.v\.\.\.'), 'vân vân'),
    (re.compile(r'\bvv\.\b'), 'vân vân'),
    (re.compile(r'\bTP\.\s'), 'Thành phố '),
    (re.compile(r'\bPGS\.\b'), 'Phó giáo sư'),
    (re.compile(r'\bGS\.\b'), 'Giáo sư'),
    (re.compile(r'\bTS\.\b'), 'Tiến sĩ'),
]

# ---------------------------------------------------------------------------
# Địa danh / tên riêng có âm Hán-Việt chuẩn (đọc tự nhiên trong tiếng Việt).
# ---------------------------------------------------------------------------
# LƯU Ý: KHÔNG phiên âm bồi các từ mượn tiếng Anh thông dụng (video, camera,
# online, robot, internet, website, email, data, radio, logic, computer,
# software, hardware, smartphone…). Những từ này được giữ nguyên chữ tiếng Anh
# gốc để engine G2P song ngữ của VieNeu-TTS v3 Turbo tự phát âm chuẩn quốc tế.
# Chỉ giữ lại các địa danh/tên riêng có âm Hán-Việt đã được Việt hoá tự nhiên.
_HAN_VIET_PLACE_MAP = [
    ('Beijing', 'Bắc Kinh'),
    ('New York', 'Niu Óoc'),
    ('London', 'Luân Đôn'),
]
_HAN_VIET_PLACE_RE = [
    (re.compile(rf'(?<![\w-]){re.escape(src)}(?![\w-])', re.IGNORECASE), dst)
    for src, dst in _HAN_VIET_PLACE_MAP
]

# ---------------------------------------------------------------------------
# Số thập phân / phân cách tổng quát
# ---------------------------------------------------------------------------
_PERCENT_DONE = '\x00PCT\x00'

# Giờ: 14h30 -> mười bốn giờ ba mươi phút | 7h -> bảy giờ | 14:30 -> mười bốn giờ ba mươi phút
_TIME_H_RE = re.compile(r'\b(\d{1,2})\s*h\s*(\d{1,2})?\b', re.IGNORECASE)
_TIME_COLON_RE = re.compile(r'\b(\d{1,2}):(\d{2})\b')
# Dải số trang/tuổi: 6-7 mét -> sáu đến bảy mét
_RANGE_RE = re.compile(r'\b(\d+)\s*[-–—]\s*(\d+)\b')


def _normalize_time(text: str) -> str:
    def h_repl(m: re.Match) -> str:
        hour = int(m.group(1))
        body = f'{number_to_words_vi(hour)} giờ'
        if m.group(2):
            minute = int(m.group(2))
            min_words = number_to_words_vi(minute)
            if minute == 0:
                min_words = 'đúng'
                return f'{body} {min_words}'
            if minute % 10 == 0 and minute >= 10:
                min_words = f'{number_to_words_vi(minute // 10)} mươi'
            if minute == 15:
                min_words = 'mười lăm'
            if minute == 30:
                min_words = 'ba mươi'
            body += f' {min_words} phút'
        return body

    def colon_repl(m: re.Match) -> str:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            return m.group(0)
        if minute == 0:
            return f'{number_to_words_vi(hour)} giờ'
        return f'{number_to_words_vi(hour)} giờ {number_to_words_vi(minute)} phút'

    text = _TIME_COLON_RE.sub(colon_repl, text)
    text = _TIME_H_RE.sub(h_repl, text)
    return text


def _normalize_ranges(text: str) -> str:
    """``6-7`` → "sáu đến bảy" (chỉ khi cả hai vế là số thuần)."""

    def repl(m: re.Match) -> str:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            return m.group(0)
        return f'{number_to_words_vi(lo)} đến {number_to_words_vi(hi)}'

    return _RANGE_RE.sub(repl, text)


def _normalize_leading_zeros(text: str) -> str:
    """``007`` → "không không bảy" (đọc từng chữ số cho mã số)."""

    def repl(m: re.Match) -> str:
        return ' '.join(_DIGIT_WORDS[int(d)] for d in m.group(1))

    return re.sub(r'(?<![\w.,])(0\d+)(?![\w.,])', repl, text)


_NUMBER_RE = re.compile(r'(?<![\w])(\d{1,3}(?:[.,]\d{3}){1,3}(?:[.,]\d+)?|\d+[.,]\d+|\d+)')


def _normalize_standalone_numbers(text: str) -> str:
    """Đọc mọi nhóm số còn lại thành chữ (``25`` → "hai mươi lăm")."""

    def repl(m: re.Match) -> str:
        return _read_number_literal(m.group(1))

    return _NUMBER_RE.sub(repl, text)


_VNI_DETECT_RE = re.compile(
    r'(?:Ñ|ñ|ö|Ö|æ|Æ|î|Î|'
    r'[aeouyAEOUY][ùøûõïÙØÛÕÏ]|'
    r'[aeoAEO][âåäÂÅÄ]|'
    r'[aeAE][áàãÁÀÃ]|'
    r'[aA][êéèúüëÊÉÈÚÜË])'
)

_VNI_PAIRS = [
    ('aù', 'á'), ('aø', 'à'), ('aû', 'ả'), ('aõ', 'ã'), ('aï', 'ạ'),
    ('aâ', 'â'), ('aá', 'ấ'), ('aà', 'ầ'), ('aå', 'ẩ'), ('aã', 'ẫ'), ('aä', 'ậ'),
    ('aê', 'ă'), ('aé', 'ắ'), ('aè', 'ằ'), ('aú', 'ẳ'), ('aü', 'ẵ'), ('aë', 'ặ'),
    ('AÙ', 'Á'), ('AØ', 'À'), ('AÛ', 'Ả'), ('AÕ', 'Ã'), ('AÏ', 'Ạ'),
    ('AÂ', 'Â'), ('AÁ', 'Ấ'), ('AÀ', 'Ầ'), ('AÅ', 'Ẩ'), ('AÃ', 'Ẫ'), ('AÄ', 'Ậ'),
    ('AÊ', 'Ă'), ('AÉ', 'Ắ'), ('AÈ', 'Ằ'), ('AÚ', 'Ẳ'), ('AÜ', 'Ẵ'), ('AË', 'Ặ'),
    ('eù', 'é'), ('eø', 'è'), ('eû', 'ẻ'), ('eõ', 'ẽ'), ('eï', 'ẹ'),
    ('eâ', 'ê'), ('eá', 'ế'), ('eà', 'ề'), ('eå', 'ể'), ('eã', 'ễ'), ('eä', 'ệ'),
    ('EÙ', 'É'), ('EØ', 'È'), ('EÛ', 'Ẻ'), ('EÕ', 'Ẽ'), ('EÏ', 'Ẹ'),
    ('EÂ', 'Ê'), ('EÁ', 'Ế'), ('EÀ', 'Ề'), ('EÅ', 'Ể'), ('EÃ', 'Ễ'), ('EÄ', 'Ệ'),
    ('où', 'ó'), ('oø', 'ò'), ('oû', 'ỏ'), ('oõ', 'õ'), ('oï', 'ọ'),
    ('oâ', 'ô'), ('oá', 'ố'), ('oà', 'ồ'), ('oå', 'ổ'), ('oã', 'ỗ'), ('oä', 'ộ'),
    ('ôù', 'ớ'), ('ôø', 'ờ'), ('ôû', 'ở'), ('ôõ', 'ỡ'), ('ôï', 'ợ'),
    ('OÙ', 'Ó'), ('OØ', 'Ò'), ('OÛ', 'Ỏ'), ('OÕ', 'Õ'), ('OÏ', 'Ọ'),
    ('OÂ', 'Ô'), ('OÁ', 'Ố'), ('OÀ', 'Ồ'), ('OÅ', 'Ổ'), ('OÃ', 'Ỗ'), ('OÄ', 'Ộ'),
    ('ÔÙ', 'Ớ'), ('ÔØ', 'Ờ'), ('ÔÛ', 'Ở'), ('ÔÕ', 'Ỡ'), ('ÔÏ', 'Ợ'),
    ('uù', 'ú'), ('uø', 'ù'), ('uû', 'ủ'), ('uõ', 'ũ'), ('uï', 'ụ'),
    ('öù', 'ứ'), ('öø', 'ừ'), ('öû', 'ử'), ('öõ', 'ữ'), ('öï', 'ự'),
    ('UÙ', 'Ú'), ('UØ', 'Ù'), ('UÛ', 'Ủ'), ('UÕ', 'Ũ'), ('UÏ', 'Ụ'),
    ('ÖÙ', 'Ứ'), ('ÖØ', 'Ừ'), ('ÖÛ', 'Ử'), ('ÖÕ', 'Ữ'), ('ÖÏ', 'Ự'),
    ('yù', 'ý'), ('yø', 'ỳ'), ('yû', 'ỷ'), ('yõ', 'ỹ'),
    ('YÙ', 'Ý'), ('YØ', 'Ỳ'), ('YÛ', 'Ỷ'), ('YÕ', 'Ỹ'),
    ('Ñ', 'Đ'), ('ñ', 'đ'), ('Ö', 'Ư'), ('ö', 'ư'), ('æ', 'ỉ'), ('Æ', 'Ỉ'), ('î', 'ỵ'), ('Î', 'Ỵ'),
]

_LOWER_VI_START_RE = re.compile(
    r'^[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]'
)
_SENTENCE_END_RE = re.compile(r'[.!?…:;\"”\')\]»]$')


def convert_vni_to_unicode(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    if not _VNI_DETECT_RE.search(text):
        return text
    for src, dst in _VNI_PAIRS:
        text = text.replace(src, dst)
    return unicodedata.normalize('NFC', text)


_CHAPTER_HEADING_RE = re.compile(
    r'^(chương|phần|quyển|hồi|tập|mục|giới thiệu|lời mở đầu|lời nói đầu|'
    r'phần dẫn|ngoại truyện|vĩ thanh|lời bạt|lời kết|prologue|epilogue|chapter|part|book)\b',
    re.IGNORECASE,
)
_PARA_CHAPTER_RE = re.compile(
    r'^(chương|hồi|quyển)\s+(?:\d+|[ivxlcdm]+)(?:\s*[:.\-–—]\s*.*)?$',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Ký hiệu toán học (fallback khi cache/LLM không phủ)
# ---------------------------------------------------------------------------
_MATH_SYMBOL_WORDS = [
    ('^2', ' bình phương'),
    ('^3', ' lập phương'),
    ('√', ' căn bậc hai '),
    ('∫', ' tích phân '),
    ('∑', ' tổng của '),
    ('α', 'an-pha'),
    ('β', 'bê-ta'),
    ('γ', 'gam-ma'),
    ('δ', 'đen-ta'),
    ('π', 'pi'),
    ('%', ' phần trăm'),
]
_SUPERSCRIPT_RE = re.compile(r'\^(\d+)')

_MATH_SPEECH = {
    '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn', '5': 'năm',
    '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín',
}


def _normalize_math(text: str) -> str:
    """Đọc các ký hiệu toán còn sót: ``x^5`` → "x mũ năm", ``√9`` → "căn bậc hai 9"."""
    if not re.search(r'[\^√∫∑αβγδπ%]', text):
        return text
    for src, dst in _MATH_SYMBOL_WORDS:
        text = text.replace(src, dst)
    text = _SUPERSCRIPT_RE.sub(
        lambda m: ' mũ ' + ' '.join(_MATH_SPEECH.get(d, d) for d in m.group(1)),
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Hàm chuẩn hoá chính
# ---------------------------------------------------------------------------
def normalize_vi_text(text: str) -> str:
    """Chuẩn hoá text tiếng Việt thành chữ thuần cho TTS."""
    # 1. Số La Mã sau danh từ thứ tự (Chương IV, thế kỷ XXI…)
    def _replace_roman(m: re.Match) -> str:
        words = roman_to_words_vi(m.group(2))
        return f'{m.group(1)}{words}' if words else m.group(0)

    text = _ROMAN_AFTER_PREFIX_RE.sub(_replace_roman, text)

    # 2. Đề mục La Mã ở đầu dòng ("I. Mở đầu")
    def _replace_roman_heading(m: re.Match) -> str:
        words = roman_to_words_vi(m.group(1))
        return words if words else m.group(0)

    text = _ROMAN_HEADING_RE.sub(_replace_roman_heading, text)

    # 3. Ngày tháng năm (đầy đủ trước, ngắn sau)
    text = _normalize_standalone_year(text)
    text = _normalize_full_date(text)
    text = _normalize_month_ref(text)
    text = _normalize_lunar(text)
    text = _normalize_ordinal_day(text)
    text = _normalize_short_date(text)

    # 3b. Giờ giấc, dải số, mã số có số 0 đứng đầu
    text = _normalize_time(text)
    text = _normalize_ranges(text)
    text = _normalize_fraction(text)
    text = _normalize_leading_zeros(text)

    # 4. Viết tắt tiếng Việt chính thức (trước khi đọc số để "TP.HCM" không bị
    #    cắt thành "TP.").
    for pattern, repl in _ABBR_MAP:
        text = pattern.sub(repl, text)
    for pattern, repl in _ABBR_TAIL:
        text = pattern.sub(repl, text)
    for pattern, repl in _ORG_ABBR_MAP:
        text = pattern.sub(repl, text)

    # 5. Địa danh / tên riêng có âm Hán-Việt chuẩn (Bắc Kinh, Luân Đôn…).
    #    Từ mượn tiếng Anh thông dụng được giữ nguyên cho engine song ngữ.
    for pattern, repl in _HAN_VIET_PLACE_RE:
        text = pattern.sub(repl, text)

    # 5b. Self-Enriching Hybrid Normalizer: tra cache động, phát hiện công
    #     thức toán/hoá & biểu thức lạ, gọi LLM fallback + auto-enrich. Toàn bộ
    #     tầng này graceful-degrade: thiếu key/offline cũng không crash.
    text = enrich_and_normalize(text)

    # 5c. Ký hiệu toán học còn lại (cache không phủ): số mũ, dấu ^, căn, tích phân.
    text = _normalize_math(text)

    # 6. Giờ giấc, khoảng số, mã số (trước đo lường & số đứng riêng)
    text = _normalize_clock(text)
    text = _normalize_digit_code(text)
    text = _normalize_number_range(text)

    # 7. Đơn vị & đo lường
    text = _normalize_measurements(text)

    # 8. Số còn lại -> chữ
    text = _normalize_standalone_numbers(text)

    # 9. Dọn markdown/XML còn sót để TTS không đọc ký tự lạ
    text = re.sub(r'[*_~#><|]+', ' ', text)
    text = re.sub(r'\[(?:\d+|footnote|chú thích)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_html_to_markdown(
    html: str, default_title: str = '', allow_leading_orphan: bool = False
) -> tuple[str, str]:
    soup = BeautifulSoup(html, 'lxml')

    for tag in soup(['script', 'style', 'sup', 'nav', 'footer', 'header']):
        tag.decompose()

    for a in soup.find_all('a'):
        if re.match(r'^\[\d+\]$', a.get_text(strip=True)):
            a.decompose()
        else:
            a.unwrap()

    body = soup.body or soup
    heading_tags = [
        h.name for h in body.find_all(['h1', 'h2', 'h3']) if h.get_text(strip=True)
    ]
    primary_heading_tag = None
    for tag_name in ('h1', 'h2', 'h3'):
        if tag_name in heading_tags:
            primary_heading_tag = tag_name
            break

    first_title = ''
    blocks: list[str] = []

    for elem in body.find_all(['h1', 'h2', 'h3', 'p', 'hr', 'blockquote', 'li', 'div']):
        if elem.name == 'div' and elem.find(['p', 'h1', 'h2', 'h3', 'blockquote', 'li']):
            continue
        if elem.name == 'hr':
            blocks.append('---')
            continue
        raw = convert_vni_to_unicode(elem.get_text(' ', strip=True))
        if not raw:
            continue
        # Strip inline URLs, footnote brackets, markdown chars
        raw = re.sub(r'https?://\S+', '', raw)
        raw = re.sub(r'\[\d+\]', '', raw)
        raw = re.sub(r'[*_~]+', '', raw)
        raw = re.sub(r'\s+', ' ', raw).strip()
        if not raw:
            continue
        if re.match(r'^(\*\s*){2,}$|^(-{3,})$', raw):
            blocks.append('---')
        elif elem.name in ('h1', 'h2', 'h3'):
            is_chapter_heading = (
                not first_title
                or elem.name == primary_heading_tag
                or bool(_CHAPTER_HEADING_RE.match(raw))
            )
            if is_chapter_heading:
                if not first_title:
                    first_title = raw
                h1_line = f'# {raw}'
                if not blocks or blocks[-1] != h1_line:
                    blocks.append(h1_line)
            else:
                blocks.append(f'## {raw}')
        elif (
            len(raw) <= 80
            and not raw.endswith(('.', '!', '?', ',', ';'))
            and _PARA_CHAPTER_RE.match(raw)
        ):
            if not first_title:
                first_title = raw
            h1_line = f'# {raw}'
            if not blocks or blocks[-1] != h1_line:
                blocks.append(h1_line)
        else:
            if (
                blocks
                and not blocks[-1].startswith('#')
                and blocks[-1] != '---'
                and not _SENTENCE_END_RE.search(blocks[-1])
                and _LOWER_VI_START_RE.match(raw)
            ):
                blocks[-1] = f'{blocks[-1]} {raw}'
            else:
                blocks.append(raw)

    if not blocks and not first_title:
        fallback = convert_vni_to_unicode(
            re.sub(r'\s+', ' ', body.get_text(' ', strip=True)).strip()
        )
        if fallback:
            blocks.append(fallback)

    title = first_title or default_title.strip() or 'Chương không tên'
    if not allow_leading_orphan and (not blocks or not blocks[0].startswith('# ')):
        blocks.insert(0, f'# {title}')

    md_lines: list[str] = []
    for b in blocks:
        md_lines.append(b)
        md_lines.append('')

    return title, '\n'.join(md_lines).strip()


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return [sentence] if sentence else []

    parts: list[str] = []
    clauses = re.split(r'(?<=[,;:])\s+', sentence)
    current = ''
    for clause in clauses:
        if not current:
            current = clause
        elif len(current) + 1 + len(clause) <= max_chars:
            current = f'{current} {clause}'
        else:
            parts.append(current)
            current = clause
    if current:
        parts.append(current)

    final_parts: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            final_parts.append(p)
        else:
            words = p.split(' ')
            buf = ''
            for w in words:
                if not buf:
                    buf = w
                elif len(buf) + 1 + len(w) <= max_chars:
                    buf = f'{buf} {w}'
                else:
                    final_parts.append(buf)
                    buf = w
            if buf:
                final_parts.append(buf)
    return final_parts


def markdown_to_segments(markdown: str, max_chars: int = 500) -> list[AudioSegmentSpec]:
    segments: list[AudioSegmentSpec] = []
    raw_blocks = [b.strip() for b in re.split(r'\n\s*\n', markdown.strip()) if b.strip()]

    for block in raw_blocks:
        if block == '---' or re.match(r'^(-{3,}|\*\s*\*\s*\*)$', block):
            if segments:
                segments[-1].pause_after_ms = max(segments[-1].pause_after_ms, 1500)
            continue

        if block.startswith('# '):
            title_text = normalize_vi_text(block[2:].strip())
            if title_text:
                segments.append(AudioSegmentSpec(text=title_text, pause_after_ms=2000, is_chapter_title=True))
            continue

        if block.startswith('## '):
            sub_text = normalize_vi_text(block[3:].strip())
            if sub_text:
                segments.append(AudioSegmentSpec(text=sub_text, pause_after_ms=1200, is_chapter_title=False))
            continue

        normalized_para = normalize_vi_text(block)
        if not normalized_para:
            continue

        sentences = [s.strip() for s in re.split(r'(?<=[.!?…])\s+', normalized_para) if s.strip()]
        chunked: list[str] = []
        for s in sentences:
            chunked.extend(_split_long_sentence(s, max_chars=max_chars))

        for idx, chunk in enumerate(chunked):
            chunk = chunk.strip()
            if not re.search(r'\w', chunk):
                continue
            is_last_in_para = idx == len(chunked) - 1
            pause = 800 if is_last_in_para else 400
            segments.append(AudioSegmentSpec(text=chunk, pause_after_ms=pause, is_chapter_title=False))

    return segments


# ---------------------------------------------------------------------------
# Uncertainty scanner (UI review + chọn lọc gọi AI)
# ---------------------------------------------------------------------------
@dataclass
class UncertainSpan:
    """Một điểm nhập nhằng: cần người duyệt hoặc gửi AI xử lý có chọn lọc."""

    token: str
    sentence: str
    reason: str  # 'math_equation' | 'chem_formula' | 'mixed_alphanumeric' | 'unknown_abbr' | 'fraction'
    start: int  # offset của token trong ``text`` gốc
    end: int
    suggested_reading: str = ''

    def to_dict(self) -> dict:
        return {
            'token': self.token,
            'sentence': self.sentence,
            'reason': self.reason,
            'start': self.start,
            'end': self.end,
            'suggested_reading': self.suggested_reading,
        }


# Biểu thức có dấu hiệu toán học thực sự (không bắt "=" đơn lẻ trong văn xuôi).
_EQUATION_RE = re.compile(
    r'(?<![\w.])(?:[A-Za-zα-ω][A-Za-z0-9α-ω]*[\s\^0-9+\-*.()]*)?'
    r'=\s*[-+]?[A-Za-z0-9.^]+(?:\s*[-+]\s*[A-Za-z0-9.^]+)*'
)
_EXPONENT_TERM_RE = re.compile(
    r'(?<![\w.^])[A-Za-zα-ω][A-Za-z0-9α-ω]*\^\s*\d+'
    r'(?:\s*[-+]\s*\d*[A-Za-zα-ω][A-Za-z0-9α-ω]*\^?\s*\d*)*'
)
_MATH_SYMBOL_RE = re.compile(r'[√∫∑∏∂±≤≥≠∞]|\b(?:sqrt|int|sum|lim)\b')
# Token chữ + số kiểu F-35 / Zz7 / COVID-19 / H2SO4.
_MIXED_RE = re.compile(r'\b[A-Za-zẠ-ỹ]+(?:[-_]?\d+[A-Za-z]*|[A-Za-z]+[-_]?\d+)\b')
_CHEM_CANDIDATE_RE = re.compile(r'\b(?:[A-Z][a-z]?\d?){2,}\b')
# Ứng viên công thức rộng hơn (bắt cả Fe2, Cl2) — lọc kỹ ở ``_is_known_chem``.
_MIXED_CANDIDATE_RE = re.compile(r'\b[A-Z][A-Za-z]?\d+[A-Za-z0-9]*\b|\b(?:[A-Z][a-z]?\d?){2,}\b')
_FRACTION_RE = re.compile(r'(?<![\d/])(\d{1,4})\s*/\s*(\d{1,4})(?![\d/])(?![/.-]\d)')
# Viết tắt in hoa toàn bộ: WHO, UNESCO, GDP (không bắt chữ hoa đầu câu).
_UPPER_ABBR_RE = re.compile(r'(?<![^\W\d_])([A-ZĐ][A-ZĐ0-9]{1,7})(?![^\W\d_])')

# Các viết tắt đã được pipeline đọc sẵn (không phải điểm nghi vấn).
_KNOWN_ABBREVIATIONS = {
    'UBND', 'HĐND', 'THPT', 'THCS', 'DNNN', 'NXB', 'GS', 'PGS', 'TS', 'THS',
    'BS', 'TP', 'HCM', 'HN', 'ĐH', 'UNESCO', 'WHO', 'GDP', 'ADN', 'COVID',
    'VTV', 'VOV', 'VNA', 'AI', 'GDP', 'CNTT', 'KCN', 'BQL', 'SĐT', 'CMND',
    'SXKD', 'KHCN', 'KTXH', 'XHCN', 'TBCN', 'ĐCS', 'VND', 'USD', 'EUR',
}

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+')

# Ngữ cảnh trên UI phải là MỘT câu gọn: không xuống dòng, không cả đoạn văn.
SENTENCE_CONTEXT_MAX_CHARS = 120


def _one_line(text: str) -> str:
    """Ép về một dòng: mọi loại xuống dòng/khoảng trắng thừa → một dấu cách."""
    return re.sub(r'\s+', ' ', text or '').strip()


def _clip_around(token: str, sentence: str, limit: int = SENTENCE_CONTEXT_MAX_CHARS) -> str:
    """Cắt gọn quanh ``token`` nếu câu quá dài (thêm '…' ở chỗ cắt)."""
    sentence = _one_line(sentence)
    if len(sentence) <= limit:
        return sentence
    pos = sentence.find(token)
    if pos < 0:
        return sentence[: limit - 1].rstrip() + '…'
    half = max(0, (limit - len(token)) // 2)
    start = max(0, pos - half)
    end = min(len(sentence), pos + len(token) + half)
    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(sentence) else ''
    body = sentence[start:end].strip()
    # Dấu '…' cộng vào có thể vượt limit — cắt phần thân, GIỮ cả hai dấu '…'
    # để người đọc vẫn biết ngữ cảnh bị cắt ở hai phía.
    room = limit - len(prefix) - len(suffix)
    if room > 0:
        body = body[:room]
    return f'{prefix}{body}{suffix}'


def _sentence_at(text: str, pos: int, token: str = '') -> str:
    """Đúng MỘT câu chứa offset ``pos``, luôn trên một dòng.

    - Không bao giờ vượt qua ``\\n``: tách đoạn theo xuống dòng trước.
    - Chỉ trả câu chứa token (không gộp nhiều câu / cả đoạn).
    - Câu vẫn dài → cắt gọn ngữ cảnh quanh token.
    """
    if not text:
        return ''
    # 1. Giới hạn trong một dòng chứa token.
    line_start = text.rfind('\n', 0, pos) + 1
    line_end = text.find('\n', pos)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    rel = max(0, min(pos - line_start, len(line)))

    # 2. Tách câu trong dòng đó, chọn câu chứa token.
    bounds = [0] + [m.end() for m in _SENTENCE_SPLIT_RE.finditer(line)]
    start = 0
    for b in bounds:
        if b <= rel:
            start = b
        else:
            break
    end_match = _SENTENCE_SPLIT_RE.search(line, start)
    end = end_match.start() if end_match else len(line)
    sentence = line[start:end]

    return _clip_around(token, sentence)


def _suggest_math(token: str) -> str:
    try:
        return _normalize_math(normalize_vi_text(token))
    except Exception:  # pragma: no cover - phòng thủ
        return token


# ---------------------------------------------------------------------------
# Phát âm từ viết tắt
# ---------------------------------------------------------------------------
# Viết tắt tiếng Anh đã quen trong tiếng Việt — đọc theo âm, không đánh vần.
ABBREVIATION_PRONUNCIATIONS: dict[str, str] = {
    'MC': 'em-xi',
    'VIP': 'víp',
    'PR': 'pi-a',
    'CEO': 'xi-i-ô',
    'KOL': 'kây-ô-en',
    'AI': 'ây-ai',
    'ATM': 'ây-ti-em',
    # DNA đọc là "đi-en-ây" (đánh vần tiếng Anh), KHÔNG phải "đê-en-a".
    'DNA': 'đi-en-ây',
    'DJ': 'đi-giê',
    'TV': 'ti-vi',
    'FB': 'phây-bút',
    'USB': 'u-ét-xpi',
    'GPS': 'giê-pê-ét',
    'SOS': 'ét-o-ét',
    'IQ': 'ai-kiu',
    'EQ': 'i-kiu',
    # OC đọc là "ô-xi", tuyệt đối KHÔNG đọc "o-ca".
    'OC': 'ô-xi',
    'WC': 'vê-xê',
    'SIM': 'xim',
    'WIFI': 'oai-phai',
    'KPI': 'kây-pi-ai',
    'CV': 'xi-vi',
    'IT': 'ai-ti',
    'HR': 'ếch-a',
    'VAT': 'vát',
    # Song ngữ thông dụng khác hay gặp trong sách báo tiếng Việt.
    'ODA': 'ô-đê-a',
    'GDP': 'giê-đê-pê',
    'WHO': 'đắp-biu-ây-chi-ô',
    'UNESCO': 'u-nét-xcô',
    'FIFA': 'phi-pha',
    'NBA': 'en-bi-ây',
    'OK': 'ô-kê',
}

# Bảng chữ cái tiếng Việt dùng khi đánh vần viết tắt lạ (VTV → vê-tê-vê).
VIETNAMESE_LETTER_NAMES: dict[str, str] = {
    'A': 'a',
    'B': 'bê',
    'C': 'xê',
    'D': 'dê',
    'E': 'e',
    'F': 'ép',
    'G': 'giê',
    'H': 'hát',
    'I': 'i',
    'J': 'giây',
    'K': 'ca',
    'L': 'en',
    'M': 'em',
    'N': 'en',
    'O': 'o',
    'P': 'pê',
    'Q': 'quy',
    'R': 'e-rờ',
    'S': 'ét',
    'T': 'tê',
    'U': 'u',
    'V': 'vê',
    'W': 'vê-kép',
    'X': 'ích',
    'Y': 'i-dài',
    'Z': 'dét',
}


def suggest_abbreviation_reading(token: str) -> str:
    """Đề xuất cách đọc cho một từ viết tắt.

    - Ưu tiên ``ABBREVIATION_PRONUNCIATIONS`` (MC → ``em-xi``).
    - Còn lại đánh vần từng chữ theo ``VIETNAMESE_LETTER_NAMES``
      (VTV → ``vê-tê-vê``).
    - Ký tự không phải ``A-Z`` (dấu ``.``, ``&``…) bị bỏ qua.

    >>> suggest_abbreviation_reading('MC')
    'em-xi'
    >>> suggest_abbreviation_reading('VTV')
    'vê-tê-vê'
    """
    key = re.sub(r'[^A-Za-z]', '', token or '').upper()
    if not key:
        return token or ''
    if key in ABBREVIATION_PRONUNCIATIONS:
        return ABBREVIATION_PRONUNCIATIONS[key]
    letters = [VIETNAMESE_LETTER_NAMES.get(ch, ch.lower()) for ch in key]
    return '-'.join(letters)



def _looks_like_acronym(token: str) -> bool:
    """``UNESCO``/``FAO`` — chuỗi in hoa toàn bộ, không có chữ số.

    ``XY3Z9``/``H2SO4`` có chữ số nên vẫn được coi là dãy hoá học.
    """
    return token.isupper() and len(token) >= 2 and not any(ch.isdigit() for ch in token)


# Ký hiệu nguyên tố: ký tự in hoa (O, H, K) hoặc hoa+thường (Mn, Fe, Cl).
# Chỉ để nhận dạng *cấu trúc* công thức — không cần bảng nguyên tố đầy đủ vì
# token lạ đã bị lọc ở nhánh acronym/mixed trước đó.
_ELEMENT_SYMBOL_RE = re.compile(r'[A-Z][a-z]?')


def _is_simple_chem_token(token: str) -> bool:
    """``O2``/``H2``/``Cl2``/``NaCl`` — chỉ gồm ký hiệu nguyên tố + hệ số."""
    if not token or not token[0].isupper():
        return False
    i = 0
    symbols = 0
    while i < len(token):
        match = _ELEMENT_SYMBOL_RE.match(token, i)
        if not match:
            return False
        symbols += 1
        i = match.end()
        while i < len(token) and token[i].isdigit():
            i += 1
    # ≥ 2 ký hiệu (H2O, NaCl) → chắc chắn công thức. Một ký hiệu duy nhất chỉ
    # nhận khi là đơn chất quen thuộc (O2, H2, N2) — tách khỏi "Zz7"/"Xx2".
    if symbols >= 2:
        return True
    return token in {'O2', 'H2', 'N2', 'F2', 'Cl2', 'Br2', 'I2', 'O3'}


def _is_known_chem(token: str) -> bool:
    # Từ điển cache luôn thắng.
    if lookup(token):
        return True
    # Viết tắt in hoa (FAO, UNESCO) không phải công thức hoá học dù trùng khuôn.
    if _looks_like_acronym(token):
        return True
    # Dãy ký tự hoá học hợp lệ và ngắn (H2O, CO2, O2) do rule-based đọc sẵn.
    if len(token) <= 4 and _is_simple_chem_token(token):
        return True
    return False


def detect_uncertain_spans(text: str) -> list[dict]:
    """Quét nhanh các điểm nhập nhằng cần duyệt/AI.

    Trả danh sách ``dict`` (sắp theo vị trí, không trùng token cùng vị trí):
    ``{token, sentence, reason, start, end, suggested_reading}``.
    """
    if not text or not text.strip():
        return []

    spans: list[UncertainSpan] = []
    seen: set[tuple[int, int]] = set()

    def add(token: str, start: int, end: int, reason: str, suggestion: str = '') -> None:
        if (start, end) in seen:
            return
        seen.add((start, end))
        spans.append(
            UncertainSpan(
                token=token,
                sentence=_sentence_at(text, start, token),
                reason=reason,
                start=start,
                end=end,
                suggested_reading=suggestion,
            )
        )

    covered: list[tuple[int, int]] = []
    math_matches = [
        m for pattern in (_EQUATION_RE, _EXPONENT_TERM_RE, _MATH_SYMBOL_RE)
        for m in pattern.finditer(text)
    ]
    math_matches.sort(key=lambda m: (m.start(), -(m.end() - m.start())))
    kept: list[re.Match] = []
    for match in math_matches:
        if any(a <= match.start() and match.end() <= b for a, b in covered):
            continue  # đã nằm trong biểu thức dài hơn
        covered.append((match.start(), match.end()))
        kept.append(match)
    # Gộp mảnh liền kề ("x^2 + 2x" + "= 0") thành một biểu thức duy nhất.
    merged: list[tuple[int, int]] = []
    for match in kept:
        bridge = text[merged[-1][1]:match.start()] if merged else None
        if (
            merged
            and '=' in match.group(0)
            and len(bridge) <= 8
            and re.fullmatch(r'[\s+\-*/^()\d.]*', bridge)
        ):
            merged[-1] = (merged[-1][0], match.end())
        else:
            merged.append((match.start(), match.end()))
    for start, end in merged:
        token = text[start:end].strip()
        add(token, start, end, 'math_equation', _suggest_math(token))

    for match in _FRACTION_RE.finditer(text):
        token = match.group(0)
        add(token, match.start(), match.end(), 'fraction', _suggest_math(token))

    for match in _MIXED_CANDIDATE_RE.finditer(text):
        token = match.group(0)
        if _is_known_chem(token):
            continue
        if _is_simple_chem_token(token):
            add(token, match.start(), match.end(), 'chem_formula', token)

    for match in _MIXED_RE.finditer(text):
        token = match.group(0)
        if not any(ch.isdigit() for ch in token):
            continue
        if lookup(token) or _is_known_chem(token):
            continue  # đã có trong cache hoặc là dãy hoá học (O2, CO2, NaCl)
        add(token, match.start(), match.end(), 'mixed_alphanumeric', token)

    for match in _UPPER_ABBR_RE.finditer(text):
        token = match.group(1)
        # Bỏ qua nếu đây là phần đầu của một token dài hơn (đã xử lý ở trên).
        tail_end = match.end()
        if tail_end < len(text) and (text[tail_end].isalpha() or text[tail_end].isdigit()):
            continue
        sentence_start = _sentence_at(text, match.start(), token)
        if sentence_start and sentence_start[0] == token[0] and text.startswith(token, match.start()):
            # Chữ hoa đầu câu một chữ ("Trời") — không phải viết tắt.
            if token[1:2].islower() or len(token) == 1:
                continue
        if token.upper() in _KNOWN_ABBREVIATIONS or lookup(token):
            continue
        if any(ch.isdigit() for ch in token):
            continue  # đã thuộc mixed_alphanumeric
        if token[1:].islower():
            continue  # "Chương", "Xin" — chữ hoa đầu câu, không phải viết tắt
        add(token, match.start(), match.end(), 'unknown_abbr',
            suggest_abbreviation_reading(token))

    # Bỏ span nằm lồng trong span khác cùng offset (thường là ký hiệu toán nằm
    # trong cả biểu thức đã bắt trọn) — giữ span dài nhất.
    # Dấu ^ nằm ngoài mọi biểu thức đã bắt (vd. "x^2 + 2x + 1 = 0" chỉ khớp
    # phần đầu) vẫn là điểm nghi vấn.
    for match in re.finditer(r'\^', text):
        pos = match.start()
        if any(start <= pos < end for start, end in covered):
            continue
        add('^', pos, pos + 1, 'math_equation', 'mũ')

    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    deduped: list[UncertainSpan] = []
    for span in spans:
        if any(
            other.start <= span.start and span.end <= other.end and other is not span
            for other in deduped
        ):
            continue
        deduped.append(span)
    return [s.to_dict() for s in deduped]


def uncertain_spans_to_review(text: str) -> list[dict]:
    """Bản sao tiện cho UI: chỉ giữ token + câu + lý do (bỏ offset)."""
    return [
        {'token': s['token'], 'sentence': s['sentence'], 'reason': s['reason']}
        for s in detect_uncertain_spans(text)
    ]
