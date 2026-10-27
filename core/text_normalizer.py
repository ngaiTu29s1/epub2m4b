import re
import unicodedata
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from core.models import AudioSegmentSpec

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

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

_ROMAN_MAP = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
    'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
    'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
    'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
}

_ABBR_MAP = {
    r'\bTP\.?\s*HCM\b': 'Thành phố Hồ Chí Minh',
    r'\bHN\b': 'Hà Nội',
    r'\bNXB\b': 'Nhà xuất bản',
    r'\bTS\.\b': 'Tiến sĩ ',
    r'\bGS\.\b': 'Giáo sư ',
    r'\bPGS\.\b': 'Phó giáo sư ',
}


def normalize_vi_text(text: str) -> str:
    # Convert Chương/Phần + Roman numerals -> Arabic numbers
    def _replace_roman(match: re.Match) -> str:
        prefix = match.group(1)
        roman = match.group(2).upper()
        val = _ROMAN_MAP.get(roman)
        return f'{prefix} {val}' if val else match.group(0)

    text = re.sub(
        r'\b(Chương|Phần|Quyển|Tập|Hồi|Mục)\s+([IVXLCDM]+)\b',
        _replace_roman,
        text,
        flags=re.IGNORECASE,
    )

    # Convert dd/mm/yyyy -> d tháng m năm yyyy
    def _replace_full_date(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f'{d} tháng {mo} năm {y}'

    text = re.sub(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b', _replace_full_date, text)

    # Convert dd/mm -> d tháng m
    def _replace_short_date(m: re.Match) -> str:
        d, mo = int(m.group(1)), int(m.group(2))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f'{d} tháng {mo}'
        return m.group(0)

    text = re.sub(r'\b(\d{1,2})/(\d{1,2})\b', _replace_short_date, text)

    for pattern, repl in _ABBR_MAP.items():
        text = re.sub(pattern, repl, text)

    # Strip remaining markdown inline characters so TTS never reads them aloud
    text = re.sub(r'[*_~#>]+', ' ', text)
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


def markdown_to_segments(markdown: str, max_chars: int = 280) -> list[AudioSegmentSpec]:
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
            is_last_in_para = idx == len(chunked) - 1
            pause = 800 if is_last_in_para else 400
            segments.append(AudioSegmentSpec(text=chunk, pause_after_ms=pause, is_chapter_title=False))

    return segments
