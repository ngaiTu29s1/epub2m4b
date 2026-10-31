"""Self-Enriching Hybrid Normalizer (Tầng 2–4) cho chuẩn hoá văn bản TTS.

Kiến trúc đa tầng
-----------------
* **Tầng 1 (Rule-based Regex):** nằm ở :mod:`core.text_normalizer` — số, ngày
  tháng, La Mã, đơn vị tiêu chuẩn (0ms).
* **Tầng 2 (Persistent Dynamic Cache):** từ điển động lưu tại
  ``core/data/dictionary_cache.json`` (bản gốc trong repo) và đồng bộ sang
  ``workspace/dictionary_cache.json`` (bản runtime, ghi được). Tra cứu O(1) cho
  mọi thuật ngữ/công thức đã từng học.
* **Tầng 3 (LLM Fallback Scanner):** phát hiện câu chứa biểu thức toán
  (``x^2 + y^2 = z^2``), công thức hoá học (``H2SO4``, ``CO2``), từ viết tắt lạ
  hoặc ngữ cảnh nhập nhằng mà Tầng 1 & 2 chưa xử lý → gọi LLM (chuẩn
  OpenAI-compatible) để chuyển thành lời đọc tiếng Việt tự nhiên.
* **Tầng 4 (Auto-Enrich):** ghi các cặp ``(thuật ngữ -> phiên âm chuẩn)`` do LLM
  sinh ra trở lại Persistent Cache để lần sau chạy với tốc độ 0ms, không tốn token.

Cấu hình LLM qua biến môi trường (OpenAI-compatible, không hard-code):
``LLM_API_KEY``, ``LLM_BASE_URL``, ``LLM_MODEL``. Nếu thiếu key/offline thì
**graceful degradation** — trả về rule-based tốt nhất có thể, tuyệt đối không
crash pipeline tạo sách.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger('epub2m4b.llm_normalizer')

# ``core/data/dictionary_cache.json`` — bản seed trong repo.
_CACHE_SEED_PATH = Path(__file__).resolve().parent / 'data' / 'dictionary_cache.json'
# ``workspace/dictionary_cache.json`` — bản runtime, ghi được (đồng bộ).
_RUNTIME_CACHE_PATH = Path(__file__).resolve().parent.parent / 'workspace' / 'dictionary_cache.json'

_DEFAULT_BASE_URL = 'https://api.openai.com/v1'
_DEFAULT_MODEL = 'gpt-4o-mini'

_lock = threading.RLock()

# Ký tự không còn được phép lọt vào TTS: dấu mũ, căn, tích phân, chỉ số trên/dưới.
_SCRIPT_RE = re.compile(r'[\^√∫₀-₉⁰-⁹⁺⁻⁼⁽⁾]')
# Biểu thức toán dạng phương trình.
_EQUATION_RE = re.compile(r'[^\s,;:.!?]\s*[=≤≥≈]\s*[^\s,;:.!?]')
# Công thức hoá học đơn giản: chuỗi nguyên tố + chữ số (H2SO4, CO2, NaCl…).
_CHEM_RE = re.compile(r'\b(?:[A-Z][a-z]?\d?){2,}\b')
# Phân số kiểu "3/4" (tách khỏi 30/04/1975 — ngày tháng đã chạy trước).
_FRACTION_RE = re.compile(r'(?<![\d/])(\d{1,4})\s*/\s*(\d{1,4})(?![\d/])')
# "mc" trong E=mc^2 nghĩa là m nhân c.
_IMPLICIT_MUL_RE = re.compile(r'(?<=[A-Za-z])(?=[A-Za-z]\b)')


# ---------------------------------------------------------------------------
# Phát hiện biểu thức "hóc búa" (Tầng 3 trigger)
# ---------------------------------------------------------------------------
def _looks_like_chem_token(tok: str) -> bool:
    """``H2SO4``/``CO2``/``NaCl`` → True; ``NASA``/``OK``/``Vietnam`` → False."""
    if not re.fullmatch(r'(?:[A-Z][a-z]?\d?)+', tok):
        return False
    if tok != tok.lower() and any(ch.islower() for ch in tok[1:]) and not re.search(r'\d', tok):
        # ``Vietnam``/``Microsoft``: chữ thường sau ký tự đầu mà không có chỉ số.
        return False
    elements = re.findall(r'[A-Z][a-z]?', tok)
    has_digits = bool(re.search(r'\d', tok))
    # Có chỉ số hoặc >= 3 nguyên tố (đủ đặc trưng phân tử) mới coi là hoá học.
    return has_digits or len(elements) >= 3


def _needs_llm(text: str) -> bool:
    """Câu có chứa công thức/biểu thức mà cache + rule chưa chắc xử lý hết?"""
    if _SCRIPT_RE.search(text) or _EQUATION_RE.search(text):
        return True
    return any(_looks_like_chem_token(t) for t in _CHEM_RE.findall(text))


# ---------------------------------------------------------------------------
# Persistent Dynamic Cache (Tầng 2 & Tầng 4)
# ---------------------------------------------------------------------------
def _normalize_key(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip()).lower()


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _iter_raw_entries(raw: dict):
    """Chấp nhận cả ``{"entries": {...}}`` lẫn mapping phẳng seed."""
    entries = raw.get('entries') if isinstance(raw.get('entries'), dict) else raw
    for key, value in entries.items():
        if key.startswith('_'):
            continue
        if isinstance(value, str):
            yield key, {'spoken': value, 'aliases': [], 'kind': 'seed'}
        elif isinstance(value, dict) and value.get('spoken'):
            yield key, value


def _load_entries() -> dict[str, dict]:
    """Merge bản seed (repo) rồi bản runtime (workspace) — runtime thắng."""
    merged: dict[str, dict] = {}
    for source in (_read_json(_CACHE_SEED_PATH), _read_json(_RUNTIME_CACHE_PATH)):
        for key, value in _iter_raw_entries(source):
            entry = dict(value)
            entry.setdefault('literal', key)
            merged[_normalize_key(key)] = entry
    return merged


# Cache trong RAM; nạp lười ở lần tra cứu đầu tiên.
_memory_cache: dict[str, dict] | None = None


def _get_cache() -> dict[str, dict]:
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = _load_entries()
    return _memory_cache


def reload_cache() -> dict[str, dict]:
    """Buộc nạp lại từ đĩa (dùng trong test/benchmark)."""
    global _memory_cache
    with _lock:
        _memory_cache = _load_entries()
    return _memory_cache


def lookup(term: str) -> str | None:
    """Tra Persistent Cache. Trả phiên âm tiếng Việt hoặc ``None``."""
    key = _normalize_key(term)
    entry = _get_cache().get(key)
    if entry:
        return entry.get('spoken')
    # Tra tất cả alias phụ (axit sunfuric → H2SO4).
    for value in _get_cache().values():
        for alias in value.get('aliases', []) or []:
            if _normalize_key(alias) == key:
                return value.get('spoken')
    return None


def add_entry(
    term: str,
    spoken: str,
    *,
    kind: str = 'learned',
    aliases: list[str] | None = None,
    persist: bool = True,
) -> None:
    """Tầng 4: enrich cache (RAM + đĩa runtime) với cặp mới."""
    term, spoken = term.strip(), spoken.strip()
    if not term or not spoken:
        return
    with _lock:
        cache = _get_cache()
        key = _normalize_key(term)
        if cache.get(key, {}).get('spoken') == spoken:
            return
        cache[key] = {
            'spoken': spoken,
            'aliases': list(aliases or []),
            'kind': kind,
            'literal': term,
        }
        if persist:
            _persist()


def _persist() -> None:
    """Ghi cache RAM ra ``workspace/dictionary_cache.json`` (runtime, ghi được)."""
    try:
        _RUNTIME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        entries = {
            key: {
                'spoken': entry['spoken'],
                'aliases': entry.get('aliases', []),
                'kind': entry.get('kind', 'learned'),
                'literal': entry.get('literal', key),
            }
            for key, entry in _get_cache().items()
        }
        payload = {
            '_meta': {'version': 1, 'description': 'Runtime dynamic cache (auto-enriched).'},
            'entries': entries,
        }
        tmp = _RUNTIME_CACHE_PATH.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(_RUNTIME_CACHE_PATH)
    except OSError as exc:  # pragma: no cover - chỉ hỏng khi FS read-only
        logger.warning(f'⚠️ [LLM-Norm] Không ghi được runtime cache ({exc}).')


# ---------------------------------------------------------------------------
# Tầng 1.5 — Rule-based cho công thức (không cần LLM)
# ---------------------------------------------------------------------------
_NUM_WORD = {
    '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn', '5': 'năm',
    '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín',
}


def _chem_to_speech(token: str) -> str:
    """``H2SO4`` → "H hai S O bốn" (đọc nguyên tố + chỉ số dưới)."""
    parts = re.findall(r'([A-Z][a-z]?)(\d*)', token)
    out: list[str] = []
    for symbol, digits in parts:
        if not symbol:
            continue
        out.append(symbol)
        if digits:
            out.append(_NUM_WORD.get(digits[0], digits))
    return ' '.join(out)


def _math_token_to_speech(token: str) -> str:
    """Chuẩn hoá đơn giản một biểu thức: ``^2`` → " bình phương", ``^3`` → " lập phương"."""
    token = token.replace('^2', ' bình phương').replace('^3', ' lập phương')
    token = re.sub(r'\^(\d+)', lambda m: f' mũ {_NUM_WORD.get(m.group(1)[0], m.group(1))}', token)
    token = token.replace('√', ' căn bậc hai ').replace('∫', ' tích phân ')
    return token


# ---------------------------------------------------------------------------
# Tầng 3 — LLM Fallback (OpenAI-compatible)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    'Bạn là bộ chuẩn hoá văn bản cho TTS tiếng Việt (VieNeu-TTS v3 Turbo). '
    'Nhận một câu, trả về DUY NHẤT câu đó đã viết thành lời đọc tiếng Việt tự nhiên.\n'
    '\n'
    'QUY TẮC BẮT BUỘC:\n'
    '\n'
    '1) Viết tắt tiếng Anh thông dụng — LUÔN đọc thành tiếng Việt, TUYỆT ĐỐI không '
    'đánh vần thô theo mặt chữ:\n'
    "   - 'OK' -> 'ô-kê' (hoặc 'okay'). TUYỆT ĐỐI KHÔNG đọc là 'o-ca'.\n"
    "   - 'DNA' -> 'đi-en-ây' (hoặc giữ nguyên 'DNA' cho engine song ngữ). "
    "TUYỆT ĐỐI KHÔNG đọc là 'đê-en-a'.\n"
    "   - 'RNA' -> 'a-ren-ây'.\n"
    "   - 'MC' -> 'em-xi', 'CEO' -> 'xi-i-ô', 'VIP' -> 'víp', 'PR' -> 'pi-a', "
    "'KOL' -> 'kây-ô-en', 'ATM' -> 'ây-ti-em', 'GPS' -> 'giê-pê-ét', "
    "'SOS' -> 'ét-o-ét', 'WIFI' -> 'oai-phai'.\n"
    '\n'
    '2) GIỮ NGUYÊN các từ tiếng Anh mượn thông dụng (video, camera, online, robot, '
    'internet, AI, machine learning, live, stream, show, fan…). Engine là '
    'multilingual song ngữ nên tự phát âm chuẩn quốc tế — không phiên âm Việt hoá.\n'
    '\n'
    '3) Biểu thức toán học / hoá học — chuyển thành lời đọc tiếng Việt:\n'
    "   'H2SO4' -> 'H hai S O bốn', 'x^2' -> 'x bình phương', '1/2' -> 'một phần hai'.\n"
    '\n'
    '4) Định dạng đầu ra: DUY NHẤT văn bản kết quả. Không giải thích, không '
    'markdown, không xuống dòng, không thêm nhãn hay ngoặc kép.'
)


def _llm_config() -> tuple[str | None, str, str]:
    api_key = os.environ.get('LLM_API_KEY', '').strip() or None
    base_url = os.environ.get('LLM_BASE_URL', '').strip() or _DEFAULT_BASE_URL
    model = os.environ.get('LLM_MODEL', '').strip() or _DEFAULT_MODEL
    return api_key, base_url.rstrip('/'), model


def llm_available() -> bool:
    return _llm_config()[0] is not None


def _call_llm(text: str, *, timeout: float = 30.0) -> str | None:
    """Gọi LLM OpenAI-compatible. Trả None nếu thiếu key / lỗi (không raise)."""
    api_key, base_url, model = _llm_config()
    if not api_key:
        return None
    body = json.dumps({
        'model': model,
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': text},
        ],
    }).encode('utf-8')
    request = urllib.request.Request(
        f'{base_url}/chat/completions',
        data=body,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        content = payload['choices'][0]['message']['content']
        return re.sub(r'\s+', ' ', content).strip() or None
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, OSError, TimeoutError) as exc:
        logger.warning(f'⚠️ [LLM-Norm] Gọi LLM thất bại ({type(exc).__name__}: {exc}); dùng rule-based.')
        return None


# Token "lạ" cần LLM học: viết tắt in hoa (UNESCO, WHO) hoặc token có ký số/hoa lẫn.
_LEARNABLE_RE = re.compile(r'\b[A-Za-z][A-Za-z0-9]{1,9}\b')
_STOPWORDS = {
    'the', 'and', 'for', 'with', 'this', 'that', 'when', 'then', 'from',
    'bang', 'bình', 'phương', 'mũ', 'nhân', 'cộng', 'trừ', 'chia', 'phần',
    'trăm', 'bằng', 'rất', 'của', 'và', 'là', 'có', 'không', 'một', 'hai',
}


def _looks_learnable(token: str) -> bool:
    """``H2SO4``/``Zz7``/``UNESCO`` → True; ``video``/``một`` → False."""
    if token.lower() in _STOPWORDS or len(token) < 2:
        return False
    if _looks_like_chem_token(token):
        return True
    has_digit = any(ch.isdigit() for ch in token)
    has_upper = any(ch.isupper() for ch in token)
    # Hỗn hợp chữ + số (Zz7, COVID19) hoặc viết tắt in hoa (substring của LLM).
    return has_digit and has_upper


def _extract_learned_pairs(original: str, rewritten: str) -> list[tuple[str, str]]:
    """Rút các cặp (thuật ngữ → phiên âm) để enrich sau khi LLM trả lời."""
    pairs: list[tuple[str, str]] = []
    # Công thức hoá học: phiên âm rule-based luôn xác định.
    pairs.extend(
        (tok, _chem_to_speech(tok))
        for tok in _CHEM_RE.findall(original)
        if _looks_like_chem_token(tok)
    )
    # Token lạ còn lại trong bản gốc: tìm phiên âm nó xuất hiện trong câu LLM.
    known = set(_known_terms())
    for term in dict.fromkeys(_LEARNABLE_RE.findall(original)):
        if term in known or not _looks_learnable(term):
            continue
        pattern = re.compile(rf'(?<![\w-]){re.escape(term)}', re.IGNORECASE)
        if pattern.search(rewritten):
            continue  # LLM giữ nguyên token → không học được gì
        spoken = _guess_spoken(original, rewritten, term)
        if spoken:
            pairs.append((term, spoken))
    return pairs


def _known_terms() -> list[str]:
    """Toàn bộ surface form đã có trong cache (kể cả alias)."""
    out: list[str] = []
    for entry in _get_cache().values():
        if entry.get('literal'):
            out.append(entry['literal'])
        out.extend(entry.get('aliases') or [])
    return out


def _guess_spoken(original: str, rewritten: str, term: str) -> str | None:
    """Lấy cụm phiên âm thay thế ``term`` bằng diff mức ký tự.

    Diff mức từ sẽ gộp ``Zz7 = 9`` thành một khối và không tách được thuật ngữ; so
    ký tự cho phép cô lập đúng đoạn thay thế của riêng ``term``.
    """
    import difflib

    orig_chars = list(original)
    new_chars = list(rewritten)
    matcher = difflib.SequenceMatcher(a=orig_chars, b=new_chars, autojunk=False)
    term_start = original.find(term)
    if term_start < 0:
        return None
    term_end = term_start + len(term)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        # Khối cần nằm trọn trong thuật ngữ (không lấn sang chữ/ngữ cảnh xung quanh).
        if i1 >= term_start and i2 <= term_end:
            spoken = ''.join(new_chars[j1:j2]).strip()
            if not spoken:
                return None
            if difflib.SequenceMatcher(a=term, b=spoken, autojunk=False).ratio() > 0.6:
                return None
            return spoken
    return None
_MAX_LLM_CALLS = 3


def _replace_dictionary_entries(text: str, cache: dict[str, dict]) -> str:
    """Tầng 2: thay thuật ngữ đã có trong cache (key dài trước)."""
    for entry in sorted(cache.values(), key=lambda e: len(e.get('literal') or ''), reverse=True):
        spoken = entry.get('spoken')
        term = entry.get('literal') or ''
        if not spoken or not term:
            continue
        if not _SAFE_CACHE_TERM_RE.fullmatch(term):
            # Khoá được persist ở dạng lowercase (h2so4) hoặc có dấu "." trong
            # viết tắt (tp.hcm): so khớp phân biệt hoa/thường để không phá chữ.
            pattern = re.compile(rf'(?<!\w){re.escape(term)}(?!\w)')
        elif re.search(r'[A-Za-z0-9]', term):
            pattern = re.compile(rf'(?<![\w]){re.escape(term)}(?![\w])', re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(term))
        text = pattern.sub(lambda _m, r=spoken: r, text)
    return text


# Khoá an toàn để thay không phân biệt hoa/thường: không chứa dấu "." và không có
# chữ in hoa tự nhập — tránh "tp.hcm" viết thường đè lên "TP.HCM" trong câu.
_SAFE_CACHE_TERM_RE = re.compile(r"[a-z0-9][^.!?]*|[^.!?\w][^.!?]*")


def _rule_based_formulas(text: str) -> str:
    """Fallback 0-token cho công thức hoá học + số mũ."""
    text = _CHEM_RE.sub(
        lambda m: _chem_to_speech(m.group(0)) if _looks_like_chem_token(m.group(0)) else m.group(0),
        text,
    )
    text = re.sub(r'\^(\d+)', lambda m: f' mũ {_NUM_WORD.get(m.group(1)[0], m.group(1))}', text)
    text = text.replace('√', ' căn bậc hai ').replace('∫', ' tích phân ')
    return text


_OPERATOR_WORDS = [
    ('=', ' bằng '),
    ('+', ' cộng '),
    ('×', ' nhân '),
    ('≤', ' nhỏ hơn hoặc bằng '),
    ('≥', ' lớn hơn hoặc bằng '),
    ('≈', ' xấp xỉ '),
    ('-', ' trừ '),
]


def _normalize_operators(text: str) -> str:
    """Đọc nốt các toán tử còn sót (``=``, ``+``) để TTS không đọc ký hiệu thô.

    Chỉ chạy ở cuối pipeline: sau khi đã thử cache, rule và (tuỳ chọn) LLM —
    bảo đảm đầu ra không bao giờ còn ký tự toán học dù LLM có trả gì.
    """
    if not re.search(r'[=+×≤≥≈]', text):
        return text
    for src, dst in _OPERATOR_WORDS:
        if src == '-':
            # Dấu trừ chỉ đọc khi đứng tách rời (arithmetic), không đụng gạch nối từ.
            text = re.sub(r'\s-\s', dst, text)
            continue
        text = text.replace(src, dst)
    return re.sub(r'\s+', ' ', text).strip()


def _enrich_impl(text: str) -> str:
    cache = _get_cache()
    # Tầng 2: thay các thuật ngữ đã biết bằng phiên âm cache.
    text = _replace_dictionary_entries(text, cache)

    # Tầng 3: còn biểu thức lạ chưa xử lý không?
    if not _needs_llm(text):
        return text

    # Rule-based 0-token trước (hoá học, số mũ).
    text = _rule_based_formulas(text)
    if not _needs_llm(text):
        return text

    if llm_available():
        for _ in range(_MAX_LLM_CALLS):
            # Các token lạ còn lại. Đã có trong cache (vừa học ở lượt trước) thì
            # không coi là "chưa xử lý" nữa — tránh gọi LLM lặp vô ích.
            unlearned = [
                tok for tok in _LEARNABLE_RE.findall(text)
                if _looks_learnable(tok) and lookup(tok) is None
            ]
            if not unlearned or not _needs_llm(text):
                break
            rewritten = _call_llm(text)
            if not rewritten or _normalize_key(rewritten) == _normalize_key(text):
                # Không tiến triển → dừng, tránh gọi lặp vô ích (tốn token).
                break
            for term, spoken in _extract_learned_pairs(text, rewritten):
                add_entry(term, spoken)
            text = _replace_dictionary_entries(rewritten, _get_cache())

    # Dọn ký tự toán còn sót để TTS không đọc ký tự lạ.
    return _normalize_operators(re.sub(r'\s+', ' ', text).strip())


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------
def enrich_and_normalize(text: str) -> str:
    """Hợp nhất Tầng 2 (cache) + Tầng 3 (LLM) + Tầng 4 (auto-enrich).

    An toàn tuyệt đối: mọi lỗi đều nuốt và trả về văn bản rule-based.
    """
    if not text or not text.strip():
        return text
    try:
        return _enrich_impl(text)
    except Exception as exc:  # pragma: no cover - phòng thủ cuối cùng
        logger.warning(f'⚠️ [LLM-Norm] enrich_and_normalize lỗi ({type(exc).__name__}: {exc}); bỏ qua.')
        return text


__all__ = [
    'add_entry',
    'enrich_and_normalize',
    'llm_available',
    'lookup',
    'reload_cache',
]
