import json
import logging
import os
import re
import tempfile
from pathlib import Path
import gradio as gr
from core.abs_client import ABSClient, load_env_file
from core import llm_normalizer
from core.epub_parser import export_book_markdown, parse_book_markdown, parse_epub
from core.m4b_packager import export_webvtt, package_m4b
from core.models import BookMetadata, Chapter
from core.text_normalizer import detect_uncertain_spans, normalize_vi_text
from core.tts_engine import (
    ChapterAudioResult,
    EdgeTTSEngine,
    TTSEngine,
    VieNeuEngine,
    render_chapter,
)
from core.vieneu_engine import (
    DEFAULT_PRESET_VOICE,
    build_voice_catalogue,
)
from core.voice_harvester import harvest_voice_from_youtube, list_harvested_voices

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('epub2m4b')

DEFAULT_WORKSPACE = Path(__file__).resolve().parent / 'workspace'
DEFAULT_VOICES_ROOT = Path(__file__).resolve().parent / 'voices'

# Mutable at runtime so the UI and tests can point at a different collection root.
VOICES_ROOT = DEFAULT_VOICES_ROOT


def _slugify(text: str) -> str:
    slug = re.sub(r'[^\w\s-]', '', text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r'[-\s]+', '_', slug)
    return slug or 'audiobook'


def _format_hms(ms: int) -> str:
    total_sec = max(0, int(round(ms / 1000.0)))
    h, rem = divmod(total_sec, 3600)
    m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def _chapter_label(ch: Chapter) -> str:
    return f'[{ch.index:02d}] {ch.title}'


def _one_line_snippet(sentence: str, token: str, limit: int = 120) -> str:
    """Một dòng duy nhất, không xuống dòng, tối đa ``limit`` ký tự quanh ``token``."""
    text = re.sub(r'\s+', ' ', sentence or '').strip()
    if len(text) <= limit:
        return text
    pos = text.find(token)
    if pos < 0:
        return text[: limit - 1].rstrip() + '…'
    half = max(0, (limit - len(token)) // 2)
    start = max(0, pos - half)
    end = min(len(text), pos + len(token) + half)
    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(text) else ''
    body = text[start:end].strip()
    # Dấu '…' cộng vào có thể vượt limit — cắt phần thân, GIỮ cả hai dấu '…'
    # để người đọc vẫn biết ngữ cảnh bị cắt ở hai phía.
    room = limit - len(prefix) - len(suffix)
    if room > 0:
        body = body[:room]
    return f'{prefix}{body}{suffix}'


def _abbr_pattern(token: str) -> str:
    """Regex thay thế chính xác một token, không nuốt từ con.

    Token thuần chữ cái (MC, DNA) dùng ranh giới từ để ``MC`` không thay vào
    ``MCSA``. Token có ký tự đặc biệt (``x^2``, ``1/2``, ``H2SO4``) khớp nguyên
    văn vì ranh giới từ không có nghĩa ở đó.
    """
    if re.fullmatch(r'[^\W\d_]+', token, flags=re.UNICODE):
        return rf'(?<![^\W\d_]){re.escape(token)}(?![^\W\d_])'
    return re.escape(token)


def apply_uncertainty_suggestions(
    rows: list[list[str]] | None,
    md_text: str,
    state: dict | None = None,
) -> tuple[dict, str, str]:
    """Áp cách đọc đã sửa trong bảng Điểm nghi vấn vào kịch bản.

    Mỗi hàng: ``[Từ / Ký hiệu, Phân loại, Ngữ cảnh, Đề xuất phát âm]``.
    Hàng có đề xuất rỗng hoặc trùng y token bị bỏ qua.

    Trả ``(state, md_mới, thông_báo)`` để cập nhật cả ``md_editor`` lẫn
    ``state['chapters']``.
    """
    rows = list(rows or [])
    text = md_text or ''

    # Gom đề xuất theo token (giữ thứ tự trong bảng, bỏ trùng/rỗng/không đổi).
    suggestions: dict[str, str] = {}
    for row in rows:
        if not row or len(row) < 4:
            continue
        token = str(row[0] or '').strip()
        suggestion = str(row[3] or '').strip()
        if not token or not suggestion or token == suggestion:
            continue
        suggestions.setdefault(token, suggestion)

    applied: list[str] = []
    total = 0
    for token, suggestion in suggestions.items():
        text, count = re.subn(_abbr_pattern(token), lambda _m, s=suggestion: s, text)
        if count:
            applied.append(token)
            total += count

    if state and 'chapters' in state and total:
        for ch in state['chapters']:
            ch_md = ch.get('markdown', '')
            if not ch_md:
                continue
            new_ch = ch_md
            for token, suggestion in suggestions.items():
                new_ch = re.sub(
                    _abbr_pattern(token), lambda _m, s=suggestion: s, new_ch
                )
            if new_ch != ch_md:
                ch['markdown'] = new_ch
        state['uncertainty_rows'] = extract_uncertainty_rows(state)

    if total:
        detail = ', '.join(f'`{t}`' for t in applied[:8])
        more = f' … (+{len(applied) - 8} từ)' if len(applied) > 8 else ''
        message = (
            f'✅ Đã áp dụng **{total}** chỗ vào kịch bản cho '
            f'{len(applied)} từ/ký hiệu: {detail}{more}.'
        )
    else:
        message = 'ℹ️ Không có thay đổi nào để áp dụng (đề xuất trống hoặc trùng từ gốc).'

    return (state or {}), text, message


def _extract_reading(answer: str, token: str, context: str) -> str:
    """Rút cách đọc của ``token`` từ câu LLM trả về. Rỗng nếu không xác định.

    LLM nhận cả câu và có thể trả:
    * cả câu đã chuẩn hoá → diff ký tự để cô lập đúng đoạn thay của ``token``;
    * gọn đúng cụm phiên âm (``em-xi``) → dùng luôn.
    """
    cleaned = re.sub(r'\s+', ' ', str(answer or '').strip()).strip('"\'` ')
    if not cleaned or not token:
        return ''

    original = re.sub(r'\s+', ' ', str(context or '').strip()) or token
    # Coi là "cả câu đã chuẩn hoá" khi độ dài tương đương câu gốc — lúc đó phải
    # diff để tách riêng cụm phiên âm. Trả gọn ('em-xi') thì ngắn hơn hẳn.
    looks_like_sentence = len(cleaned) >= max(8, len(original) * 0.5)
    if looks_like_sentence:
        return llm_normalizer._guess_spoken(original, cleaned, token) or ''
    return cleaned


def ai_resolve_uncertainty(
    rows: list[list[str]] | None,
    state: dict | None = None,
) -> tuple[list[list[str]], str]:
    """Dùng LLM điền 'Đề xuất phát âm' tối ưu theo ngữ cảnh câu.

    Trả ``(bảng_mới, thông_báo)``. Chưa cấu hình ``LLM_API_KEY`` thì trả nguyên
    bảng kèm cảnh báo (không raise, không làm hỏng UI).
    """
    table = [list(r) for r in (rows or [])]
    if not table:
        return table, 'ℹ️ Bảng đang trống — hãy bấm "Quét lại điểm nghi vấn" trước.'

    if not llm_normalizer.llm_available():
        return table, '⚠️ Chưa cấu hình LLM_API_KEY trong .env'

    resolved = 0
    failed = 0
    for row in table:
        if len(row) < 4:
            continue
        token = str(row[0] or '').strip()
        if not token:
            continue
        context = str(row[2] or '').strip()
        try:
            answer = llm_normalizer._call_llm(context or token)
        except Exception as exc:  # pragma: no cover - phòng thủ
            logger.warning(f'⚠️ [AI RESOLVE] Lỗi gọi LLM cho {token!r}: {exc}')
            answer = None
        if not answer:
            failed += 1
            continue
        reading = _extract_reading(answer, token, context)
        if reading:
            row[3] = reading
            resolved += 1
        else:
            failed += 1

    if resolved:
        msg = f'✅ AI đã điền gợi ý phát âm cho **{resolved}** từ/ký hiệu.'
        if failed:
            msg += f' ({failed} mục không xử lý được.)'
    else:
        msg = '⚠️ AI không trả về gợi ý nào. Kiểm tra lại LLM_API_KEY / LLM_BASE_URL.'
    return table, msg


def extract_uncertainty_rows(state: dict) -> list[list[str]]:
    """Scan chapters in state for ambiguous or tricky tokens (math, chem, fractions, strange abbrs)."""
    if not state or 'chapters' not in state:
        return []
    category_map = {
        'math_equation': 'Biểu thức toán học',
        'chem_formula': 'Ký hiệu hoá học',
        'fraction': 'Phân số nhập nhằng',
        'mixed_alphanumeric': 'Từ ghép chữ & số',
        'unknown_abbr': 'Từ viết tắt lạ',
    }
    seen_tokens = set()
    rows = []
    for ch in state['chapters']:
        if not ch.get('enabled', True):
            continue
        text = ch.get('markdown', '')
        if not text:
            continue
        spans = detect_uncertain_spans(text)
        for sp in spans:
            token = sp['token']
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            cat = category_map.get(sp['reason'], sp['reason'])
            # Bảng UI chỉ hiển thị MỘT câu gọn: ép về một dòng và cắt quanh token.
            snippet = _one_line_snippet(sp.get('sentence', ''), token)
            rows.append([token, cat, snippet, sp['suggested_reading']])
            if len(rows) >= 150:
                break
        if len(rows) >= 150:
            break
    return rows


def _fuzzy_single_match(voices: list[dict], candidate: str) -> dict | None:
    """Substring match, nhưng chỉ trả về khi khớp đúng MỘT giọng.

    "adam" khớp cả "Adam" và "Adam bựa" → trả ``None`` thay vì đoán bừa.
    """
    if not candidate:
        return None
    hits = [
        v for v in voices
        if candidate in v.get('name', '').lower()
        or v.get('name', '').lower() in candidate
    ]
    return hits[0] if len(hits) == 1 else None


def get_voice_showcase_sample(voice_label_or_name: str) -> tuple[str | None, str]:
    """Retrieve pre-rendered Golden Dataset sample (.ogg or .wav) for 0ms delay preview."""
    if not voice_label_or_name:
        return None, ''

    index_file = DEFAULT_WORKSPACE / 'golden_samples' / 'index.json'
    if not index_file.exists():
        return None, ''

    try:
        data = json.loads(index_file.read_text(encoding='utf-8'))
        voices = data.get('voices', [])
        raw = voice_label_or_name.strip().lower()
        # Tên giọng suy ra từ nhãn dropdown. Nhãn có dạng
        # "<section> · <Tên> — ...", mảnh thứ 2 mới là tên → cắt ở dấu '·' ĐẦU TIÊN.
        cand_name = raw.split('·', 1)[1] if '·' in raw else raw
        cand_name = cand_name.split('—', 1)[0].strip()

        # Khớp theo thứ tự chặt dần. Substring CHỈ là phương án cuối vì "Adam"
        # là tiền tố của "Adam bựa" → sẽ chọn nhầm giọng nếu so mờ trước.
        def _lookup(predicate):
            for v in voices:
                if predicate(v):
                    return v
            return None

        matched = (
            _lookup(lambda v: v.get('label', '').strip().lower() == raw)
            or _lookup(lambda v: v.get('id', '').strip().lower() == cand_name.replace(' ', '_'))
            or _lookup(lambda v: v.get('name', '').strip().lower() == cand_name)
            # Fallback mờ: chỉ nhận khi DUY NHẤT một giọng khớp.
            or _fuzzy_single_match(voices, cand_name)
        )

        if matched:
            project_root = Path(__file__).resolve().parent
            ogg_path = Path(matched.get('ogg_path', ''))
            if not ogg_path.is_absolute():
                ogg_path = project_root / ogg_path
            wav_path = Path(matched.get('wav_path', ''))
            if not wav_path.is_absolute():
                wav_path = project_root / wav_path

            audio_file = None
            if ogg_path.exists():
                audio_file = str(ogg_path)
            elif wav_path.exists():
                audio_file = str(wav_path)

            dur = matched.get('duration_sec', 0)
            name = matched.get('name', '')
            badge = f"🎧 **{name}** — Mẫu chuẩn VieNeu: `{dur:.1f}s`"
            return audio_file, badge
    except Exception as exc:
        logger.warning(f'Lỗi nạp audio mẫu: {exc}')

    return None, ''


def handle_test_abs_connection(url: str, token: str) -> str:
    """Test connection to Audiobookshelf server."""
    client = ABSClient(base_url=url, token=token)
    res = client.check_connection()
    if res.get('connected'):
        libs = client.get_libraries()
        lib_names = [lib.get('name', '') for lib in libs if lib.get('name')]
        lib_str = f" | Thư viện: {', '.join(lib_names)}" if lib_names else ''
        return f"✅ **Kết nối thành công!** User: `{res.get('username')}` | Phiên bản: `v{res.get('server_version')}`{lib_str}"
    else:
        return f"❌ **Không thể kết nối Audiobookshelf:** {res.get('error')}"


def handle_epub_upload(epub_file: str | None, work_root: Path = DEFAULT_WORKSPACE):
    if not epub_file:
        return {}, '', '', None, gr.update(choices=[], value=[]), ''

    epub_path = Path(epub_file)
    logger.info(f'📂 [UPLOAD] Nhận file EPUB: {epub_path.name}')
    book_dir = Path(work_root) / _slugify(epub_path.stem)
    meta = parse_epub(epub_path, book_dir)
    logger.info(f'📖 [METADATA] Tên sách: "{meta.title}" | Tác giả: "{meta.author}"')

    md_path = export_book_markdown(meta, book_dir / 'book.md')
    md_text = md_path.read_text(encoding='utf-8')
    logger.info(f'📑 [PARSER] Tách thành công {len(meta.chapters)} chương | Kịch bản lưu tại: {md_path.name}')

    choices = [_chapter_label(c) for c in meta.chapters]
    selected = [_chapter_label(c) for c in meta.chapters if c.enabled]
    logger.info(f'🎯 [SELECT] Mặc định chọn {len(selected)}/{len(choices)} chương (đã lọc mục lục/bản quyền/trang rác)')

    state = {
        'book_dir': str(book_dir),
        'title': meta.title,
        'author': meta.author,
        'cover_path': str(meta.cover_path) if meta.cover_path else None,
        'chapters': [
            {
                'index': c.index,
                'title': c.title,
                'markdown': c.markdown,
                'enabled': c.enabled,
                'label': _chapter_label(c),
            }
            for c in meta.chapters
        ],
    }
    state['uncertainty_rows'] = extract_uncertainty_rows(state)
    cover_val = str(meta.cover_path) if meta.cover_path and meta.cover_path.exists() else None
    return (
        state,
        meta.title,
        meta.author,
        cover_val,
        gr.update(choices=choices, value=selected),
        md_text,
    )


def handle_epub_upload_ui(epub_file: str | None, work_root: Path = DEFAULT_WORKSPACE):
    res = handle_epub_upload(epub_file, work_root)
    state = res[0]
    uncertainty_rows = state.get('uncertainty_rows', []) if isinstance(state, dict) else []
    return (*res, uncertainty_rows)



def handle_harvest_voice(
    voice_id: str,
    query_or_url: str,
    target_minutes: float = 6.0,
    max_videos: int = 1,
):
    """Run the auto harvester and return a summary plus the profiles table.

    Returns ``(summary, rows)`` where ``rows`` is the Gradio ``Dataframe``
    value (one row per harvested voice profile).
    """
    voice_id = (voice_id or '').strip()
    query_or_url = (query_or_url or '').strip()
    if not voice_id or not query_or_url:
        return '⚠️ Cần nhập Voice ID và từ khóa/URL YouTube.', _voice_table_rows()

    logger.info(f'🎙️ [HARVEST_UI] Bắt đầu cào giọng "{voice_id}" từ "{query_or_url}"')
    try:
        result = harvest_voice_from_youtube(
            voice_id=voice_id,
            query_or_url=query_or_url,
            voices_root=Path(VOICES_ROOT),
            target_clean_minutes=float(target_minutes),
            max_videos=int(max_videos),
        )
    except Exception as exc:  # never let a UI action crash the whole Gradio app
        logger.exception('❌ [HARVEST_UI] Cào giọng thất bại')
        return f'❌ Cào giọng thất bại: {exc}', _voice_table_rows()

    summary = (
        f'✅ Đã cào xong "{voice_id}": {len(result.kept_clips)} câu / '
        f'{result.total_clean_sec / 60.0:.1f} phút sạch. '
        f'Giọng này sẵn sàng cho chế độ Instant Clone (chọn "Clone · {voice_id}" trong dropdown giọng đọc).'
    )
    logger.info(f'✅ [HARVEST_UI] {summary}')
    return summary, _voice_table_rows()


def _read_voice_profile(voice_dir: Path) -> dict:
    """Read one voice profile from disk in the shape the UI needs."""
    voice_dir = Path(voice_dir)
    profile_path = voice_dir / 'profile.json'
    golden_wav = ''
    golden_text = ''
    if profile_path.exists():
        try:
            raw = json.loads(profile_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            raw = {}
        golden_rel = raw.get('best_golden_ref_wav')
        if golden_rel:
            candidate = voice_dir / golden_rel
            if candidate.exists():
                golden_wav = str(candidate)
        golden_text = raw.get('best_golden_ref_text') or ''
    if not golden_wav:
        for candidate in sorted((voice_dir / 'golden_refs').glob('*.wav')):
            golden_wav = str(candidate)
            break
    return {'golden_ref_wav': golden_wav, 'golden_ref_text': golden_text}


def _harvested_voice_ids() -> list[str]:
    """Voice IDs available for the harvester dropdown."""
    return [v['voice_id'] for v in list_harvested_voices(Path(VOICES_ROOT))]


def _load_harvested_voices():
    """Populate the harvester table and dropdown when the app loads."""
    return _voice_table_rows(), gr.update(choices=_harvested_voice_ids())


def _refresh_all_voices():
    """Refresh the reading-voice dropdown plus the harvest table/dropdown."""
    rows, harvested_update = _load_harvested_voices()
    return _refresh_vieneu_voice_choices(), rows, harvested_update


def _vieneu_voice_choices() -> list[str]:
    """Dropdown labels for every VieNeu voice: presets + classic + clones.

    Xây từ catalogue động. Máy chưa có model thì chỉ còn Nguyễn Ngọc Ngạn và các
    entry clone (không còn danh sách giọng hardcode); ``refresh_voices_btn`` nạp
    lại khi model sẵn sàng.
    """
    try:
        labels = [o.label for o in build_voice_catalogue()]
    except Exception as exc:  # pragma: no cover - offline
        logger.warning(f'⚠️ Không dựng được catalogue giọng: {exc}')
        labels = ['Nguyễn Ngọc Ngạn (kinh điển)', 'Clone · Vui Vẻ', 'Clone · WAV của bạn']
    return labels


def _refresh_vieneu_voice_choices():
    """Build the full catalogue (may download the model) on explicit refresh."""
    try:
        labels = [o.label for o in build_voice_catalogue()]
        if labels:
            return gr.update(choices=labels)
    except Exception as exc:  # pragma: no cover - offline fallback
        logger.warning(f'⚠️ [VOICES] Không dựng được catalogue đầy đủ ({exc}).')
    return gr.update(choices=_vieneu_voice_choices())


def _voice_table_rows() -> list[list[str]]:
    """Harvested-voice table rows: ``[Voice ID, Số câu, Phút sạch, Golden ref]``."""
    rows: list[list[str]] = []
    for voice in list_harvested_voices(Path(VOICES_ROOT)):
        rows.append(
            [
                voice['voice_id'],
                str(voice['num_clips']),
                f"{voice['total_clean_minutes']:.1f}",
                voice['golden_ref_wav'],
            ]
        )
    return rows


def handle_select_all_chapters(state: dict):
    if not state or 'chapters' not in state:
        return gr.update(value=[]), ''
    all_labels = [ch['label'] for ch in state['chapters']]
    for ch in state['chapters']:
        ch['enabled'] = True
    enabled_mds = [ch['markdown'].strip() for ch in state['chapters'] if ch['markdown'].strip()]
    md_text = '\n\n'.join(enabled_mds) + ('\n' if enabled_mds else '')
    logger.info(f'✅ [SELECT_ALL] Đã tích chọn toàn bộ {len(all_labels)} chương.')
    return gr.update(value=all_labels), md_text


def handle_deselect_all_chapters(state: dict):
    if not state or 'chapters' not in state:
        return gr.update(value=[]), ''
    for ch in state['chapters']:
        ch['enabled'] = False
    logger.info('❌ [DESELECT_ALL] Đã bỏ tích toàn bộ chương.')
    return gr.update(value=[]), ''


def handle_chapter_selection(selected_labels: list[str], state: dict) -> str:
    if not state or 'chapters' not in state:
        return ''
    selected_set = set(selected_labels or [])
    enabled_mds: list[str] = []
    for ch in state['chapters']:
        ch['enabled'] = ch['label'] in selected_set
        if ch['enabled'] and ch['markdown'].strip():
            enabled_mds.append(ch['markdown'].strip())
    logger.info(f'🔘 [SELECTION] Thay đổi lựa chọn: {len(selected_set)}/{len(state["chapters"])} chương được chọn.')
    return '\n\n'.join(enabled_mds) + ('\n' if enabled_mds else '')


def extract_voice_name(label: str) -> str:
    """Tên giọng chuẩn từ một nhãn dropdown (hoặc tên trần).

    Nhãn dropdown có dạng ``"<section> · <Tên> — <giới tính · vùng · phong cách>"``.
    Lấy phần sau dấu ``·`` (dấu đầu tiên), cắt ở ``—`` hoặc ``-``, rồi strip.

    >>> extract_voice_name('Giọng kể chuyện · Thiện Minh — Nam · Bắc · Phong cách kể chuyện')
    'Thiện Minh'
    >>> extract_voice_name('Giọng tuyển chọn ⭐ · Adam bựa — Nam · Bắc')
    'Adam bựa'
    >>> extract_voice_name('Thiện Minh')
    'Thiện Minh'
    """
    raw = (label or '').strip()
    if not raw:
        return ''
    if '·' in raw:
        raw = raw.split('·', 1)[1]
    # Cắt phần metadata sau tên: '—' hoặc ' - '.
    for sep in ('—', '–', ' - '):
        if sep in raw:
            raw = raw.split(sep, 1)[0]
            break
    return raw.strip()


def _is_preset_label(label: str) -> bool:
    """Nhãn dropdown thuộc một giọng preset (không phải clone/classic đặc biệt)."""
    if not label:
        return False
    if label.startswith('Clone ·') or label.startswith('Nguyễn Ngọc Ngạn'):
        return False
    return True


def _create_tts_engine(
    engine_choice: str,
    voice_choice: str,
    rate_pct: int,
    clone_ref_audio: str | None = None,
    clone_ref_text: str = '',
) -> TTSEngine:
    rate_str = f'{int(rate_pct):+d}%'

    if 'Edge-TTS' in engine_choice:
        return EdgeTTSEngine(voice=voice_choice or 'vi-VN-NamMinhNeural', rate=rate_str)

    # Default and preferred engine: VieNeu-TTS v3 Turbo.
    label = (voice_choice or '').strip()
    if label.startswith('Clone · WAV') or label.startswith('Clone · wav'):
        return VieNeuEngine(
            mode='clone',
            ref_audio=clone_ref_audio or None,
            ref_text=clone_ref_text or None,
        )
    if label.startswith('Clone ·'):
        # A named harvested profile / bundled clone (e.g. Vui Vẻ).
        ref_audio, ref_text = _clone_reference_for_label(label)
        return VieNeuEngine(mode='clone', ref_audio=ref_audio, ref_text=ref_text)
    if label.startswith('Nguyễn Ngọc Ngạn'):
        return VieNeuEngine(mode='classic')
    if _is_preset_label(label):
        # Nhãn dropdown mới dạng "Giọng tuyển chọn ⭐ · Thiện Minh — Nam · Bắc · …"
        # → phải trích tên chuẩn, không được truyền nguyên nhãn làm tên giọng.
        return VieNeuEngine(mode='preset', voice=extract_voice_name(label) or DEFAULT_PRESET_VOICE)
    return VieNeuEngine(mode='preset', voice=DEFAULT_PRESET_VOICE)


def _clone_reference_for_label(label: str) -> tuple[Path | None, str | None]:
    """Resolve a ``Clone · <name>`` dropdown label to a reference WAV + text."""
    name = label.split('·', 1)[1].strip() if '·' in label else ''
    key = name.lower().replace(' ', '_')
    for option in build_voice_catalogue():
        if option.mode == 'clone' and option.ref_audio and option.label.split('·', 1)[1].strip().lower() == name.lower():
            return option.ref_audio, option.ref_text
    voice_dir = Path(VOICES_ROOT) / key
    if voice_dir.exists():
        profile = _read_voice_profile(voice_dir)
        ref_wav = profile.get('golden_ref_wav')
        if ref_wav:
            return Path(ref_wav), profile.get('golden_ref_text')
    return None, None


def handle_preview_tts(
    sample_text: str,
    engine_choice: str,
    voice_choice: str,
    rate_pct: int,
    clone_ref_audio: str | None,
    clone_ref_text: str,
) -> str | None:
    cleaned = normalize_vi_text(sample_text or '')
    if not cleaned:
        return None
    logger.info(f'🔊 [PREVIEW] Thử giọng {engine_choice} ({voice_choice}, {rate_pct:+d}%): "{cleaned[:60]}..."')
    try:
        engine = _create_tts_engine(
            engine_choice=engine_choice,
            voice_choice=voice_choice,
            rate_pct=rate_pct,
            clone_ref_audio=clone_ref_audio,
            clone_ref_text=clone_ref_text,
        )
        preview_dir = DEFAULT_WORKSPACE / '_preview'
        preview_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=preview_dir, suffix='.wav', delete=False) as tmp:
            out_wav = Path(tmp.name)
        engine.synthesize_segment(cleaned, out_wav)
        logger.info(f'✔️ [PREVIEW] Hoàn tất đoạn nghe thử: {out_wav}')
        return str(out_wav)
    except Exception as exc:
        logger.exception('❌ [PREVIEW] Lỗi sinh giọng đọc thử')
        raise gr.Error(f'Lỗi sinh giọng đọc thử: {exc}') from exc


def build_timeline_rows(results: list[ChapterAudioResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    offset_ms = 0
    for r in results:
        start_ms = offset_ms
        end_ms = offset_ms + r.duration_ms
        rows.append(
            [
                r.title,
                _format_hms(start_ms),
                _format_hms(end_ms),
                _format_hms(r.duration_ms),
            ]
        )
        offset_ms = end_ms
    return rows


def handle_generate_m4b(
    state: dict,
    book_title: str,
    book_author: str,
    md_text: str,
    engine_choice: str,
    voice_choice: str,
    rate_pct: int,
    clone_ref_audio: str | None,
    clone_ref_text: str,
    force_rerender: bool = False,
    abs_auto_push: bool = False,
    abs_url: str = '',
    abs_token: str = '',
    abs_lib_id: str = '',
    progress=gr.Progress(),
):
    chapters = parse_book_markdown(md_text or '')
    if not chapters:
        logger.error('❌ [ERROR] Kịch bản Markdown đang trống.')
        raise ValueError('Kịch bản Markdown đang trống. Hãy tải file EPUB hoặc nhập nội dung # Chương.')

    book_dir = Path((state or {}).get('book_dir') or (DEFAULT_WORKSPACE / _slugify(book_title)))
    book_dir.mkdir(parents=True, exist_ok=True)
    cover_str = (state or {}).get('cover_path')
    cover_path = Path(cover_str) if cover_str and Path(cover_str).exists() else None

    logger.info('==================================================')
    logger.info(f'🚀 [START] Bắt đầu tạo sách nói: "{book_title}" — Tác giả: "{book_author}"')
    logger.info(f'⚙️ [CONFIG] Engine: {engine_choice} | Giọng: {voice_choice} | Tốc độ: {rate_pct:+d}%')
    logger.info(f'📚 [CHAPTERS] Tổng số chương cần tổng hợp audio: {len(chapters)}')
    logger.info('==================================================')

    engine = _create_tts_engine(
        engine_choice=engine_choice,
        voice_choice=voice_choice,
        rate_pct=rate_pct,
        clone_ref_audio=clone_ref_audio,
        clone_ref_text=clone_ref_text,
    )

    chaps_out_dir = book_dir / 'chapters_audio'
    results: list[ChapterAudioResult] = []
    total = len(chapters)
    for idx, ch in enumerate(chapters, start=1):
        progress((idx - 1) / max(1, total), desc=f'Đang tổng hợp ({idx}/{total}): {ch.title}')
        logger.info(f'⏳ [{idx}/{total}] Đang xử lý: "{ch.title}" ...')
        res = render_chapter(ch, engine, chaps_out_dir, force=bool(force_rerender))
        results.append(res)
        logger.info(f'✔️ [{idx}/{total}] Xong "{ch.title}" ({res.duration_ms / 1000:.1f}s, {len(res.cues)} cues)')

    progress(0.95, desc='Đang đóng gói .M4B & .VTT bằng FFmpeg...')
    logger.info('📦 [PACKAGING] Đang đóng gói .M4B và .VTT bằng FFmpeg...')
    meta = BookMetadata(
        title=book_title.strip() or 'Sách Nói',
        author=book_author.strip() or 'Không rõ tác giả',
        cover_path=cover_path,
        chapters=chapters,
    )
    slug = _slugify(meta.title)
    out_m4b = package_m4b(meta, results, book_dir / f'{slug}.m4b')
    out_vtt = export_webvtt(results, book_dir / f'{slug}.vtt')
    timeline_rows = build_timeline_rows(results)
    total_dur = _format_hms(sum(r.duration_ms for r in results))

    abs_msg = ''
    if abs_auto_push:
        progress(0.98, desc='Đang đồng bộ sang Audiobookshelf...')
        try:
            client = ABSClient(
                base_url=abs_url or None,
                token=abs_token or None,
                library_id=abs_lib_id or None,
            )
            if client.is_configured() or (client.storage_path and client.storage_path.exists()):
                logger.info('📚 [ABS] Bắt đầu đẩy sách sang Audiobookshelf...')
                sync_res = client.sync_audiobook(
                    file_path=out_m4b,
                    author=meta.author,
                    title=meta.title,
                    library_id=abs_lib_id or None,
                    auto_scan=True,
                )
                if sync_res.get('success'):
                    method = sync_res.get('method', 'upload')
                    scanned = 'đã scan' if sync_res.get('scanned') else 'chưa scan'
                    abs_msg = f' | 📚 [Audiobookshelf] Đã chuyển ({method}, {scanned})!'
                    logger.info(f'🎉 [ABS] {abs_msg}')
                else:
                    err = sync_res.get('error', 'Lỗi không xác định')
                    abs_msg = f' | ⚠️ [ABS] Không thể đẩy sách: {err}'
                    logger.warning(abs_msg)
            else:
                abs_msg = ' | ⚠️ [ABS] Bỏ qua đẩy sách (chưa cấu hình URL hoặc Token).'
                logger.info(abs_msg)
        except Exception as exc:
            abs_msg = f' | ⚠️ [ABS Lỗi] {exc}'
            logger.exception(f'❌ [ABS] Ngoại lệ khi đồng bộ: {exc}')

    status = f'Hoàn tất {len(results)} chương — Tổng thời lượng: {total_dur} | File: {out_m4b.name}{abs_msg}'
    logger.info(f'🎉 [SUCCESS] {status}')
    logger.info(f'📁 [OUTPUT] Sách nói M4B: {out_m4b}')
    logger.info(f'📁 [OUTPUT] Phụ đề WebVTT: {out_vtt}')
    logger.info('==================================================')
    progress(1.0, desc='Hoàn tất!')
    return str(out_m4b), str(out_m4b), str(out_vtt), timeline_rows, status



def build_app() -> gr.Blocks:
    with gr.Blocks(title='EPUB to M4B Studio (Vietnamese AI Audiobook)') as demo:
        state = gr.State({})

        gr.Markdown(
            '# 🎧 EPUB to M4B Audiobook Studio\n'
            'Chuyển đổi sách `.epub` Tiếng Việt sang `.m4b` có Chapter Timeline & phụ đề `.vtt` cho **Audiobookshelf**.'
        )

        with gr.Tabs():
            with gr.Tab('📖 Sách nói Studio (.EPUB → .M4B)'):
                # KHU VỰC 1
                with gr.Group():
                    gr.Markdown('### 1. Tải sách `.epub` & Hiệu đính kịch bản Clean Markdown (`.md`)')
                    with gr.Row():
                        with gr.Column(scale=1):
                            epub_input = gr.File(label='Kéo thả file .epub vào đây', file_types=['.epub'])
                            cover_preview = gr.Image(label='Ảnh bìa sách', type='filepath', height=220)
                            book_title = gr.Textbox(label='Tên sách')
                            book_author = gr.Textbox(label='Tác giả')
                            with gr.Row():
                                select_all_btn = gr.Button('✅ Chọn tất cả', size='sm', variant='secondary')
                                deselect_all_btn = gr.Button('❌ Bỏ chọn tất cả', size='sm', variant='secondary')
                            chapter_selector = gr.CheckboxGroup(
                                label='Danh sách chương (Bỏ tích chương Mục lục / Bản quyền)',
                                choices=[],
                                value=[],
                            )
                        with gr.Column(scale=2):
                            md_editor = gr.Textbox(
                                label='Clean Markdown (.md) — # Tên chương | --- Nghỉ chuyển cảnh (1.5s)',
                                lines=24,
                                placeholder='# Chương 1: Khởi đầu\n\nNội dung chương sách...',
                            )

                    with gr.Accordion('🔍 Điểm nghi vấn cần soát duyệt (Uncertainty Review)', open=False):
                        gr.Markdown(
                            '⚡ **VieNeu Normalizer** đã tự động phiên âm số, ngày tháng, tiền tệ chuẩn xác. '
                            'Bảng dưới liệt kê các công thức toán, ký hiệu hoá học, từ viết tắt lạ hoặc phân số nhập nhằng được quét từ sách để bạn soát nhanh:'
                        )
                        uncertainty_table = gr.Dataframe(
                            headers=['Từ / Ký hiệu', 'Phân loại', 'Ngữ cảnh trong câu', 'Đề xuất phát âm'],
                            datatype=['str', 'str', 'str', 'str'],
                            label='Danh sách điểm cần chú ý',
                            # Chỉ cột "Đề xuất phát âm" cho sửa trực tiếp (nhấp đúp).
                            interactive=True,
                            static_columns=[0, 1, 2],
                        )
                        with gr.Row():
                            refresh_uncertainty_btn = gr.Button(
                                '🔄 Quét lại điểm nghi vấn theo các chương đã chọn',
                                size='sm', variant='secondary',
                            )
                            ai_resolve_btn = gr.Button(
                                '🤖 AI gợi ý phát âm (LLM)', size='sm', variant='secondary'
                            )
                            apply_uncertainty_btn = gr.Button(
                                '✅ Áp dụng cách đọc vào kịch bản', size='sm', variant='primary'
                            )
                        apply_uncertainty_status = gr.Markdown()

                # KHU VỰC 2
                with gr.Group():
                    gr.Markdown('### 2. Cấu hình Giọng đọc (TTS Engine) & Nghe thử (Preview)')
                    with gr.Row():
                        engine_choice = gr.Radio(
                            label='Chế độ Engine',
                            choices=[
                                'VieNeu-TTS v3 Turbo (Mặc định — chất lượng cao)',
                                'Edge-TTS (Siêu nhanh, không cần GPU)',
                            ],
                            value='VieNeu-TTS v3 Turbo (Mặc định — chất lượng cao)',
                            scale=1,
                        )
                        rate_slider = gr.Slider(
                            label='Tốc độ đọc (%) — chỉ áp dụng cho Edge-TTS',
                            minimum=-30,
                            maximum=50,
                            step=5,
                            value=0,
                            scale=1,
                        )

                    initial_voice_choices = _vieneu_voice_choices()
                    init_voice = initial_voice_choices[0] if initial_voice_choices else DEFAULT_PRESET_VOICE
                    init_sample_file, init_badge = get_voice_showcase_sample(init_voice)

                    with gr.Row():
                        voice_choice = gr.Dropdown(
                            label='🎙️ Chọn giọng đọc',
                            choices=initial_voice_choices,
                            value=init_voice,
                            interactive=True,
                            scale=3,
                        )
                        refresh_voices_btn = gr.Button('🔄 Làm mới danh sách giọng', scale=1)

                    with gr.Group():
                        with gr.Row():
                            sample_audio_player = gr.Audio(
                                label='🎧 Nghe thử mẫu giọng VieNeu (0ms delay - Golden Dataset 1 phút)',
                                value=init_sample_file,
                                type='filepath',
                                interactive=False,
                                scale=3,
                            )
                            sample_badge = gr.Markdown(init_badge or 'Chọn giọng ở trên để nghe mẫu tức thì.', scale=2)

                    with gr.Group(visible=False) as clone_controls:
                        gr.Markdown(
                            '#### 🧬 Instant Clone — tải lên file WAV mẫu (8–15s)'
                        )
                        with gr.Row():
                            clone_ref_audio = gr.Audio(
                                label='Audio mẫu cho Instant Clone (WAV sạch, 8–15s)',
                                type='filepath',
                                scale=1,
                            )
                            clone_ref_text = gr.Textbox(
                                label='Transcript khớp với Audio mẫu (nên điền để bám giọng sát hơn)',
                                lines=4,
                                scale=2,
                            )

                    with gr.Row():
                        preview_text = gr.Textbox(
                            label='Câu văn đọc thử',
                            value='Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi.',
                            scale=3,
                        )
                        preview_btn = gr.Button('🔊 Đọc thử (Preview)', variant='secondary', scale=1)
                        preview_audio = gr.Audio(label='Âm thanh đọc thử', type='filepath', scale=2)

                # KHU VỰC 3
                with gr.Group():
                    gr.Markdown('### 3. Đóng gói `.M4B` & Bảng Chapter Timeline')
                    with gr.Row():
                        force_checkbox = gr.Checkbox(
                            label='Render lại toàn bộ (bỏ qua cache các chương đã xong)',
                            value=False,
                        )
                        generate_btn = gr.Button('🚀 Tạo Sách Nói (.M4B)', variant='primary')

                    with gr.Accordion('📚 Tích hợp Audiobookshelf (Server Homelab)', open=False):
                        with gr.Row():
                            abs_url_input = gr.Textbox(
                                label='ABS Server URL',
                                value=os.getenv('ABS_URL', 'http://192.168.1.100:13378'),
                                placeholder='http://192.168.1.100:13378',
                                scale=2,
                            )
                            abs_token_input = gr.Textbox(
                                label='ABS API Token',
                                type='password',
                                value=os.getenv('ABS_TOKEN', ''),
                                placeholder='Nhập API Token từ ABS Settings -> Users',
                                scale=2,
                            )
                            abs_lib_input = gr.Textbox(
                                label='Library ID (để trống để tự tìm)',
                                value=os.getenv('ABS_LIBRARY_ID', ''),
                                placeholder='Để trống = tự chọn thư viện sách nói',
                                scale=1,
                            )
                        with gr.Row():
                            abs_auto_push = gr.Checkbox(
                                label='🚀 Tự động đẩy sang Audiobookshelf sau khi đóng gói M4B',
                                value=os.getenv('ABS_AUTO_SCAN', 'true').lower() in ('1', 'true', 'yes'),
                                scale=2,
                            )
                            abs_test_btn = gr.Button('🔌 Kiểm tra kết nối ABS', variant='secondary', scale=1)
                        abs_status_md = gr.Markdown(
                            '💡 *Khi bật tuỳ chọn này, file M4B sau khi render xong trên PC GPU sẽ tự động đẩy sang Audiobookshelf server (qua HTTP hoặc mount) và kích hoạt quét thư viện ngay lập tức.*'
                        )

                    status_box = gr.Textbox(label='Trạng thái xử lý', interactive=False)
                    timeline_table = gr.Dataframe(
                        headers=['Chương', 'Bắt đầu', 'Kết thúc', 'Thời lượng'],
                        datatype=['str', 'str', 'str', 'str'],
                        label='Chapter Timeline (FFMETADATA)',
                        interactive=False,
                    )
                    with gr.Row():
                        m4b_player = gr.Audio(label='Nghe sách nói (.m4b)', type='filepath')
                        m4b_download = gr.File(label='Tải về .M4B (Audiobookshelf)')
                        vtt_download = gr.File(label='Tải về phụ đề .VTT (WebVTT)')

            with gr.Tab('🎙️ Cào & Tuyển chọn Giọng Tự động (Auto Voice Harvester)'):
                with gr.Group():
                    gr.Markdown(
                        '### 🎙️ Cào & Tuyển chọn Giọng Tự động (Auto Voice Harvester)\n'
                        'Tự động tải video YouTube, tách nhạc nền, cắt câu và lọc giọng chủ kênh '
                        'thành dataset **F5-TTS** & **GPT-SoVITS**.'
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            harvest_voice_id = gr.Textbox(
                                label='Voice ID (tên profile, ví dụ: vui_ve)',
                                placeholder='vui_ve',
                            )
                            harvest_query = gr.Textbox(
                                label='Từ khóa tìm kiếm / URL kênh / URL video YouTube',
                                placeholder='kênh Vui Vẻ hoặc https://www.youtube.com/watch?v=...',
                            )
                            with gr.Row():
                                harvest_minutes = gr.Number(
                                    label='Số phút sạch mục tiêu', value=6.0, minimum=1.0, maximum=60.0
                                )
                                harvest_max_videos = gr.Number(
                                    label='Số video tối đa', value=1, minimum=1, maximum=20, precision=0
                                )
                            harvest_btn = gr.Button(
                                '🚀 Bắt đầu Cào & Tuyển chọn Giọng', variant='primary'
                            )
                        with gr.Column(scale=2):
                            harvested_voice_dropdown = gr.Dropdown(
                                label='Profile giọng đã cào (dùng cho Instant Clone)',
                                choices=_harvested_voice_ids(),
                                value='vui_ve' if 'vui_ve' in _harvested_voice_ids() else None,
                                interactive=True,
                            )
                            harvest_status = gr.Textbox(label='Trạng thái cào giọng', interactive=False)
                            harvest_table = gr.Dataframe(
                                headers=['Voice ID', 'Số câu', 'Phút sạch', 'Golden ref'],
                                datatype=['str', 'str', 'str', 'str'],
                                label='Danh sách giọng đã cào',
                                value=_voice_table_rows(),
                                interactive=False,
                            )

        def handle_voice_change(label: str):
            is_clone = 'Clone' in (label or '')
            sample_file, badge = get_voice_showcase_sample(label)
            return gr.update(visible=is_clone), sample_file, badge

        voice_choice.change(
            fn=handle_voice_change,
            inputs=[voice_choice],
            outputs=[clone_controls, sample_audio_player, sample_badge],
        )
        epub_input.change(
            fn=handle_epub_upload_ui,
            inputs=[epub_input],
            outputs=[state, book_title, book_author, cover_preview, chapter_selector, md_editor, uncertainty_table],
        )
        refresh_uncertainty_btn.click(
            fn=extract_uncertainty_rows,
            inputs=[state],
            outputs=[uncertainty_table],
        )
        ai_resolve_btn.click(
            fn=ai_resolve_uncertainty,
            inputs=[uncertainty_table, state],
            outputs=[uncertainty_table, apply_uncertainty_status],
        )
        apply_uncertainty_btn.click(
            fn=apply_uncertainty_suggestions,
            inputs=[uncertainty_table, md_editor, state],
            outputs=[state, md_editor, apply_uncertainty_status],
        )
        abs_test_btn.click(
            fn=handle_test_abs_connection,
            inputs=[abs_url_input, abs_token_input],
            outputs=[abs_status_md],
        )
        harvest_btn.click(
            fn=handle_harvest_voice,
            inputs=[harvest_voice_id, harvest_query, harvest_minutes, harvest_max_videos],
            outputs=[harvest_status, harvest_table],
        )
        refresh_voices_btn.click(
            fn=_refresh_all_voices,
            inputs=None,
            outputs=[voice_choice, harvest_table, harvested_voice_dropdown],
        )
        demo.load(
            fn=_load_harvested_voices,
            inputs=None,
            outputs=[harvest_table, harvested_voice_dropdown],
        )
        chapter_selector.change(
            fn=handle_chapter_selection,
            inputs=[chapter_selector, state],
            outputs=[md_editor],
        )
        select_all_btn.click(
            fn=handle_select_all_chapters,
            inputs=[state],
            outputs=[chapter_selector, md_editor],
        )
        deselect_all_btn.click(
            fn=handle_deselect_all_chapters,
            inputs=[state],
            outputs=[chapter_selector, md_editor],
        )
        preview_btn.click(
            fn=handle_preview_tts,
            inputs=[
                preview_text,
                engine_choice,
                voice_choice,
                rate_slider,
                clone_ref_audio,
                clone_ref_text,
            ],
            outputs=[preview_audio],
        )
        generate_btn.click(
            fn=handle_generate_m4b,
            inputs=[
                state,
                book_title,
                book_author,
                md_editor,
                engine_choice,
                voice_choice,
                rate_slider,
                clone_ref_audio,
                clone_ref_text,
                force_checkbox,
                abs_auto_push,
                abs_url_input,
                abs_token_input,
                abs_lib_input,
            ],
            outputs=[m4b_player, m4b_download, vtt_download, timeline_table, status_box],
        )

    return demo


demo = build_app()


if __name__ == '__main__':
    import os
    import sys

    port = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
    host = os.getenv('GRADIO_SERVER_NAME', '0.0.0.0')

    if '--watch' in sys.argv or '-w' in sys.argv:
        from gradio.cli.commands.reload import main as reload_main

        clean_args = [arg for arg in sys.argv[1:] if arg not in ('--watch', '-w')]
        reload_main(demo_path=Path(__file__), watch_dirs=['core'], app_args=clean_args)
    else:
        demo.launch(server_name=host, server_port=port)

