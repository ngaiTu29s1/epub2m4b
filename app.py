import json
import logging
import re
import tempfile
from pathlib import Path
import gradio as gr
from core.epub_parser import export_book_markdown, parse_book_markdown, parse_epub
from core.m4b_packager import export_webvtt, package_m4b
from core.models import BookMetadata, Chapter
from core.text_normalizer import normalize_vi_text
from core.tts_engine import (
    ChapterAudioResult,
    EdgeTTSEngine,
    F5TTSEngine,
    TTSEngine,
    ViterboxTTSEngine,
    render_chapter,
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
    cover_val = str(meta.cover_path) if meta.cover_path and meta.cover_path.exists() else None
    return (
        state,
        meta.title,
        meta.author,
        cover_val,
        gr.update(choices=choices, value=selected),
        md_text,
    )


def handle_harvest_voice(
    voice_id: str,
    query_or_url: str,
    target_minutes: float = 6.0,
    max_videos: int = 1,
):
    """Run the auto harvester and return a summary, the golden ref, and table rows.

    Returns ``(summary, golden_ref_wav, golden_ref_text, rows)`` where ``rows``
    is the Gradio ``Dataframe`` value (one row per harvested voice profile).
    """
    voice_id = (voice_id or '').strip()
    query_or_url = (query_or_url or '').strip()
    if not voice_id or not query_or_url:
        return '⚠️ Cần nhập Voice ID và từ khóa/URL YouTube.', None, '', _voice_table_rows()

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
        return f'❌ Cào giọng thất bại: {exc}', None, '', _voice_table_rows()

    profile = _read_voice_profile(result.voice_dir)
    golden_wav = profile['golden_ref_wav']
    summary = (
        f'✅ Đã cào xong "{voice_id}": {len(result.kept_clips)} câu / '
        f'{result.total_clean_sec / 60.0:.1f} phút sạch. '
        f'Đã xuất `metadata.csv` (F5-TTS) và `gpt_sovits.list` (GPT-SoVITS).'
    )
    logger.info(f'✅ [HARVEST_UI] {summary}')
    return summary, golden_wav or None, profile['golden_ref_text'], _voice_table_rows()


def handle_select_harvested_voice(voice_id: str | None):
    """Load a harvested profile and auto-populate the F5-TTS reference fields.

    Returns ``(voice_choices_update, summary, golden_ref_wav, golden_ref_text)``.
    """
    voices = list_harvested_voices(Path(VOICES_ROOT))
    choices = [v['voice_id'] for v in voices]
    selected = next((v for v in voices if v['voice_id'] == voice_id), None)
    if selected is None:
        return gr.update(choices=choices), '', None, ''

    summary = (
        f"🎙️ **{selected['voice_id']}** — {selected['num_clips']} câu / "
        f"{selected['total_clean_minutes']:.1f} phút sạch "
        f"({len(selected['processed_video_ids'])} video đã xử lý)"
    )
    if not selected['golden_ref_wav']:
        summary += '\n⚠️ Chưa có golden reference — hãy cào thêm hoặc kiểm tra lại thư mục.'
    logger.info(f'🎙️ [VOICE_SELECT] Nạp profile "{selected["voice_id"]}" vào Khu vực 2 (F5-TTS)')
    return (
        gr.update(choices=choices),
        summary,
        selected['golden_ref_wav'] or None,
        selected['golden_ref_text'],
    )


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


def _create_tts_engine(
    engine_choice: str,
    voice_choice: str,
    rate_pct: int,
    rvc_pth: str = '',
    rvc_index: str = '',
    f5_ref_audio: str | None = None,
    f5_ref_text: str = '',
) -> TTSEngine:
    rate_str = f'{int(rate_pct):+d}%'
    if 'Viterbox' in engine_choice:
        # Viterbox reads Vietnamese directly and clones the Vui Vẻ timbre from
        # the golden reference; the tuned defaults in ViterboxTTSEngine give a
        # calm, clearly-articulated pace.
        return ViterboxTTSEngine()
    if 'F5-TTS' in engine_choice:
        if not f5_ref_audio:
            raise ValueError('Vui lòng tải lên file audio mẫu (10-15s) cho F5-TTS.')
        return F5TTSEngine(ref_audio=Path(f5_ref_audio), ref_text=f5_ref_text)
    if 'RVC' in engine_choice:
        if not rvc_pth:
            raise ValueError('Vui lòng nhập đường dẫn file model .pth cho chế độ Edge-TTS + RVC.')
        return EdgeTTSEngine(
            voice=voice_choice,
            rate=rate_str,
            rvc_model_path=Path(rvc_pth),
            rvc_index_path=Path(rvc_index) if rvc_index else None,
        )
    return EdgeTTSEngine(voice=voice_choice, rate=rate_str)


def handle_preview_tts(
    sample_text: str,
    engine_choice: str,
    voice_choice: str,
    rate_pct: int,
    rvc_pth: str,
    rvc_index: str,
    f5_ref_audio: str | None,
    f5_ref_text: str,
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
            rvc_pth=rvc_pth,
            rvc_index=rvc_index,
            f5_ref_audio=f5_ref_audio,
            f5_ref_text=f5_ref_text,
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
    rvc_pth: str,
    rvc_index: str,
    f5_ref_audio: str | None,
    f5_ref_text: str,
    force_rerender: bool = False,
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
        rvc_pth=rvc_pth,
        rvc_index=rvc_index,
        f5_ref_audio=f5_ref_audio,
        f5_ref_text=f5_ref_text,
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
    status = f'Hoàn tất {len(results)} chương — Tổng thời lượng: {total_dur} | File: {out_m4b.name}'
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

                # KHU VỰC 2
                with gr.Group():
                    gr.Markdown('### 2. Cấu hình Giọng đọc (TTS Engine) & Nghe thử (Preview)')
                    with gr.Row():
                        engine_choice = gr.Radio(
                            label='Chế độ Engine',
                            choices=[
                                'Viterbox Zero-shot TTS (Giọng Vui Vẻ)',
                                'Edge-TTS (Siêu nhanh)',
                                'F5-TTS Clone (Nhân bản giọng RTX 3060)',
                                'Edge-TTS + RVC (Đổi màu giọng RTX 3060)',
                            ],
                            value='Viterbox Zero-shot TTS (Giọng Vui Vẻ)' if 'vui_ve' in _harvested_voice_ids() else 'Edge-TTS (Siêu nhanh)',
                        )

                    with gr.Row(visible=not ('vui_ve' in _harvested_voice_ids())) as edge_controls:
                        voice_choice = gr.Dropdown(
                            label='Giọng đọc cơ sở (Microsoft Edge Neural)',
                            choices=['vi-VN-NamMinhNeural', 'vi-VN-HoaiMyNeural'],
                            value='vi-VN-NamMinhNeural',
                            scale=2,
                        )
                        rate_slider = gr.Slider(
                            label='Tốc độ đọc (%)',
                            minimum=-30,
                            maximum=50,
                            step=5,
                            value=0,
                            scale=1,
                        )

                    # Khối cấu hình F5-TTS hiển thị rõ ràng ngay tại Khu vực 2
                    vui_ve_init = _read_voice_profile(Path(VOICES_ROOT) / 'vui_ve') if 'vui_ve' in _harvested_voice_ids() else {}
                    with gr.Group(visible='vui_ve' in _harvested_voice_ids()) as f5_controls:
                        gr.Markdown('#### 🎙️ Nhân bản giọng F5-TTS (Zero-shot Voice Cloning trên GPU RTX 3060)')
                        with gr.Row():
                            harvested_voice_dropdown = gr.Dropdown(
                                label='Profile giọng đã cào (chọn để nạp vào Khu vực 2)',
                                choices=_harvested_voice_ids(),
                                value='vui_ve' if 'vui_ve' in _harvested_voice_ids() else None,
                                interactive=True,
                                scale=3,
                            )
                            refresh_voices_btn = gr.Button('🔄 Làm mới danh sách giọng', scale=1)
                        harvested_summary = gr.Markdown(
                            '🎙️ **vui_ve** — 226 câu / 26.7 phút sạch (đã nạp Golden Ref)'
                            if 'vui_ve' in _harvested_voice_ids() else ''
                        )
                        with gr.Row():
                            f5_ref_audio = gr.Audio(
                                label='Audio mẫu Golden Reference (5–10s)',
                                value=vui_ve_init.get('golden_ref_wav') or None,
                                type='filepath',
                                scale=1,
                            )
                            f5_ref_text = gr.Textbox(
                                label='Transcript khớp với Audio mẫu F5-TTS',
                                value=vui_ve_init.get('golden_ref_text') or '',
                                lines=4,
                                scale=2,
                            )

                    with gr.Group(visible=False) as rvc_controls:
                        gr.Markdown('#### 🎛️ Cấu hình RVC v2 Voice Conversion')
                        with gr.Row():
                            rvc_pth = gr.Textbox(label='Đường dẫn model RVC (.pth)', placeholder='/path/to/voice.pth')
                            rvc_index = gr.Textbox(label='Đường dẫn file RVC (.index)', placeholder='/path/to/voice.index')

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
                            harvest_status = gr.Textbox(label='Trạng thái cào giọng', interactive=False)
                            harvest_table = gr.Dataframe(
                                headers=['Voice ID', 'Số câu', 'Phút sạch', 'Golden ref'],
                                datatype=['str', 'str', 'str', 'str'],
                                label='Danh sách giọng đã cào',
                                value=_voice_table_rows(),
                                interactive=False,
                            )

        def toggle_engine_ui(choice: str):
            is_f5 = 'F5-TTS' in choice
            is_rvc = 'RVC' in choice
            is_edge = not is_f5 and not is_rvc
            return (
                gr.update(visible=is_edge or is_rvc),
                gr.update(visible=is_f5),
                gr.update(visible=is_rvc),
            )

        engine_choice.change(
            fn=toggle_engine_ui,
            inputs=[engine_choice],
            outputs=[edge_controls, f5_controls, rvc_controls],
        )
        epub_input.change(
            fn=handle_epub_upload,
            inputs=[epub_input],
            outputs=[state, book_title, book_author, cover_preview, chapter_selector, md_editor],
        )
        harvest_btn.click(
            fn=handle_harvest_voice,
            inputs=[harvest_voice_id, harvest_query, harvest_minutes, harvest_max_videos],
            outputs=[harvest_status, f5_ref_audio, f5_ref_text, harvest_table],
        )
        harvested_voice_dropdown.change(
            fn=handle_select_harvested_voice,
            inputs=[harvested_voice_dropdown],
            outputs=[harvested_voice_dropdown, harvested_summary, f5_ref_audio, f5_ref_text],
        )
        refresh_voices_btn.click(
            fn=_load_harvested_voices,
            inputs=None,
            outputs=[harvest_table, harvested_voice_dropdown],
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
                rvc_pth,
                rvc_index,
                f5_ref_audio,
                f5_ref_text,
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
                rvc_pth,
                rvc_index,
                f5_ref_audio,
                f5_ref_text,
                force_checkbox,
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

