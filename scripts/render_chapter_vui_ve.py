"""Render the 'PHẦN DẪN' chapter to Vietnamese audio with the Vui Vẻ RVC voice.

Pipeline: Edge-TTS (vi-VN-NamMinhNeural) -> RVC v2 voice conversion -> 24 kHz WAV.
Outputs ``workspace/dua_tre_hu_tu_kim_tran/output_chapter_vui_ve.wav`` (and a
packaged ``.m4b`` when the metadata is available).
"""
from __future__ import annotations

import logging
import wave
from pathlib import Path

from core.epub_parser import parse_book_markdown
from core.tts_engine import EdgeTTSEngine, render_chapter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('render_chapter_vui_ve')

BOOK_MD = Path('workspace/dua_tre_hu_tu_kim_tran/book.md')
OUT_DIR = Path('workspace/dua_tre_hu_tu_kim_tran')
MODEL_DIR = Path('models/rvc_vui_ve')


def main() -> Path:
    model_pth = MODEL_DIR / 'vui_ve.pth'
    index_pth = MODEL_DIR / 'vui_ve.index'
    if not model_pth.exists():
        # Fall back to the pretrained base generator if no fine-tune is present.
        model_pth = Path('models/rvc_pretrained/pretrained_v2/f0G40k.pth')
    logger.info(f'🎚️ RVC model: {model_pth}')
    logger.info(f'🔎 RVC index: {index_pth if index_pth.exists() else "(none)"}')

    engine = EdgeTTSEngine(
        voice='vi-VN-NamMinhNeural',
        rvc_model_path=model_pth,
        rvc_index_path=index_pth if index_pth.exists() else None,
        rvc_index_rate=0.75,
    )

    chapters = parse_book_markdown(BOOK_MD.read_text(encoding='utf-8'))
    chapter = next((c for c in chapters if 'DẪN' in c.title.upper()), chapters[0])
    logger.info(f"📖 Render chương: '{chapter.title}' ({len(chapter.markdown)} ký tự)")

    result = render_chapter(chapter, engine, OUT_DIR / 'chapters_audio', force=True)
    final = OUT_DIR / 'output_chapter_vui_ve.wav'
    final.write_bytes(result.audio_path.read_bytes())

    with wave.open(str(final), 'rb') as wf:
        dur_s = wf.getnframes() / wf.getframerate()
    logger.info(f'✅ Đã ghi: {final} ({dur_s:.1f}s, {result.duration_ms}ms)')

    # Package a single-chapter .m4b as well (the plan accepts .wav OR .m4b).
    try:
        from core.models import BookMetadata
        from core.m4b_packager import package_m4b

        meta = BookMetadata(
            title='Đứa Trẻ Hư - PHẦN DẪN (Vui Vẻ)',
            author='Kim Trần',
            cover_path=BOOK_MD.parent / 'cover.jpeg',
        )
        m4b = OUT_DIR / 'output_chapter_vui_ve.m4b'
        package_m4b(meta, [result], m4b)
        logger.info(f'🎬 Đã đóng gói: {m4b}')
    except Exception as exc:  # pragma: no cover - packaging is best-effort here
        logger.warning(f'⚠️ Bỏ qua đóng gói m4b: {exc}')
    return final


if __name__ == '__main__':
    main()
