"""Render the ``# PHẦN DẪN`` chapter with the optimized Viterbox engine.

Produces a Vietnamese audiobook clip as a 24 kHz mono WAV plus a packaged
single-chapter ``.m4b``:

* Viterbox TTS (Vui Vẻ clone) -> ``output_viterbox_tts.wav`` & ``.m4b``

The ``--with-vc`` flag additionally renders the Edge-TTS + ChatterboxVC mode
for A/B comparison (``output_chatterbox_vc.wav`` & ``.m4b``).

Run with the repo venv::

    .venv/bin/python scripts/render_chapter_chatterbox.py
"""
from __future__ import annotations

import logging
import sys
import wave
from pathlib import Path

from core.epub_parser import parse_book_markdown
from core.tts_engine import TTSEngine, render_chapter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('render_chapter_chatterbox')

BOOK_MD = Path('workspace/dua_tre_hu_tu_kim_tran/book.md')
OUT_DIR = Path('workspace/dua_tre_hu_tu_kim_tran')
AUDIO_DIR = OUT_DIR / 'chapters_audio'
GOLDEN_REF = Path('voices/vui_ve/golden_refs/golden_01.wav')


def _find_chapter():
    chapters = parse_book_markdown(BOOK_MD.read_text(encoding='utf-8'))
    return next((c for c in chapters if 'DẪN' in c.title.upper()), chapters[0])


def _render(label: str, engine: TTSEngine, chapter, out_wav: Path) -> Path:
    logger.info(f'🎬 [{label}] Bắt đầu render chương "{chapter.title}"...')
    result = render_chapter(chapter, engine, AUDIO_DIR, force=True)
    out_wav.write_bytes(result.audio_path.read_bytes())

    with wave.open(str(out_wav), 'rb') as wf:
        dur_s = wf.getnframes() / wf.getframerate()
    logger.info(f'✅ [{label}] Đã ghi {out_wav} ({dur_s:.1f}s, {result.duration_ms}ms)')

    try:
        from core.m4b_packager import package_m4b
        from core.models import BookMetadata

        meta = BookMetadata(
            title=f'Đứa Trẻ Hư - PHẦN DẪN ({label})',
            author='Kim Trần',
            cover_path=BOOK_MD.parent / 'cover.jpeg',
        )
        m4b = out_wav.with_suffix('.m4b')
        package_m4b(meta, [result], m4b)
        logger.info(f'🎬 [{label}] Đã đóng gói {m4b}')
    except Exception as exc:  # packaging is best-effort
        logger.warning(f'⚠️ [{label}] Bỏ qua đóng gói m4b: {exc}')
    return out_wav


def main() -> None:
    chapter = _find_chapter()
    logger.info(f"📖 Chương: '{chapter.title}' ({len(chapter.markdown)} ký tự)")

    from core.chatterbox_engine import ChatterboxVCEngine, ViterboxTTSEngine

    # Primary mode: Viterbox zero-shot TTS cloning the Vui Vẻ golden reference.
    viterbox = ViterboxTTSEngine(audio_prompt=GOLDEN_REF)
    _render('Viterbox TTS', viterbox, chapter, OUT_DIR / 'output_viterbox_tts.wav')

    if '--with-vc' in sys.argv:
        # Optional A/B mode: Edge-TTS pronunciation + S3Gen voice conversion.
        vc = ChatterboxVCEngine(audio_prompt=GOLDEN_REF)
        _render('Edge+ChatterboxVC', vc, chapter, OUT_DIR / 'output_chatterbox_vc.wav')


if __name__ == '__main__':
    main()
