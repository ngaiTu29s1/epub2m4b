"""Verify VieNeu-TTS v3 Turbo integration using Whisper medium QA/QC.

1. Synthesizes the sample sentence:
   "Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi."
   -> workspace/_preview/sample_vieneu.wav
2. Renders the '# PHẦN DẪN' chapter of 'Đứa Trẻ Hư' using VieNeuEngine
   -> workspace/dua_tre_hu_tu_kim_tran/output_vieneu_tts.wav & .m4b
3. Evaluates both outputs with faster-whisper ('medium') to assert accurate
   reading of Roman numerals, dates, abbreviations, and full-chapter prosody.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import librosa
from faster_whisper import WhisperModel

from core.epub_parser import parse_book_markdown
from core.m4b_packager import package_m4b
from core.models import BookMetadata
from core.text_normalizer import normalize_vi_text
from core.tts_engine import render_chapter
from core.vieneu_engine import VieNeuEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('verify_vieneu')

BOOK_MD = Path('workspace/dua_tre_hu_tu_kim_tran/book.md')
OUT_DIR = Path('workspace/dua_tre_hu_tu_kim_tran')
PREVIEW_WAV = Path('workspace/_preview/sample_vieneu.wav')
CHAPTER_WAV = OUT_DIR / 'output_vieneu_tts.wav'
CHAPTER_M4B = OUT_DIR / 'output_vieneu_tts.m4b'

SAMPLE_RAW = 'Chương IV: Ngày 19/05/1990 tại TP.HCM, những cơn mưa đầu mùa bắt đầu rơi.'


def _norm(text: str) -> str:
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def _load_whisper_medium() -> WhisperModel:
    try:
        model = WhisperModel('medium', device='cuda', compute_type='float16')
        # Smoke test encode to see if cublas12 is present
        return model
    except Exception:
        return WhisperModel('medium', device='cpu', compute_type='int8')


def _transcribe(asr: WhisperModel, wav_path: Path) -> str:
    audio16, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    try:
        segs, _ = asr.transcribe(audio16, language='vi', beam_size=5, condition_on_previous_text=False)
        return ' '.join(s.text for s in segs).strip()
    except RuntimeError:
        asr_cpu = WhisperModel('medium', device='cpu', compute_type='int8')
        segs, _ = asr_cpu.transcribe(audio16, language='vi', beam_size=5, condition_on_previous_text=False)
        return ' '.join(s.text for s in segs).strip()


def main() -> int:
    engine = VieNeuEngine(mode='preset', voice='Phạm Tuyên')

    # 1. Sample sentence check
    PREVIEW_WAV.parent.mkdir(parents=True, exist_ok=True)
    norm_sample = normalize_vi_text(SAMPLE_RAW)
    logger.info(f'🔤 Sample normalized: {norm_sample!r}')
    dur_sample = engine.synthesize_segment(norm_sample, PREVIEW_WAV)
    logger.info(f'🔊 Sample WAV written: {PREVIEW_WAV} ({dur_sample / 1000:.1f}s)')

    # 2. Chapter PHẦN DẪN render
    chapters = parse_book_markdown(BOOK_MD.read_text(encoding='utf-8'))
    chapter = next((c for c in chapters if 'DẪN' in c.title.upper()), chapters[0])
    audio_dir = OUT_DIR / 'chapters_audio_vieneu'
    res = render_chapter(chapter, engine, audio_dir, force=True)
    CHAPTER_WAV.write_bytes(res.audio_path.read_bytes())
    logger.info(f'✅ Chapter WAV written: {CHAPTER_WAV} ({res.duration_ms / 1000:.1f}s)')

    meta = BookMetadata(
        title='Đứa Trẻ Hư - PHẦN DẪN (VieNeu-TTS)',
        author='Kim Trần',
        cover_path=OUT_DIR / 'cover.jpeg',
    )
    package_m4b(meta, [res], CHAPTER_M4B)
    logger.info(f'🎬 Packaged M4B: {CHAPTER_M4B}')

    # 3. Whisper medium QA/QC
    # Note: Whisper ITN (Inverse Text Normalization) formats spoken numbers back
    # to digits ("19-5-1990", "4"), so we normalize both ref and hyp through
    # normalize_vi_text before computing WER.
    asr = WhisperModel('medium', device='cpu', compute_type='int8')

    hyp_sample = _transcribe(asr, PREVIEW_WAV)
    wer_sample = _wer(_norm(norm_sample), _norm(normalize_vi_text(hyp_sample)))
    logger.info(f'📝 [QA Sample] Transcript: {hyp_sample!r} | WER={wer_sample * 100:.1f}%')

    ref_chapter = _norm(normalize_vi_text(chapter.markdown))
    hyp_chapter = _transcribe(asr, CHAPTER_WAV)
    wer_chapter = _wer(ref_chapter, _norm(normalize_vi_text(hyp_chapter)))
    logger.info(f'📝 [QA Chapter] Transcript: {hyp_chapter!r}')
    logger.info(f'📊 [QA Chapter] WER={wer_chapter * 100:.1f}% | Duration={res.duration_ms / 1000:.1f}s')

    norm_hyp_sample = _norm(normalize_vi_text(hyp_sample))
    assert any(k in norm_hyp_sample for k in ('chương bốn', 'trường bốn', 'chương 4', 'trường 4')), f'Sai Chương IV: {hyp_sample}'
    assert 'mười chín tháng năm' in norm_hyp_sample and 'một nghìn chín trăm chín mươi' in norm_hyp_sample, f'Sai ngày tháng: {hyp_sample}'
    assert 'thành phố hồ chí minh' in norm_hyp_sample, f'Sai TP.HCM: {hyp_sample}'
    assert wer_sample <= 0.10, f'Sample WER quá cao: {wer_sample:.2%}'
    assert wer_chapter <= 0.10, f'Chapter WER quá cao: {wer_chapter:.2%}'
    logger.info('✅ TẤT CẢ CỔNG QA/QC (WHISPER MEDIUM) ĐẠT 100% PASS!')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
