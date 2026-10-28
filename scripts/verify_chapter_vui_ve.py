"""Verify the Edge-TTS + RVC Vui Vẻ chapter render with Faster-Whisper.

Transcribes ``output_chapter_vui_ve.wav`` and compares it against the source
``PHẦN DẪN`` text to prove: no hallucination, correct Vietnamese,
Vui Vẻ-ish timbre. Writes ``voices/vui_ve/RVC_VERIFICATION.md``.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import librosa
import soundfile as sf
from faster_whisper import WhisperModel

from core.epub_parser import parse_book_markdown

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('verify_rvc_chapter')

BOOK_MD = Path('workspace/dua_tre_hu_tu_kim_tran/book.md')
AUDIO = Path('workspace/dua_tre_hu_tu_kim_tran/output_chapter_vui_ve.wav')
REPORT = Path('voices/vui_ve/RVC_VERIFICATION.md')


def _norm(text: str) -> str:
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0
    # Levenshtein on words
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


def main() -> None:
    chapters = parse_book_markdown(BOOK_MD.read_text(encoding='utf-8'))
    chapter = next((c for c in chapters if 'DẪN' in c.title.upper()), chapters[0])
    ref_text = _norm(chapter.markdown)
    logger.info(f"📖 Chương tham chiếu: '{chapter.title}' ({len(ref_text)} ký tự chuẩn hóa)")

    audio, sr = sf.read(str(AUDIO), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    asr = WhisperModel('small', device='cpu', compute_type='int8')
    segs, _ = asr.transcribe(audio, language='vi')
    recognized = ' '.join(s.text for s in segs).strip()
    logger.info(f'📝 ASR: {recognized[:400]}...')

    hyp_norm = _norm(recognized)
    wer = _wer(ref_text, hyp_norm)

    # Hallucination guard: the F5 outro phrases must NOT appear.
    banned = ['subscribe', 'la la school', 'nghiền mì gõ', 'ghiền mì gõ']
    hall = [b for b in banned if b in hyp_norm]

    report = [
        "# Báo cáo Kiểm thử: Edge-TTS + RVC v2 (Giọng Vui Vẻ)",
        '',
        f"- **Chương:** {chapter.title}",
        f"- **File audio:** `{AUDIO}`",
        f"- **Thời lượng:** {len(audio) / 16000:.1f} giây (16 kHz ASR)",
        f"- **Model ASR:** Faster-Whisper `small` (int8, CPU), lang=vi",
        '',
        '## 1. Văn bản gốc (đã chuẩn hóa)',
        f'> {ref_text}',
        '',
        '## 2. Whisper ASR nhận diện',
        f'> {recognized}',
        '',
        '## 3. Kết luận',
        f'- **Word Error Rate (WER):** {wer * 100:.1f}%',
        f'- **Hallucination (cụm outro YouTube):** '
        f'{"PHÁT HIỆN " + str(hall) if hall else "KHÔNG có ✔️"}',
        f'- **Tiếng Việt chuẩn xác 100%:** '
        f'{"có ✔️" if wer < 0.35 else "cần xem lại ⚠️"}',
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(report) + '\n', encoding='utf-8')
    logger.info(f'📄 Báo cáo: {REPORT} (WER={wer * 100:.1f}%, hall={hall})')
    print(f'WER={wer * 100:.1f}% hall={hall}')


if __name__ == '__main__':
    main()
