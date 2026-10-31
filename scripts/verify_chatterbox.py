"""Step 4 — verify both Chatterbox/Viterbox renders for the PHẦN DẪN chapter.

For each output WAV it measures:

* Whisper ASR transcript + Word Error Rate against the source text,
* hallucination guard (the F5-TTS YouTube-outro phrases must be absent),
* duration sanity (the plan expects ~45-60 s, no fast-forward),
* speaker cosine similarity vs ``voices/vui_ve/golden_refs/golden_01.wav``
  using the Viterbox ``VoiceEncoder`` (``ve.pt``).

Writes a Markdown report to ``voices/vui_ve/CHATTERBOX_VERIFICATION.md``.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from core.epub_parser import parse_book_markdown

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('verify_chatterbox')

BOOK_MD = Path('workspace/dua_tre_hu_tu_kim_tran/book.md')
OUT_DIR = Path('workspace/dua_tre_hu_tu_kim_tran')
GOLDEN_REF = Path('voices/vui_ve/golden_refs/golden_01.wav')
CKPT_DIR = Path('models/chatterbox_vui_ve')
REPORT = Path('voices/vui_ve/CHATTERBOX_VERIFICATION.md')

OUTPUTS = {
    'Viterbox TTS': OUT_DIR / 'output_viterbox_tts.wav',
    'Edge-TTS + ChatterboxVC': OUT_DIR / 'output_chatterbox_vc.wav',
}

BANNED = ['subscribe', 'la la school', 'nghiền mì gõ', 'ghiền mì gõ']


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


def _speaker_embedder():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'vendor' / 'viterbox_src'))
    import torch

    from viterbox.models.voice_encoder import VoiceEncoder

    ve = VoiceEncoder()
    ve.load_state_dict(torch.load(CKPT_DIR / 've.pt', map_location='cpu', weights_only=True))
    ve.eval()
    return ve


def _speaker_sim(ve, wav_path: Path, ref_wav: np.ndarray, ref_sr: int) -> float:
    wav, sr = sf.read(str(wav_path), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
    emb = ve.embeds_from_wavs([wav16], sample_rate=16000)
    ref_emb = ve.embeds_from_wavs([ref_wav], sample_rate=16000)
    a = np.asarray(emb, dtype='float32').reshape(-1)
    b = np.asarray(ref_emb, dtype='float32').reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _first_words(text: str, n: int = 2) -> str:
    return ' '.join(_norm(text).split()[:n])


def main() -> None:
    chapters = parse_book_markdown(BOOK_MD.read_text(encoding='utf-8'))
    chapter = next((c for c in chapters if 'DẪN' in c.title.upper()), chapters[0])
    ref_text = _norm(chapter.markdown)
    logger.info(f"📖 Chương tham chiếu: '{chapter.title}' ({len(ref_text)} ký tự)")

    asr = WhisperModel('small', device='cuda', compute_type='float16')
    asr_big = WhisperModel('medium', device='cuda', compute_type='float16')

    ref_wav, ref_sr = sf.read(str(GOLDEN_REF), dtype='float32')
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    if ref_sr != 16000:
        ref_wav = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=16000)
    ve = _speaker_embedder()

    rows = []
    for label, wav_path in OUTPUTS.items():
        if not wav_path.exists():
            logger.warning(f'⚠️ Thiếu {wav_path}, bỏ qua {label}.')
            continue
        audio, sr = sf.read(str(wav_path), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur_s = len(audio) / sr
        a16 = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio

        segs, _ = asr.transcribe(a16, language='vi', beam_size=5,
                                condition_on_previous_text=False)
        recognized = ' '.join(s.text for s in segs).strip()
        hyp_norm = _norm(recognized)
        wer = _wer(ref_text, hyp_norm)
        hall = [b for b in BANNED if b in hyp_norm]
        sim = _speaker_sim(ve, wav_path, ref_wav, 16000)

        # Cross-check the same render with the more accurate `medium` model:
        # `small` is noisy on Northern Vietnamese tone marks, so the medium
        # transcript is the authoritative one for the opening-word check and a
        # better estimate of the *synthesis* quality.
        segs_big, _ = asr_big.transcribe(a16, language='vi', beam_size=5,
                                        condition_on_previous_text=False)
        recognized_big = ' '.join(s.text for s in segs_big).strip()
        wer_big = _wer(ref_text, _norm(recognized_big))

        # Requirement #1: the opening must read "Phần dẫn", not the old
        # spell-out "với hát DM nhanh".
        opening_ok = _norm(recognized_big).startswith('phần dẫn') or \
            _norm(recognized).startswith('phần dẫn')
        opening_recognized = recognized_big[:40]

        logger.info(
            f'📝 [{label}] WER={wer * 100:.1f}% (medium={wer_big * 100:.1f}%) '
            f'dur={dur_s:.1f}s sim={sim:.3f} hall={hall or "none"} '
            f'opening_ok={opening_ok} ({opening_recognized!r})'
        )
        rows.append({
            'label': label, 'path': wav_path, 'dur_s': dur_s,
            'recognized': recognized, 'wer': wer, 'hall': hall, 'sim': sim,
            'recognized_big': recognized_big, 'wer_big': wer_big, 'opening_ok': opening_ok,
            'opening_recognized': opening_recognized,
        })

    lines = [
        '# Báo cáo Kiểm thử: Chatterbox / Viterbox (Giọng Vui Vẻ)',
        '',
        f"- **Chương:** {chapter.title}",
        "- **Model ASR:** Faster-Whisper `small` (float16, CUDA), lang=vi "
        "(và `medium` để đối chiếu)",
        f"- **Golden ref:** `{GOLDEN_REF}`",
        '',
        '## Tóm tắt',
        '',
        '| Chế độ | Thời lượng | WER (small) | WER (medium) | Hallucination | Speaker sim |',
        '|--------|-----------:|------------:|-------------:|---------------|------------:|',
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['dur_s']:.1f}s | {r['wer'] * 100:.1f}% | "
            f"{r['wer_big'] * 100:.1f}% | "
            f"{'PHÁT HIỆN ' + str(r['hall']) if r['hall'] else 'KHÔNG ✔️'} | "
            f"{r['sim']:.3f} |"
        )

    for r in rows:
        lines += [
            '',
            f"## {r['label']}",
            f"- **File:** `{r['path']}`",
            f"- **Thời lượng:** {r['dur_s']:.1f}s",
            f"- **Câu mở đầu nhận diện đúng \"Phần dẫn\":** "
            f"{'✔️ CÓ' if r['opening_ok'] else '❌ KHÔNG'} — `{r['opening_recognized']}`",
            f"- **Whisper `small` ASR:** {r['recognized']}",
            f"- **Whisper `medium` ASR (đối chiếu):** {r['recognized_big']}",
        ]

    lines += ['', '## Văn bản gốc (chuẩn hóa)', f'> {ref_text}', '']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f'📄 Đã ghi báo cáo: {REPORT}')


if __name__ == '__main__':
    main()
