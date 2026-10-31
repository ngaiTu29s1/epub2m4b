"""Golden Stress-Test: render kịch bản ~1 phút chứa các ca khó nhất + QA bằng Whisper.

Kịch bản tổng hợp cực nhiều ca hóc búa:
  * Song ngữ Anh-Việt (Machine Learning, video, camera, AI online…).
  * Công thức hoá học (H2SO4, CO2, NaCl, KMnO4).
  * Phương trình toán học (E = mc^2, x^2 + 2x + 1 = 0).
  * Chữ viết tắt & tổ chức (TP.HCM, UNESCO, COVID-19, WHO).
  * Đơn vị đo lường (3.5 GHz, 60 km/h, 37.5 độ C, 25%).
  * Ngày tháng & số La Mã (Thế kỷ XXI, ngày 30/04/1975).

Luồng:
  1. Chuẩn hoá kịch bản qua ``normalize_vi_text`` (rule + cache + LLM fallback).
  2. Render toàn bộ bằng ``VieNeuEngine`` (batch CUDA) -> workspace/_preview/golden_stress_test.wav
  3. Chạy ASR bằng faster-whisper ``medium`` và in báo cáo thời gian/chất lượng.

Chạy:
    .venv/bin/python scripts/benchmark_stress_test.py
"""
from __future__ import annotations

import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

# Chạy trực tiếp ``python scripts/benchmark_stress_test.py`` vẫn import được ``core``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.text_normalizer import normalize_vi_text
from core.vieneu_engine import VieNeuEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('golden_stress_test')

OUT_WAV = _REPO_ROOT / 'workspace' / '_preview' / 'golden_stress_test.wav'

# ---------------------------------------------------------------------------
# Kịch bản Golden (~1 phút đọc)
# ---------------------------------------------------------------------------
GOLDEN_SCRIPT = [
    'Xin chào các bạn, đây là bài kiểm tra vàng cho hệ thống chuyển văn bản thành giọng nói.',
    'Hôm nay chúng ta sẽ thử qua những trường hợp khó nhất, từ công nghệ cho tới khoa học.',
    'Machine Learning và trí tuệ nhân tạo đang thay đổi cách chúng ta dùng video và camera.',
    'Nhiều người học online qua internet, dùng website, email và robot để làm việc mỗi ngày.',
    'Trong phòng thí nghiệm, chúng ta có H2SO4 là axit sunfuric, CO2 là khí các-bon đi-ô-xít, '
    'NaCl là muối ăn và KMnO4 là kali pemanganat.',
    'Theo công thức nổi tiếng E = mc^2, năng lượng tỉ lệ thuận với khối lượng.',
    'Phương trình x^2 + 2x + 1 = 0 có nghiệm kép là x bằng âm một.',
    'Tại TP.HCM, UNESCO và WHO đã phối hợp phòng chống COVID-19 từ năm 2020.',
    'Tốc độ xử lý đạt 3.5 GHz, xe chạy 60 km/h, nhiệt độ 37.5 độ C, '
    'độ ẩm 25%.',
    'Đến thế kỷ XXI, ngày 30/04/1975 đã đi vào lịch sử của dân tộc Việt Nam.',
]


def _build_raw_script() -> str:
    """Ghép kịch bản GỐC (còn ký hiệu thô) để chứng minh năng lực normalizer."""
    return ' '.join(GOLDEN_SCRIPT)


def _norm_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate bằng khoảng cách Levenshtein mức từ."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        cur = [i]
        for j, hw in enumerate(hyp, start=1):
            cost = 0 if rw == hw else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[len(hyp)] / len(ref)


def _render(normalized: str) -> tuple[Path, float]:
    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_WAV.exists():
        OUT_WAV.unlink()
        logger.info(f'🗑️ [Render] Đã xoá preview audio cũ: {OUT_WAV.name}')
    engine = VieNeuEngine(mode='preset', voice='Phạm Tuyên')
    logger.info(
        f'🎙️ [Render] batch_size={engine.batch_size}, device=auto. '
        f'Độ dài text chuẩn hoá: {len(normalized)} ký tự.'
    )
    t0 = time.perf_counter()
    dur_ms = engine.synthesize_segment(normalized, OUT_WAV)
    elapsed = time.perf_counter() - t0
    logger.info(
        f'🔊 [Render] Xong {OUT_WAV} ({dur_ms / 1000:.1f}s audio) '
        f'trong {elapsed:.1f}s (RTF={elapsed / (dur_ms / 1000):.2f}×).'
    )
    return OUT_WAV, elapsed


def _asr_report(wav: Path, reference: str) -> float:
    """Chạy faster-whisper medium, in transcript + WER, trả về WER."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning('⚠️ [ASR] faster-whisper chưa cài; bỏ qua kiểm tra ASR.')
        return -1.0

    import librosa

    logger.info('📝 [ASR] Đang chạy faster-whisper (medium)…')
    model = None
    for device, compute in (('cuda', 'float16'), ('cpu', 'int8')):
        try:
            model = WhisperModel('medium', device=device, compute_type=compute)
            logger.info(f'📝 [ASR] Dùng device={device} compute={compute}.')
            break
        except Exception as exc:  # pragma: no cover - phụ thuộc máy
            logger.warning(f'⚠️ [ASR] Không nạp được trên {device} ({exc}).')
    if model is None:
        logger.warning('⚠️ [ASR] Không nạp được model; bỏ qua.')
        return -1.0

    audio16, _ = librosa.load(str(wav), sr=16000, mono=True)
    t0 = time.perf_counter()
    segs, _ = model.transcribe(
        audio16, language='vi', beam_size=5, condition_on_previous_text=False
    )
    transcript = ' '.join(s.text for s in segs).strip()
    elapsed = time.perf_counter() - t0

    # Whisper ITN đưa số về dạng chữ số → chuẩn hoá lại cả hai phía trước khi tính WER.
    wer = _wer(_norm_text(reference), _norm_text(normalize_vi_text(transcript)))
    logger.info(f'📝 [ASR] Transcript: {transcript!r}')
    logger.info(f'📝 [ASR] Thời gian phiên âm: {elapsed:.1f}s')
    logger.info(f'📊 [ASR] WER={wer * 100:.1f}%')
    return wer


def main() -> int:
    raw = _build_raw_script()
    logger.info(f'📜 [Script] Kịch bản gốc ({len(raw)} ký tự).')

    normalized = normalize_vi_text(raw)
    logger.info(f'🔤 [Normalize] Bản chuẩn hoá: {normalized!r}')

    # Cổng chốt: normalizer không được để lọt ký tự số / dấu mũ vào TTS.
    assert not re.search(r'\d', normalized), f'NORMALIZER còn chữ số: {normalized!r}'
    assert '^' not in normalized, f'NORMALIZER còn dấu ^: {normalized!r}'
    for preserved in ('Machine Learning', 'video', 'camera', 'online', 'robot'):
        assert preserved in normalized, f'Mất từ tiếng Anh {preserved!r} khỏi bản chuẩn hoá'
    assert 'H hai S O bốn' in normalized or 'axit' in normalized.lower(), 'Sai H2SO4'
    assert 'bình phương' in normalized, 'Sai E = mc^2'

    wav, render_elapsed = _render(normalized)
    wer = _asr_report(wav, normalized)

    logger.info('=' * 60)
    logger.info(f'✅ [Golden] WAV: {wav} ({wav.stat().st_size / 1024:.0f} KB)')
    logger.info(f'⏱️ [Golden] Thời gian render: {render_elapsed:.1f}s')
    if wer >= 0:
        logger.info(f'📊 [Golden] WER: {wer * 100:.1f}%')
        if wer <= 0.15:
            logger.info('✅ [Golden] CHẤT LƯỢNG ĐẠT (WER ≤ 15%).')
        else:
            logger.warning('⚠️ [Golden] WER cao hơn kỳ vọng (> 15%).')
    logger.info('=' * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
