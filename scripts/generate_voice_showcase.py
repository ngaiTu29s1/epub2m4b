#!/usr/bin/env python
"""Render kho mẫu giọng Golden Dataset cho MỌI giọng chính thống của VieNeu.

Mỗi giọng đọc cùng kịch bản Golden ~1 phút (``GOLDEN_SCRIPT`` trong
``scripts/benchmark_stress_test.py``), xuất:

    workspace/golden_samples/<voice_id>.wav   PCM 16-bit, 24 kHz mono
    workspace/golden_samples/<voice_id>.ogg   Opus 48 kbps mono (Telegram voice note)
    workspace/golden_samples/index.json       metadata cho cả bộ giọng

Kịch bản được normalizer xử lý MỘT LẦN rồi mới đem đi render cho từng giọng (batch
pipeline của engine), nên 25 giọng tốn đúng 25 lượt render thay vì 25 × normalizer.

Cách dùng:
    .venv/bin/python scripts/generate_voice_showcase.py
    .venv/bin/python scripts/generate_voice_showcase.py --no-ogg        # chỉ WAV
    .venv/bin/python scripts/generate_voice_showcase.py --voices "Phạm Tuyên,Trúc Ly"
    .venv/bin/python scripts/generate_voice_showcase.py --force        # render lại
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import soundfile as sf

from core.text_normalizer import normalize_vi_text
from core.vieneu_engine import (
    _get_ffmpeg_bin,
    get_categorized_voices,
    get_available_preset_voices,
    load_vieneu_model,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('voice_showcase')

OUT_DIR = _REPO_ROOT / 'workspace' / 'golden_samples'
INDEX_PATH = OUT_DIR / 'index.json'

# Opus voice-note settings Telegram nhận native.
OPUS_BITRATE = '48k'
OPUS_CHANNELS = '1'


# ---------------------------------------------------------------------------
# Golden script (tái sử dụng nguyên văn từ benchmark stress test)
# ---------------------------------------------------------------------------
def _golden_script() -> str:
    from scripts.benchmark_stress_test import _build_raw_script

    return _build_raw_script()


def _normalized_script() -> str:
    """Kịch bản đã qua normalizer — giống hệt benchmark, render 1 lần dùng chung."""
    return normalize_vi_text(_golden_script())


# ---------------------------------------------------------------------------
# Voice id / naming
# ---------------------------------------------------------------------------
def _slug(text: str) -> str:
    """``Phạm Tuyên`` → ``Pham_Tuyen`` (ASCII, an toàn cho tên file/URL)."""
    ascii_text = unicodedata.normalize('NFD', text)
    ascii_text = ''.join(ch for ch in ascii_text if unicodedata.category(ch) != 'Mn')
    ascii_text = ascii_text.replace('đ', 'd').replace('Đ', 'D')
    ascii_text = re.sub(r'[^A-Za-z0-9]+', '_', ascii_text)
    return ascii_text.strip('_') or 'voice'


def _voice_id(name: str) -> str:
    return _slug(name)


# ---------------------------------------------------------------------------
# Audio export
# ---------------------------------------------------------------------------
def _has_opus_encoder(ffmpeg: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg, '-hide_banner', '-encoders'],
            capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        return False
    return 'libopus' in out or re.search(r'\bopus\b', out) is not None


def _wav_to_opus(wav: Path, ogg: Path) -> bool:
    """Chuyển WAV → OGG/Opus 48 kbps mono. False nếu không có encoder."""
    ffmpeg = shutil.which('ffmpeg')
    try:
        if not ffmpeg:
            ffmpeg = _get_ffmpeg_bin()
    except Exception:  # pragma: no cover - không có ffmpeg ở mọi đường
        logger.warning('⚠️ Không tìm thấy ffmpeg; bỏ qua bitrate Opus.')
        return False
    if not ffmpeg or not Path(ffmpeg).exists():
        logger.warning('⚠️ Không tìm thấy ffmpeg; bỏ qua bitrate Opus.')
        return False
    if not _has_opus_encoder(ffmpeg):
        logger.warning('⚠️ ffmpeg không có Opus encoder; bỏ qua .ogg.')
        return False
    result = subprocess.run(
        [
            ffmpeg, '-y', '-v', 'error',
            '-i', str(wav),
            '-c:a', 'libopus', '-b:a', OPUS_BITRATE, '-ac', OPUS_CHANNELS,
            '-application', 'voip',
            str(ogg),
        ],
        check=False,
    )
    if result.returncode != 0 or not ogg.exists():
        logger.warning(f'⚠️ Chuyển Opus thất bại cho {wav.name} (rc={result.returncode}).')
        return False
    return True


def _duration_sec(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return round(float(info.frames) / float(info.samplerate), 2)
    except Exception:  # pragma: no cover
        return 0.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_voice(engine, text: str, wav_path: Path) -> float:
    """Render một giọng; trả về số giây audio."""
    engine.synthesize_segment(text, wav_path)
    return _duration_sec(wav_path)


def _build_index_entry(
    name: str,
    label: str,
    wav_path: Path,
    ogg_path: Path | None,
    duration_sec: float,
) -> dict:
    return {
        'id': _voice_id(name),
        'name': name,
        'label': label,
        'ogg_path': str(ogg_path.relative_to(_REPO_ROOT)) if ogg_path else None,
        'wav_path': str(wav_path.relative_to(_REPO_ROOT)),
        'duration_sec': duration_sec,
    }


def generate(
    voices: list[tuple[str, str]] | None = None,
    make_ogg: bool = True,
    force: bool = False,
) -> list[dict]:
    """Render showcase cho ``voices`` (mặc định: mọi giọng chính thống)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if voices is None:
        voices = get_available_preset_voices()
    if not voices:
        raise RuntimeError(
            'Không lấy được danh sách giọng từ VieNeu (model chưa tải?). '
            'Chạy lại khi đã có model.'
        )

    text = _normalized_script()
    logger.info(
        f'🎙️ Render {len(voices)} giọng · kịch bản {len(text)} ký tự · batch=auto.'
    )

    entries: list[dict] = []
    started = time.perf_counter()
    for i, (label, name) in enumerate(voices, 1):
        vid = _voice_id(name)
        wav_path = OUT_DIR / f'{vid}.wav'
        ogg_path = OUT_DIR / f'{vid}.ogg'

        if wav_path.exists() and not force:
            logger.info(f'⏭️  [{i}/{len(voices)}] {name}: đã có, bỏ qua (dùng --force để render lại).')
        else:
            from core.vieneu_engine import VieNeuEngine

            engine = VieNeuEngine(mode='preset', voice=name)
            t0 = time.perf_counter()
            duration = _render_voice(engine, text, wav_path)
            logger.info(
                f'✅ [{i}/{len(voices)}] {name}: {duration:.1f}s audio '
                f'trong {time.perf_counter() - t0:.1f}s.'
            )

        ogg_ok = ogg_path.exists()
        if make_ogg and (force or not ogg_ok):
            ogg_ok = _wav_to_opus(wav_path, ogg_path)
        entries.append(
            _build_index_entry(
                name, label, wav_path,
                ogg_path if ogg_ok else None,
                _duration_sec(wav_path),
            )
        )

    INDEX_PATH.write_text(
        json.dumps(
            {
                'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'script_chars': len(text),
                'voice_count': len(entries),
                'categories': {
                    cat: [n for _lab, n in bucket]
                    for cat, bucket in get_categorized_voices(
                        None
                    ).items()
                },
                'voices': entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    logger.info(
        f'🗂️  Ghi {INDEX_PATH} ({len(entries)} giọng, '
        f'{time.perf_counter() - started:.1f}s tổng).'
    )
    return entries


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-ogg', action='store_true', help='Chỉ xuất WAV, bỏ qua Opus.')
    parser.add_argument('--force', action='store_true', help='Render lại cả file đã có.')
    parser.add_argument(
        '--voices', default='',
        help='Danh sách tên giọng cách nhau bởi dấu phẩy (mặc định: tất cả).',
    )
    parser.add_argument(
        '--device', default=None,
        help="'cuda' / 'cpu' (mặc định: tự dò).",
    )
    return parser.parse_args(argv)


def _select_voices(spec: str) -> list[tuple[str, str]] | None:
    if not spec.strip():
        return None
    wanted = [v.strip().lower() for v in spec.split(',') if v.strip()]
    all_voices = get_available_preset_voices()
    chosen = [
        (label, name) for label, name in all_voices
        if name.lower() in wanted
    ]
    missing = set(wanted) - {name.lower() for _lab, name in chosen}
    if missing:
        raise SystemExit(f'❌ Không thấy giọng: {sorted(missing)}')
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.device:
        os.environ.setdefault('VIENEU_DEVICE', args.device)
    voices = _select_voices(args.voices)
    if voices is None:  # nạp model một lần trước khi liệt kê
        load_vieneu_model(args.device)
    entries = generate(voices=voices, make_ogg=not args.no_ogg, force=args.force)
    logger.info(f'🏁 Xong: {len(entries)} giọng trong {OUT_DIR}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
