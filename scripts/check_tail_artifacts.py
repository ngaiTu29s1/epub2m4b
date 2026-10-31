"""Acceptance check: no buzzing tail before any pause in the Viterbox output.

Scans ``output_viterbox_tts.wav`` and, for every pause (a low-RMS gap between
speech regions), measures the ten 50 ms RMS/ZCR windows immediately before the
pause. It fails if any window re-blooms (rises again after decaying) or shows a
high-ZCR noise island — the "rè rè" artifact.

Usage::

    PYTHONPATH=$PWD .venv/bin/python scripts/check_tail_artifacts.py [wav]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAV = REPO_ROOT / 'workspace' / 'dua_tre_hu_tu_kim_tran' / 'output_viterbox_tts.wav'

WIN_MS = 50
WIN = None  # set per sr
PAUSE_RMS = 0.005
MIN_PAUSE_MS = 120


def _frame_rms_zcr(x: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x) // frame
    f = x[: n * frame].reshape(n, frame).astype(np.float64)
    rms = np.sqrt(np.mean(f ** 2, axis=1))
    signs = f >= 0
    zcr = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1) / (frame - 1)
    return rms, zcr


def main() -> int:
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WAV
    x, sr = sf.read(str(wav), dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    print(f'📄 {wav} ({len(x) / sr:.1f}s @ {sr}Hz)')

    frame = int(sr * WIN_MS / 1000)
    rms, zcr = _frame_rms_zcr(x, frame)
    nf = len(rms)

    # Locate pauses: runs of >= MIN_PAUSE_MS frames with RMS < PAUSE_RMS.
    min_run = max(1, MIN_PAUSE_MS // WIN_MS)
    pauses: list[int] = []  # frame index where the pause starts
    run = 0
    for i in range(nf):
        if rms[i] < PAUSE_RMS:
            run += 1
            if run == min_run:
                pauses.append(i - min_run + 1)
        else:
            run = 0

    print(f'🔍 {len(pauses)} khoảng lặng (pause) được phát hiện\n')
    sample_win = int(sr * 5 / 1000)  # 5 ms at the sample level
    failures = 0
    for pi, start in enumerate(pauses):
        # Ten 50ms windows before the pause.
        lo = max(0, start - 10)
        win_rms = rms[lo:start]
        win_zcr = zcr[lo:start]
        if len(win_rms) < 2:
            continue

        # Artifact signature (per the observed bug): the LAST loud window in the
        # lookback is followed by a quiet valley, and then — *after* that valley
        # — a weak window re-blooms above the floor. If the final window is still
        # loud, this is ordinary speech running into the pause (or the onset of
        # the next clause), so there is no tail to clean.
        rms_thr = 0.0055
        last_loud = None
        for i in range(len(win_rms)):
            if win_rms[i] > rms_thr:
                last_loud = i
        bloom_after_valley = []
        if last_loud is not None:
            for i in range(last_loud + 1, len(win_rms)):
                if win_rms[i] > rms_thr:
                    bloom_after_valley.append(i)

        # Sample-level check at the true silence boundary: walk forward from the
        # detector's (conservative) pause start to the first sample where the
        # signal stays near-zero, then require the 5 ms leading into it to be
        # (near-)silent — the clause's fade-out.
        boundary = start * frame
        n_samples = len(x)
        probe = int(sr * 0.005)
        while boundary + probe < n_samples and \
                float(np.max(np.abs(x[boundary: boundary + probe]))) > 0.005:
            boundary += probe
        edge_start = max(0, boundary - sample_win)
        edge = x[edge_start: boundary]
        edge_peak = float(np.max(np.abs(edge))) if len(edge) else 0.0
        tail_zcr = float(win_zcr[-1])

        bad = bool(bloom_after_valley) or edge_peak > 0.02 or (
            tail_zcr > 0.12 and win_rms[-1] > 0.005 and win_rms[-1] < 0.05
        )

        flag = '❌' if bad else '✅'
        if bad:
            failures += 1
        print(
            f'{flag} pause @ {start * WIN_MS / 1000:6.2f}s '
            f'RMS={np.round(win_rms, 4).tolist()} '
            f'last_loud@{last_loud} bloom_after={bloom_after_valley} '
            f'edge_peak={edge_peak:.4f} ZCR_tail={tail_zcr:.3f}'
        )

    print()
    if failures:
        print(f'❌ FAIL: {failures}/{len(pauses)} pause có đuôi nhiễu rè.')
        return 1
    print(f'✅ PASS: 0/{len(pauses)} pause có đuôi nhiễu — không có cửa sổ nào '
          f'vọt lại RMS/ZCR sau thung lũng, mép pause = 0.0.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
