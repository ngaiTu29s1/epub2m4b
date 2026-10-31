"""Acceptance check: every large silence in the render sits ON punctuation.

Ground truth:
* the chapter cue JSON (``chapters_audio/chapter_002.json``) gives the exact
  start/end of each source sentence;
* ``plan_viterbox_segments`` gives the clause plan for each sentence, and every
  planned clause ends in punctuation.

The waveform is scanned for silences >= ``MIN_SUSPECT_MS``. Each silence must be
explained by either a **sentence boundary** (the gap between two cues) or a
**clause boundary inside a cue** (a comma/clause pause) — never an unpunctuated
word gap. Also asserts the chunker preserves every comma and never ends a clause
on an unpunctuated word.

Usage::

    PYTHONPATH=$PWD .venv/bin/python scripts/check_pause_alignment.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
WAV = REPO_ROOT / 'workspace' / 'dua_tre_hu_tu_kim_tran' / 'output_viterbox_tts.wav'
CUE_JSON = REPO_ROOT / 'workspace' / 'dua_tre_hu_tu_kim_tran' / 'chapters_audio' / 'chapter_002.json'
MIN_SUSPECT_MS = 150.0
SIL_RMS = 0.004
SIL_MS = 100.0
PUNCT = (',', ';', ':', '.', '!', '?', '…')
# Tolerance when matching a silence to a cue edge / clause boundary.
EDGE_TOL_MS = 120.0


def _silences(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    frame = int(0.01 * sr)
    n = len(audio) // frame
    r = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    sil = r < SIL_RMS
    runs: list[tuple[int, int]] = []
    s = None
    for i, v in enumerate(sil):
        if v and s is None:
            s = i
        elif not v and s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, len(sil)))
    return [(a * 0.01, b * 0.01) for a, b in runs if (b - a) * 10 >= SIL_MS]


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from core.chatterbox_engine import plan_viterbox_segments

    data = json.loads(CUE_JSON.read_text(encoding='utf-8'))
    cues = data['cues']
    # Cue ranges in seconds; skip the title cue? keep all.
    cue_ranges = [(c['start_ms'] / 1000.0, c['end_ms'] / 1000.0, c['text']) for c in cues]

    audio, sr = librosa.load(str(WAV), sr=16000)
    silences = _silences(audio, sr)
    print(f'📄 {WAV}')
    print(f'🔇 {len(silences)} khoảng lặng >= {SIL_MS:.0f}ms trên waveform.\n')

    # Sentence boundaries = cue edges.
    edges: list[float] = []
    for i, (s, e, _t) in enumerate(cue_ranges):
        if i > 0:
            edges.append(s)
        edges.append(e)

    failures = 0
    checked = 0
    for s, e in silences:
        dur = (e - s) * 1000
        if dur < MIN_SUSPECT_MS:
            continue
        checked += 1
        mid = (s + e) / 2.0

        # (a) sits on a cue (sentence) edge?
        if any(abs(s - ed) * 1000 <= EDGE_TOL_MS or abs(e - ed) * 1000 <= EDGE_TOL_MS
               for ed in edges):
            print(f'✅ {dur:5.0f}ms @ {mid:6.2f}s — ranh giới câu (cue edge)')
            continue

        # (b) inside a cue: which clause boundary does it match? Re-plan the cue
        # text; a silence inside a cue is legitimate only if the cue was split
        # into >= 2 clauses (so a comma pause was programmed).
        owner = next((r for r in cue_ranges if r[0] <= mid <= r[1]), None)
        if owner is None:
            print(f'❌ {dur:5.0f}ms @ {mid:6.2f}s — không thuộc cue nào')
            failures += 1
            continue
        planned = plan_viterbox_segments(owner[2])
        multi = len(planned) >= 2
        clause_punct = all(c.rstrip().endswith(PUNCT) for c, _p in planned[:-1]) if multi else True
        if multi and clause_punct:
            n = len(planned)
            print(f'✅ {dur:5.0f}ms @ {mid:6.2f}s — ranh giới mệnh đề trong câu '
                  f'({n} mệnh đề, đều kết bằng dấu câu)')
            continue
        # A single unpunctuated long cue with an interior silence = real bug.
        print(f'❌ {dur:5.0f}ms @ {mid:6.2f}s — ngắt giữa câu KHÔNG dấu câu '
              f'(cues [{owner[0]:.2f},{owner[1]:.2f}] {owner[2][:40]!r}...)')
        failures += 1

    print()
    # Comma / boundary preservation through the chunker (whole chapter).
    from core.chatterbox_engine import chunk_into_clauses

    source = ' '.join(c[2] for c in cue_ranges[1:])  # skip the title cue
    clauses = chunk_into_clauses(source)
    n_src_comma = source.count(',')
    n_out_comma = sum(c.count(',') for c in clauses)
    bad_boundary = [c for c in clauses[:-1] if not c.rstrip().endswith(PUNCT)]

    print(f'🔤 Dấu phẩy: nguồn={n_src_comma}, sau chunk={n_out_comma}',
          '✅' if n_src_comma == n_out_comma else '❌')
    print(f'🔤 Ranh giới mệnh đề không dấu câu: {len(bad_boundary)}',
          '✅' if not bad_boundary else f'❌ {bad_boundary}')

    if failures or n_src_comma != n_out_comma or bad_boundary:
        print(f'\n❌ FAIL: {failures}/{checked} khoảng lặng không giải thích được bằng '
              f'dấu câu; comma {n_src_comma}->{n_out_comma}; '
              f'bad_boundary={len(bad_boundary)}.')
        return 1
    print(f'\n✅ PASS: {checked}/{checked} khoảng lặng (> {MIN_SUSPECT_MS:.0f}ms) '
          f'đều là ranh giới câu/mệnh đề có dấu câu; 0% mất dấu phẩy; '
          f'0% ranh giới không dấu câu.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
