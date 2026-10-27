"""Harvest full voice dataset for vui_ve (~20-25 clean minutes)."""
import sys
from pathlib import Path

# Running `python scripts/harvest_full.py` puts `scripts/` on sys.path, not the
# project root, so `core` would not be importable without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from core.voice_harvester import harvest_voice_from_youtube

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    result = harvest_voice_from_youtube(
        voice_id="vui_ve",
        query_or_url="https://www.youtube.com/@VuiVe",
        voices_root=Path("voices"),
        target_clean_minutes=25.0,
        max_videos=8,
    )
    total_clean_minutes = result.total_clean_sec / 60.0
    golden_rel = ''
    if result.profile_path.exists():
        import json

        profile = json.loads(result.profile_path.read_text(encoding='utf-8'))
        golden_rel = profile.get('best_golden_ref_wav', '')
        processed_video_ids = profile.get('processed_video_ids', [])
    else:
        processed_video_ids = []
    print("=== HARVEST COMPLETE ===")
    print(f"Total clean minutes: {total_clean_minutes:.2f}")
    print(f"Num clips: {len(result.kept_clips)}")
    print(f"Golden refs: {len(result.golden_refs)}")
    print(f"Videos processed: {processed_video_ids}")
    print(f"Best golden ref: {golden_rel or result.golden_refs[0].clip_path if result.golden_refs else ''}")
    print(f"Voice dir: {result.voice_dir}")
