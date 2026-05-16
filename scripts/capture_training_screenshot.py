from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from PIL import ImageGrab

ACTIVITY_LABELS = {"coding", "reading", "writing", "video", "browser", "messaging", "gaming", "idle", "unknown"}
EDUCATION_LABELS = {"programming", "math", "science", "language", "research", "exam_prep", "note_taking", "general"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one labeled screenshot for classifier training.")
    parser.add_argument("--activity", required=True, choices=sorted(ACTIVITY_LABELS))
    parser.add_argument("--education", required=True, choices=sorted(EDUCATION_LABELS))
    parser.add_argument("--output", type=Path, default=Path("data/screenshots"))
    args = parser.parse_args()

    folder = args.output / args.activity / args.education
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image = ImageGrab.grab(all_screens=False)
    path = folder / filename
    image.convert("RGB").save(path, quality=90, optimize=True)
    print({"saved": str(path.resolve())}, flush=True)


if __name__ == "__main__":
    main()
