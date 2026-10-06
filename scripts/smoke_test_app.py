"""Run the app prediction path and measure warmed end-to-end latency."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import app

    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        if args.crop:
            image = image.crop(tuple(args.crop))
        array = np.asarray(image)
        timings = []
        for _ in range(args.runs):
            started_at = time.perf_counter()
            result = app.predict_coffee(array)
            timings.append((time.perf_counter() - started_at) * 1000)
        print(
            f"{image_path.name}: {result} | median={np.median(timings):.1f} ms, "
            f"max={max(timings):.1f} ms"
        )


if __name__ == "__main__":
    main()
