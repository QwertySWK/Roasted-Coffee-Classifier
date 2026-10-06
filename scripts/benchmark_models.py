"""Compare model predictions and warmed inference latency on one image."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


CLASSES = ("Light", "Medium", "Dark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    if args.crop:
        image = image.crop(tuple(args.crop))
    array = np.expand_dims(
        np.asarray(image.resize((224, 224)), dtype=np.float32), axis=0
    )

    for model_path in args.models:
        model = tf.keras.models.load_model(model_path)
        model.predict(array, verbose=0)  # warm-up
        timings = []
        for _ in range(args.runs):
            started_at = time.perf_counter()
            probabilities = model.predict(array, verbose=0)[0]
            timings.append((time.perf_counter() - started_at) * 1000)
        predicted = int(np.argmax(probabilities))
        probability_text = ", ".join(
            f"{name}={value * 100:.2f}%" for name, value in zip(CLASSES, probabilities)
        )
        print(
            f"{model_path}: {CLASSES[predicted]} ({probabilities[predicted] * 100:.2f}%) | "
            f"{probability_text} | median={np.median(timings):.1f} ms, max={max(timings):.1f} ms"
        )


if __name__ == "__main__":
    main()
