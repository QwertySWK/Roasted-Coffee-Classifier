"""Select a correctly classified sample nearest the median confidence."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--class-index", type=int, required=True, choices=(0, 1, 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.source.glob("source_*"))
    arrays = []
    for path in files:
        with Image.open(path) as opened:
            arrays.append(np.asarray(opened.convert("RGB").resize((224, 224)), dtype=np.float32))
    images = np.asarray(arrays)
    specs = (
        (Path("src/coffee_roast_group_model.keras"), 0.45),
        (Path("src/coffee_roast_real_group_model.keras"), 0.55),
    )
    probabilities = sum(
        tf.keras.models.load_model(path).predict(images, batch_size=32, verbose=0) * weight
        for path, weight in specs
    )
    predicted = np.argmax(probabilities, axis=1)
    candidates = np.flatnonzero(predicted == args.class_index)
    if not len(candidates):
        raise RuntimeError("No correctly classified samples found")
    confidences = probabilities[candidates, args.class_index]
    median = float(np.median(confidences))
    selected = int(candidates[np.argmin(np.abs(confidences - median))])
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(files[selected], args.destination)
    print(f"selected {files[selected]} at {probabilities[selected, args.class_index] * 100:.2f}%")


if __name__ == "__main__":
    main()
