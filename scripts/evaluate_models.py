"""Evaluate one or more classifiers on the same three-class image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix


CLASSES = ("Light", "Medium", "Dark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefix", help="Only evaluate files whose names start with this prefix")
    return parser.parse_args()


def load_images(data: Path, prefix: str | None) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label, class_name in enumerate(CLASSES):
        files = sorted(
            path
            for path in (data / class_name).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (prefix is None or path.name.startswith(prefix))
        )
        for path in files:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB").resize((224, 224))
            images.append(np.asarray(image, dtype=np.float32))
            labels.append(label)
        if not files:
            raise ValueError(f"No matching images for {class_name} in {data} (prefix={prefix!r})")
    return np.asarray(images), np.asarray(labels)


def main() -> None:
    args = parse_args()
    images, truth = load_images(args.data, args.prefix)
    for model_path in args.models:
        model = tf.keras.models.load_model(model_path)
        predicted = np.argmax(
            model.predict(images, batch_size=args.batch_size, verbose=0), axis=1
        )
        report = classification_report(
            truth,
            predicted,
            labels=range(len(CLASSES)),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )
        print(f"{model_path}: accuracy={report['accuracy']:.4f}")
        print(confusion_matrix(truth, predicted))


if __name__ == "__main__":
    main()
