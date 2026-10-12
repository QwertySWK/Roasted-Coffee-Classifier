"""Export the two production Keras classifiers as one fully quantized TFLite model."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from train_rpi_tflite import export_int8, load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-model", type=Path, default=Path("src/coffee_roast_group_model.keras"))
    parser.add_argument(
        "--real-group-model",
        type=Path,
        default=Path("src/coffee_roast_real_group_model.keras"),
    )
    parser.add_argument(
        "--data", type=Path, default=Path("datasets/Coffee Bean Roast.v2i.folder")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("src/coffee_roast_rpi_ensemble_int8.tflite")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def renamed_model(path: Path, name: str) -> tf.keras.Model:
    loaded = tf.keras.models.load_model(path)
    return tf.keras.Model(loaded.input, loaded.output, name=name)


def main() -> None:
    args = parse_args()
    group = renamed_model(args.group_model, "synthetic_group_model")
    real_group = renamed_model(args.real_group_model, "real_group_model")
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    group_scores = tf.keras.layers.Rescaling(0.45, name="group_weight")(group(inputs))
    real_scores = tf.keras.layers.Rescaling(0.55, name="real_group_weight")(real_group(inputs))
    outputs = tf.keras.layers.Add(name="weighted_probabilities")([group_scores, real_scores])
    ensemble = tf.keras.Model(inputs, outputs, name="coffee_roast_ensemble")

    calibration = load_split(args.data, "train", args.batch_size, args.seed, True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_int8(ensemble, calibration.unbatch().batch(1), args.output)
    print(f"Saved: {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
