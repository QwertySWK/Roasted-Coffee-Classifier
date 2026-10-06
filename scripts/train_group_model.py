"""Train and evaluate a three-class model on multi-bean images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


CLASSES = ("Light", "Medium", "Dark")
IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/grouped"))
    parser.add_argument(
        "--initial-model", type=Path, default=Path("src/coffee_roast_model.keras")
    )
    parser.add_argument("--model", type=Path, default=Path("src/coffee_roast_group_model.keras"))
    parser.add_argument(
        "--tflite", type=Path, default=Path("src/coffee_roast_group_model_quantized.tflite")
    )
    parser.add_argument("--report", type=Path, default=Path("reports/group_model_metrics.json"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_split(path: Path, batch_size: int, seed: int, shuffle: bool) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        class_names=list(CLASSES),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(initial_model_path: Path) -> tuple[tf.keras.Model, tf.keras.Model]:
    model = tf.keras.models.load_model(initial_model_path)
    base_model = next(
        layer for layer in model.layers if layer.name.startswith("mobilenetv2")
    )
    base_model.trainable = False
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, object]:
    truth: list[int] = []
    probabilities: list[np.ndarray] = []
    for images, labels in dataset:
        truth.extend(labels.numpy().tolist())
        probabilities.extend(model(images, training=False).numpy())
    predicted = np.argmax(np.asarray(probabilities), axis=1)
    return {
        "confusion_matrix": confusion_matrix(truth, predicted).tolist(),
        "classification_report": classification_report(
            truth,
            predicted,
            labels=range(len(CLASSES)),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    train = load_split(args.data / "train", args.batch_size, args.seed, True)
    validation = load_split(args.data / "val", args.batch_size, args.seed, False)
    test = load_split(args.data / "test", args.batch_size, args.seed, False)

    model, base_model = build_model(args.initial_model)
    compile_model(model, 2e-4)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    history = model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    compile_model(model, 1e-5)
    fine_tune_history = model.fit(
        train,
        validation_data=validation,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    args.tflite.write_bytes(converter.convert())

    report = {
        "classes": list(CLASSES),
        "source_split_policy": "directory splits are source-disjoint; see the dataset generation script",
        "train_history": history.history,
        "fine_tune_history": fine_tune_history.history,
        "validation": evaluate(model, validation),
        "test": evaluate(model, test),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
