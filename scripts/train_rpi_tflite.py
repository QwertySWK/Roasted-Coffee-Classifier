"""Train the coffee-roast classifier and export a Raspberry Pi friendly TFLite model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


CLASSES = ("Light", "Medium", "Dark")
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/Coffee Bean Roast.v2i.folder"),
        help="Dataset root containing train, valid, and test directories",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("src"))
    parser.add_argument("--report", type=Path, default=Path("reports/rpi_model_metrics.json"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help="Optional compatible .keras model used instead of ImageNet initialization",
    )
    return parser.parse_args()


def load_split(
    data_root: Path,
    split: str,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """Load only roasted classes; the Green directory is intentionally ignored."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_root / split,
        labels="inferred",
        class_names=list(CLASSES),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )
    return dataset.prefetch(AUTOTUNE)


def build_model(initial_model: Path | None = None) -> tuple[tf.keras.Model, tf.keras.Model]:
    if initial_model is not None:
        model = tf.keras.models.load_model(initial_model)
        if model.output_shape[-1] != len(CLASSES):
            raise ValueError(f"Initial model must have {len(CLASSES)} outputs")
        base_model = next(
            layer for layer in model.layers if layer.name.lower().startswith("mobilenetv2")
        )
        return model, base_model

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="normalize")(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="roast")(x)
    return tf.keras.Model(inputs, outputs, name="coffee_roast_mobilenetv2"), base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def evaluate_keras(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, object]:
    truth: list[int] = []
    scores: list[np.ndarray] = []
    for images, labels in dataset:
        truth.extend(labels.numpy().tolist())
        scores.extend(model(images, training=False).numpy())
    predicted = np.argmax(np.asarray(scores), axis=1)
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


def representative_samples(dataset: tf.data.Dataset, limit: int = 200):
    emitted = 0
    for images, _ in dataset:
        for image in images:
            yield [tf.expand_dims(tf.cast(image, tf.float32), 0)]
            emitted += 1
            if emitted >= limit:
                return


def export_int8(model: tf.keras.Model, calibration: tf.data.Dataset, output: Path) -> None:
    """Export full INT8 internals with uint8 I/O for easy image handling on the Pi."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_samples(calibration)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    output.write_bytes(converter.convert())


def evaluate_tflite(model_path: Path, dataset: tf.data.Dataset) -> dict[str, object]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    in_scale, in_zero = input_info["quantization"]
    out_scale, out_zero = output_info["quantization"]
    truth: list[int] = []
    predicted: list[int] = []

    for images, labels in dataset:
        for image, label in zip(images.numpy(), labels.numpy()):
            quantized = np.clip(np.rint(image / in_scale + in_zero), 0, 255).astype(np.uint8)
            interpreter.set_tensor(input_info["index"], quantized[None, ...])
            interpreter.invoke()
            output = interpreter.get_tensor(output_info["index"])[0]
            scores = (output.astype(np.float32) - out_zero) * out_scale
            truth.append(int(label))
            predicted.append(int(np.argmax(scores)))

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
        "input": {
            "shape": input_info["shape"].tolist(),
            "dtype": str(input_info["dtype"].__name__),
            "quantization": [float(in_scale), int(in_zero)],
        },
        "output": {
            "shape": output_info["shape"].tolist(),
            "dtype": str(output_info["dtype"].__name__),
            "quantization": [float(out_scale), int(out_zero)],
        },
    }


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    train = load_split(args.data, "train", args.batch_size, args.seed, True)
    validation = load_split(args.data, "valid", args.batch_size, args.seed, False)
    test = load_split(args.data, "test", args.batch_size, args.seed, False)

    model, base_model = build_model(args.initial_model)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    compile_model(model, 3e-4)
    head_history = model.fit(
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
    fine_history = model.fit(
        train,
        validation_data=validation,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
        verbose=2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = args.output_dir / "coffee_roast_rpi.keras"
    tflite_path = args.output_dir / "coffee_roast_rpi_int8.tflite"
    labels_path = args.output_dir / "coffee_roast_labels.json"
    model.save(keras_path)
    export_int8(model, train.unbatch().batch(1), tflite_path)
    labels_path.write_text(json.dumps(list(CLASSES), indent=2), encoding="utf-8")

    report = {
        "classes": list(CLASSES),
        "dataset": str(args.data),
        "split_policy": "train/valid/test directories; Green excluded from roasted classes",
        "head_history": head_history.history,
        "fine_tune_history": fine_history.history,
        "keras_test": evaluate_keras(model, test),
        "tflite_test": evaluate_tflite(tflite_path, test),
        "artifacts": {
            "keras": str(keras_path),
            "tflite": str(tflite_path),
            "labels": str(labels_path),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    accuracy = report["tflite_test"]["classification_report"]["accuracy"]
    print(f"TFLite test accuracy: {accuracy:.4f}")
    print(f"Saved: {tflite_path} ({tflite_path.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
