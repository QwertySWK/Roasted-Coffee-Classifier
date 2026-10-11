"""Find a fixed two-model blend that balances group and single-bean accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from evaluate_models import load_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", type=Path)
    parser.add_argument("model_b", type=Path)
    parser.add_argument("--single-data", type=Path, default=Path("datasets/test"))
    parser.add_argument(
        "--group-data", type=Path, default=Path("datasets/real-group-crops/test")
    )
    parser.add_argument("--single-validation", type=Path, default=Path("datasets/grouped/val"))
    parser.add_argument("--group-validation", type=Path, default=Path("datasets/real-group-crops/val"))
    parser.add_argument("--config", type=Path, default=Path("src/ensemble_config.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/ensemble_metrics.json"))
    return parser.parse_args()


def probabilities(model: tf.keras.Model, images: np.ndarray) -> np.ndarray:
    return model.predict(images, batch_size=32, verbose=0)


def main() -> None:
    args = parse_args()
    single_images, single_truth = load_images(args.single_validation, "single_")
    group_images, group_truth = load_images(args.group_validation, "source_")
    model_a = tf.keras.models.load_model(args.model_a)
    model_b = tf.keras.models.load_model(args.model_b)
    single_a = probabilities(model_a, single_images)
    single_b = probabilities(model_b, single_images)
    group_a = probabilities(model_a, group_images)
    group_b = probabilities(model_b, group_images)

    best_score = -1.0
    selected_weight = 0.0
    for weight_b in np.linspace(0, 1, 21):
        single = np.argmax(single_a * (1 - weight_b) + single_b * weight_b, axis=1)
        group = np.argmax(group_a * (1 - weight_b) + group_b * weight_b, axis=1)
        single_accuracy = float(np.mean(single == single_truth))
        group_accuracy = float(np.mean(group == group_truth))
        score = (single_accuracy + group_accuracy) / 2
        if score > best_score:
            best_score = score
            selected_weight = float(weight_b)
        print(
            f"weight_b={weight_b:.2f} single={single_accuracy:.4f} "
            f"group={group_accuracy:.4f} mean={(single_accuracy + group_accuracy) / 2:.4f}"
        )

    # Freeze the weight before reading either held-out test set.
    weight_b = selected_weight
    single_images, single_truth = load_images(args.single_data, None)
    group_images, group_truth = load_images(args.group_data, "source_")
    single_a = probabilities(model_a, single_images)
    single_b = probabilities(model_b, single_images)
    group_a = probabilities(model_a, group_images)
    group_b = probabilities(model_b, group_images)
    single = np.argmax(single_a * (1 - weight_b) + single_b * weight_b, axis=1)
    group = np.argmax(group_a * (1 - weight_b) + group_b * weight_b, axis=1)
    print(f"selected weight_b={weight_b:.2f}")
    print("single confusion matrix:")
    print(confusion_matrix(single_truth, single))
    print("group confusion matrix:")
    print(confusion_matrix(group_truth, group))
    specs = [
        {"path": str(args.model_a.resolve()), "weight": 1 - weight_b},
        {"path": str(args.model_b.resolve()), "weight": weight_b},
    ]
    report = {
        "classes": ["Light", "Medium", "Dark"],
        "models": specs,
        "selection_policy": "maximize mean single/group validation accuracy; test used only after selection",
        "validation_mean_accuracy": best_score,
        "single_bean_test": {"images": len(single_truth), "accuracy": float(np.mean(single == single_truth)), "confusion_matrix": confusion_matrix(single_truth, single).tolist()},
        "source_held_out_real_group_test": {"images": len(group_truth), "accuracy": float(np.mean(group == group_truth)), "confusion_matrix": confusion_matrix(group_truth, group).tolist()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps({"classes": report["classes"], "models": specs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
