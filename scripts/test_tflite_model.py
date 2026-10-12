"""Inspect a TFLite model and run predictions on local images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, default=Path("src/coffee_roast_rpi_int8.tflite"))
    parser.add_argument("--labels", type=Path, default=Path("src/coffee_roast_labels.json"))
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def quantize(values: np.ndarray, info: dict) -> np.ndarray:
    if np.issubdtype(info["dtype"], np.floating):
        return values.astype(info["dtype"])
    scale, zero_point = info["quantization"]
    limits = np.iinfo(info["dtype"])
    return np.clip(np.rint(values / scale + zero_point), limits.min, limits.max).astype(info["dtype"])


def dequantize(values: np.ndarray, info: dict) -> np.ndarray:
    if np.issubdtype(info["dtype"], np.floating):
        return values.astype(np.float32)
    scale, zero_point = info["quantization"]
    return (values.astype(np.float32) - zero_point) * scale


def main() -> None:
    args = parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    interpreter = Interpreter(model_path=str(args.model), num_threads=args.threads)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    _, height, width, _ = input_info["shape"]
    print(
        f"input={input_info['shape'].tolist()} {input_info['dtype'].__name__} "
        f"quantization={input_info['quantization']}"
    )
    print(
        f"output={output_info['shape'].tolist()} {output_info['dtype'].__name__} "
        f"quantization={output_info['quantization']}"
    )

    for image_path in args.images:
        image = Image.open(image_path).convert("RGB").resize((int(width), int(height)))
        values = np.asarray(image, dtype=np.float32)[None, ...]
        interpreter.set_tensor(input_info["index"], quantize(values, input_info))
        started = time.perf_counter()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - started) * 1000
        scores = dequantize(interpreter.get_tensor(output_info["index"])[0], output_info)
        best = int(np.argmax(scores))
        details = ", ".join(f"{label}={score:.4f}" for label, score in zip(labels, scores))
        print(f"{image_path}: {labels[best]} ({scores[best]:.4f}), {elapsed_ms:.1f} ms | {details}")


if __name__ == "__main__":
    main()
