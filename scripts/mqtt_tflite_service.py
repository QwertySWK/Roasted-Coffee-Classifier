"""MQTT image-classification service for Raspberry Pi 5.

Subscribe payload formats:
1. Raw JPEG/PNG bytes.
2. JSON: {"image_base64": "...", "request_id": "optional"}

The service publishes a JSON prediction to the configured output topic.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import paho.mqtt.client as mqtt

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
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("src/coffee_roast_rpi_ensemble_int8.tflite"),
    )
    parser.add_argument("--labels", type=Path, default=Path("src/coffee_roast_labels.json"))
    parser.add_argument("--host", default=os.getenv("MQTT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--input-topic", default=os.getenv("MQTT_INPUT_TOPIC", "coffee/image"))
    parser.add_argument("--output-topic", default=os.getenv("MQTT_OUTPUT_TOPIC", "coffee/prediction"))
    parser.add_argument("--username", default=os.getenv("MQTT_USERNAME"))
    parser.add_argument("--password", default=os.getenv("MQTT_PASSWORD"))
    parser.add_argument("--client-id", default="coffee-roast-rpi5")
    parser.add_argument("--qos", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args()


class RoastClassifier:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=4)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        _, self.height, self.width, self.channels = self.input["shape"]
        if self.channels != 3:
            raise ValueError(f"Expected RGB input, got {self.input['shape']}")

    @staticmethod
    def _quantize(values: np.ndarray, info: dict) -> np.ndarray:
        dtype = info["dtype"]
        if np.issubdtype(dtype, np.floating):
            return values.astype(dtype)
        scale, zero_point = info["quantization"]
        if scale <= 0:
            raise ValueError("Invalid TFLite input quantization scale")
        limits = np.iinfo(dtype)
        return np.clip(np.rint(values / scale + zero_point), limits.min, limits.max).astype(dtype)

    @staticmethod
    def _dequantize(values: np.ndarray, info: dict) -> np.ndarray:
        if np.issubdtype(info["dtype"], np.floating):
            return values.astype(np.float32)
        scale, zero_point = info["quantization"]
        return (values.astype(np.float32) - zero_point) * scale

    def predict(self, image_bytes: bytes) -> dict[str, object]:
        started = time.perf_counter()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((int(self.width), int(self.height)), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32)[None, ...]
        self.interpreter.set_tensor(self.input["index"], self._quantize(values, self.input))
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output["index"])[0]
        scores = self._dequantize(raw, self.output)
        best = int(np.argmax(scores))
        return {
            "class": self.labels[best],
            "confidence": round(float(scores[best]), 6),
            "probabilities": {
                label: round(float(score), 6) for label, score in zip(self.labels, scores)
            },
            "inference_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def decode_payload(payload: bytes) -> tuple[bytes, str | None]:
    if payload.lstrip().startswith(b"{"):
        message = json.loads(payload.decode("utf-8"))
        return base64.b64decode(message["image_base64"], validate=True), message.get("request_id")
    return payload, None


def main() -> None:
    args = parse_args()
    classifier = RoastClassifier(args.model, args.labels)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=args.client_id)
    if args.username:
        client.username_pw_set(args.username, args.password)

    def on_connect(client, _userdata, _flags, reason_code, _properties):
        if reason_code != 0:
            raise ConnectionError(f"MQTT connection failed: {reason_code}")
        client.subscribe(args.input_topic, qos=args.qos)
        print(f"Subscribed to {args.input_topic} on {args.host}:{args.port}")

    def on_message(client, _userdata, message):
        response: dict[str, object]
        request_id: str | None = None
        try:
            image_bytes, request_id = decode_payload(message.payload)
            response = classifier.predict(image_bytes)
            response["status"] = "ok"
        except Exception as error:
            response = {"status": "error", "error": str(error)}
        if request_id is not None:
            response["request_id"] = request_id
        client.publish(
            args.output_topic,
            json.dumps(response, ensure_ascii=False),
            qos=args.qos,
        )
        print(json.dumps(response, ensure_ascii=False))

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
