import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import time


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "src" / "coffee_roast_model_v3.keras"

MODEL_INPUT_SIZE = (224, 224)

CLASS_NAMES = [
    "Light",
    "Medium",
    "Dark"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading V3 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Model:", MODEL_PATH)
print("Class order:", CLASS_NAMES)


# ============================================================
# PREDICT
# ============================================================

def predict_coffee(img):

    if img is None:
        return "กรุณาอัปโหลดรูปภาพ"

    # Gradio image -> PIL
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("RGB")

    # Resize
    img = img.resize(MODEL_INPUT_SIZE)

    # IMPORTANT:
    # V3 มี Rescaling อยู่ในโมเดล
    # ดังนั้นส่งค่า pixel 0-255 เข้าไปตรง ๆ
    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Predict
    start = time.perf_counter()

    prediction = model.predict(
        img_array,
        verbose=0
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    probabilities = prediction[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        float(probabilities[predicted_index])
        * 100
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = (
        f"ระดับการคั่ว: {predicted_class}\n"
        f"ความมั่นใจ: {confidence:.2f}%\n\n"
        f"Light:   {probabilities[0] * 100:.2f}%\n"
        f"Medium:  {probabilities[1] * 100:.2f}%\n"
        f"Dark:    {probabilities[2] * 100:.2f}%\n\n"
        f"เวลาในการประมวลผล: {elapsed_ms:.2f} ms"
    )

    return result


# ============================================================
# GRADIO APP
# ============================================================

app = gr.Interface(
    fn=predict_coffee,

    inputs=gr.Image(
        type="numpy",
        label="แนบรูปเมล็ดกาแฟ"
    ),

    outputs=gr.Textbox(
        label="ผลการทำนาย"
    ),

    title="Coffee Bean Roast Classifier V3",

    description=(
        "อัปโหลดรูปเมล็ดกาแฟ "
        "เพื่อจำแนกระดับการคั่ว: "
        "Light / Medium / Dark"
    ),

    flagging_mode="never"
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.launch(
        server_name="127.0.0.1",
        server_port=7860
    )