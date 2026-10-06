import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time

MODEL_INPUT_SIZE = (224, 224)

# 1. โหลดโมเดลสองตัวที่เสริมกัน: ตัวแรกคงความแม่นยำของภาพเมล็ดเดี่ยว
# และตัวที่สองเรียนรู้จากภาพกองเมล็ดจริงที่แยก source ระหว่าง train/test
MODEL_SPECS = (
    ('coffee_roast_group_model.keras', 0.45),
    ('coffee_roast_real_group_model.keras', 0.55),
)
models = [
    (tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), filename)), weight)
    for filename, weight in MODEL_SPECS
]

# 2. กำหนดรายชื่อคลาสให้ตรงกับตอนเทรน
CLASS_NAMES = ['Light', 'Medium', 'Dark']


def warm_up_model():
    """Initialize TensorFlow's inference path before the first user request."""
    warmup_input = np.zeros((1, *MODEL_INPUT_SIZE, 3), dtype=np.float32)
    started_at = time.perf_counter()
    for model, _ in models:
        model.predict(warmup_input, verbose=0)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    print(f"Model warm-up completed in {elapsed_ms:.0f} ms")


warm_up_model()

# 3. ฟังก์ชันเตรียมภาพและทำนายผล
def predict_coffee(img):
    if img is None:
        return "กรุณาอัปโหลดรูปภาพ"
    
    img_pil = Image.fromarray(img).convert('RGB')
    img_pil = img_pil.resize(MODEL_INPUT_SIZE)
    
    # ดึงค่า 0-255 ตรงๆ แล้วส่งเข้าโมเดล
    img_array = np.array(img_pil, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = sum(
        model.predict(img_array, verbose=0)[0] * weight
        for model, weight in models
    )
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    return f"{CLASS_NAMES[predicted_index]} ({confidence:.2f}%)"

# 4. สร้างหน้า UI 
app = gr.Interface(
    fn=predict_coffee,
    inputs=gr.Image(label="อัปโหลดรูปภาพเมล็ดกาแฟ"),
    outputs=gr.Textbox(label="ระดับการคั่ว (ความมั่นใจ)"),
    title="Coffee Bean Roast Classifier",
    description="ทดสอบโมเดลจำแนกระดับการคั่วเมล็ดกาแฟ (Light, Medium, Dark)",
    flagging_mode="never"
)

# 5. สั่งรันเว็บเซิร์ฟเวอร์
if __name__ == "__main__":
    app.launch()
