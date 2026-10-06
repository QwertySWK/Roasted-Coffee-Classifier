import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# 1. โหลดโมเดล
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'coffee_roast_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# 2. กำหนดรายชื่อคลาสให้ตรงกับตอนเทรน
CLASS_NAMES = ['Light', 'Medium', 'Dark']

# 3. ฟังก์ชันเตรียมภาพและทำนายผล
def predict_coffee(img):
    if img is None:
        return "กรุณาอัปโหลดรูปภาพ"
    
    img_pil = Image.fromarray(img).convert('RGB')
    img_pil = img_pil.resize((224, 224))
    
    # ดึงค่า 0-255 ตรงๆ แล้วส่งเข้าโมเดล
    img_array = np.array(img_pil, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
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