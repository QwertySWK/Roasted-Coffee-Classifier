import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import time
import json
from pathlib import Path
from threading import Lock

MODEL_INPUT_SIZE = (224, 224)

# Small image requests benefit from a bounded CPU pool rather than scheduling
# work across every logical core. Configure before loading any model.
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 1. โหลดโมเดลสองตัวที่เสริมกัน: ตัวแรกคงความแม่นยำของภาพเมล็ดเดี่ยว
# และตัวที่สองเรียนรู้จากภาพกองเมล็ดจริงที่แยก source ระหว่าง train/test
MODEL_SPECS = (
    ('coffee_roast_group_model.keras', 0.45),
    ('coffee_roast_real_group_model.keras', 0.55),
)
config_path = Path(__file__).with_name('ensemble_config.json')
if config_path.exists():
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['classes'] != ['Light', 'Medium', 'Dark']:
        raise ValueError('Ensemble class order must be Light, Medium, Dark')
    MODEL_SPECS = tuple((item['path'], item['weight']) for item in config['models'])
if any(weight < 0 for _, weight in MODEL_SPECS) or not np.isclose(sum(weight for _, weight in MODEL_SPECS), 1):
    raise ValueError('Ensemble weights must be nonnegative and sum to 1')
models = []
interpreters = []
for filename, weight in MODEL_SPECS:
    if weight == 0:
        continue
    model_path = Path(os.path.join(os.path.dirname(__file__), filename))
    lite_path = model_path.with_name(model_path.stem + '_quantized.tflite')
    if lite_path.exists():
        interpreter = tf.lite.Interpreter(model_path=str(lite_path), num_threads=2)
        interpreter.allocate_tensors()
        interpreters.append((interpreter, interpreter.get_input_details()[0]['index'],
                             interpreter.get_output_details()[0]['index'], weight))
    else:
        models.append((tf.keras.models.load_model(model_path), weight))

# 2. กำหนดรายชื่อคลาสให้ตรงกับตอนเทรน
CLASS_NAMES = ['Light', 'Medium', 'Dark']
inference_lock = Lock()


@tf.function(input_signature=[tf.TensorSpec((1, *MODEL_INPUT_SIZE, 3), tf.float32)])
def infer_coffee(images):
    """Run the ensemble as one graph without Keras predict's batch machinery."""
    return tf.add_n([
        model(images, training=False)[0] * weight
        for model, weight in models
    ])


def inference(images):
    # Lite interpreters share mutable buffers; serialize concurrent requests.
    with inference_lock:
        predictions = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        for interpreter, input_index, output_index, weight in interpreters:
            interpreter.set_tensor(input_index, images)
            interpreter.invoke()
            predictions += interpreter.get_tensor(output_index)[0] * weight
        if models:
            predictions += infer_coffee(tf.convert_to_tensor(images)).numpy()
        return predictions


def warm_up_model():
    """Initialize TensorFlow's inference path before the first user request."""
    warmup_input = np.zeros((1, *MODEL_INPUT_SIZE, 3), dtype=np.float32)
    started_at = time.perf_counter()
    inference(warmup_input)
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
    
    predictions = inference(img_array)
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
