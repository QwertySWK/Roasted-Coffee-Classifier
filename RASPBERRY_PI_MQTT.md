# Raspberry Pi 5 + MQTT deployment

โมเดลทำ inference ใน Python service บน Raspberry Pi 5 โดย service เชื่อมต่อกับ
MQTT broker เพื่อรับภาพและส่งผลลัพธ์กลับ ไม่ได้โหลดโมเดลเข้าไปทำงานภายใน brokerโดยตรง

## ไฟล์ที่ต้องคัดลอกไปยัง Pi

- `src/coffee_roast_rpi_ensemble_int8.tflite` (แนะนำสำหรับภาพกองเมล็ดจริง)
- `src/coffee_roast_rpi_int8.tflite` (โมเดลที่ train จากชุดข้อมูลใหม่)
- `src/coffee_roast_labels.json`
- `src/requirements-rpi.txt`
- `scripts/mqtt_tflite_service.py`

## ติดตั้งและรัน

```bash
sudo apt update
sudo apt install -y python3-venv libjpeg-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements-rpi.txt
python scripts/mqtt_tflite_service.py --host localhost
```

ถ้า broker ต้องใช้ username/password ให้ตั้งค่า environment variables ก่อนรัน:

```bash
export MQTT_HOST=192.168.1.10
export MQTT_PORT=1883
export MQTT_USERNAME=coffee-device
export MQTT_PASSWORD='change-me'
python scripts/mqtt_tflite_service.py
```

ค่าเริ่มต้น subscribe ที่ `coffee/image` และ publish ที่ `coffee/prediction`

## รูปแบบ input และ output

ส่งไฟล์ JPEG/PNG เป็น raw MQTT payload ได้โดยตรง:

```bash
mosquitto_pub -h RASPBERRY_PI_IP -t coffee/image -q 1 -f test.jpg
mosquitto_sub -h RASPBERRY_PI_IP -t coffee/prediction -q 1 -v
```

หรือส่ง JSON ที่มี `image_base64` และ `request_id` ก็ได้ ผลลัพธ์เป็น JSON ที่มี
`class`, `confidence`, `probabilities`, `inference_ms` และ `request_id`

สำหรับ production ควรเปิด TLS และ authentication ที่ broker และจำกัดขนาด MQTT
message ให้พอดีกับภาพจากกล้อง
