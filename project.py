import cv2
from ultralytics import YOLO
import mysql.connector
import paho.mqtt.client as mqtt
from datetime import datetime

# === YOLO MODEL ===
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("IMG_3583.mp4")

# === MySQL SETUP ===
db = mysql.connector.connect(
    host="192.168.1.1",        # IP ของ MySQL server (เช่น "192.168.1.10")
    user="user1",        # เปลี่ยนเป็น user ของ Shane
    password="1234",# เปลี่ยนเป็น password
    database="people_counter"
)
cursor = db.cursor()

# === MQTT SETUP ===
MQTT_BROKER = "localhost"    # ถ้าใช้ Mosquitto บน PC ตัวเอง ใช้ "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_IN = "people/in"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# === COUNTER ===
count_in = 0
line_y = 760
prev_ids = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])  # detect เฉพาะคน (class 0)

    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # ตรวจสอบการข้ามเส้น
            if track_id in prev_ids:
                prev_y = prev_ids[track_id]
                if prev_y < line_y and cy >= line_y:  # เดินลง (IN)
                    count_in += 1
                    now = datetime.now()

                    # === Save to MySQL ===
                    cursor.execute(
                        "INSERT INTO people_log (direction, count, timestamp) VALUES (%s, %s, %s)",
                        ("IN", count_in, now)
                    )
                    db.commit()

                    # === Publish to MQTT ===
                    client.publish(MQTT_TOPIC_IN, f"{count_in}")

            prev_ids[track_id] = cy

            # วาดกรอบ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # วาดเส้นนับ
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    # แสดงจำนวน
    cv2.putText(frame, f"IN: {count_in}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("People Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
db.close()
client.disconnect()
