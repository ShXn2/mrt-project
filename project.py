from ultralytics import YOLO
import cv2
import paho.mqtt.client as mqtt
import mysql.connector
from datetime import datetime

# MQTT
broker = "broker.hivemq.com"  # หรือ 'localhost' ถ้าใช้ Mosquitto
client = mqtt.Client()
client.connect(broker, 1883, 60)

# MySQL
db = mysql.connector.connect(
    host="localhost",
    user="your_user",
    password="your_password",
    database="people_counter"
)
cursor = db.cursor()

# YOLO
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("your_video.mp4")

line_y = 760
count_in = 0
track_memory = {}
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0], conf=0.4)
    frame_copy = frame.copy()
    cv2.line(frame_copy, (840, line_y), (1250, line_y), (0, 0, 255), 3)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for track_id, box in zip(ids, boxes):
            x1, y1, x2, y2 = box.astype(int)
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            if track_id not in track_memory:
                track_memory[track_id] = [cy]
            else:
                track_memory[track_id].append(cy)
                if len(track_memory[track_id]) >= 2 and track_id not in counted_ids:
                    dy = cy - track_memory[track_id][-2]
                    if track_memory[track_id][-2] > line_y >= cy and dy < 0:
                        count_in += 1
                        counted_ids.add(track_id)

                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        cursor.execute("INSERT INTO people_log (direction, count, timestamp) VALUES (%s, %s, %s)",
                                       ("IN", count_in, now))
                        db.commit()

                        client.publish("people/in", f"{count_in}")

    cv2.putText(frame_copy, f"IN: {count_in}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    cv2.imshow("Counter", frame_copy)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.disconnect()
cursor.close()
db.close()
