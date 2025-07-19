
import cv2
from ultralytics import YOLO

VIDEO_PATH = "video_2025-07-19_21-07-37.mp4"
TARGET_CLASSES = ["backpack", "handbag", "laptop"]
CONF_THRESHOLD = 0.5

# Загружаем модель YOLO11x (предобучена на COCO)
model = YOLO("yolo11x.pt")
model.to("cuda")  # перенос на GPU

# Получаем список всех классов COCO
class_names = model.names

# Открываем видео
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Видео {VIDEO_PATH} не найдено!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция объектов на кадре
    results = model.predict(frame, device=0, conf=CONF_THRESHOLD, verbose=False)

    detected_objects = []
    annotated_frame = frame.copy()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            cls_name = class_names[cls_id]
            conf = float(box.conf)

            # Фильтруем по классам
            if cls_name in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_objects.append(f"{cls_name} ({conf:.2f})")
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{cls_name} {conf:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Выводим список объектов в верхнем левом углу
    y0 = 20
    for obj_text in detected_objects:
        cv2.putText(annotated_frame, obj_text, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y0 += 25

    # Показ кадра
    cv2.imshow("YOLO11x Video Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
