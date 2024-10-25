import cv2
import numpy as np
import os
from ultralytics import YOLO

# Модель "Yolov8n" и видео
model_path = os.path.join(os.path.dirname(__file__), 'yolov8n.pt')
video_path = os.path.join(os.path.dirname(__file__), 'what.mp4')

# Загрузка модели YOLOv8
model = YOLO(model_path)

# Константы для перевода пикселей в метры (коэффициент)
PIXELS_TO_METERS = 0.05


# Функция для вычисления скорости
def calculate_speed(pixels_distance, video_fps):
    distance_meters = pixels_distance * PIXELS_TO_METERS
    speeds_m_s = distance_meters * video_fps
    return speeds_m_s


# Загрузка видео
cap = cv2.VideoCapture(video_path)

# Настройка параметров
previous_positions = {}
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Трекинг объектов
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            # Если обнаружен автомобиль (class_id == 2 для автомобилей)
            if class_id == 2:
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2

                # Рисуем прямоугольник вокруг объекта
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                # Проверка и запоминание предыдущей позиции
                if class_id not in previous_positions:
                    previous_positions[class_id] = (center_x, center_y)
                    continue

                # Расчет перемещения и скорости
                prev_x, prev_y = previous_positions[class_id]
                distance_pixels = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                speed_m_s = calculate_speed(distance_pixels, fps)

                # Обновление предыдущей позиции
                previous_positions[class_id] = (center_x, center_y)

                # Отображение скорости на кадре
                cv2.putText(frame, f'Speed: {speed_m_s:.2f} m/s', (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Отображение метки "Car" на кадре
                cv2.putText(frame, 'Car', (int(x_min), int(y_min) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Отображение кадра с трекингом и скоростью
    cv2.imshow('Car Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
