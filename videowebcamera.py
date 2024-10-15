import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # YOLOv8 nano modelini yükle

# Web kamerasını aç
cap = cv2.VideoCapture(0)

# Döngü ile videoyu işle
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin yap
    results = model(frame)

    # İlk sonucu al ve tespitleri çizdir
    annotated_frame = results[0].plot()

    # Sonuçları video üzerinde göster
    cv2.imshow('YOLOv8', annotated_frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
