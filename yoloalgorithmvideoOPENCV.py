from ultralytics import YOLO
import cv2

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')

# Video dosyasını aç
cap = cv2.VideoCapture('arabavideo.mp4')

# Döngü ile videoyu işle
while cap.isOpened():
    ret, frame = cap.read()  # Her kareyi oku
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

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
