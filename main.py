import cv2
import torch

# YOLO v5 modelini yükle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Video dosyasını aç
video_path = 'arabavideo.mp4'  # Video dosyanızın yolunu buraya yazın
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()  # Her kareyi oku
    if not ret:
        break  # Video sona erdiyse döngüyü kır

    # Nesne tespiti yap
    results = model(frame)

    # Sonuçları göster
    results.render()  # Tespit edilen nesneleri görüntü üzerinde çizin

    # Görüntüyü göster
    cv2.imshow('Nesne Tespiti', frame)

    # 'q' tuşuna basılınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video kaynağını serbest bırak
cap.release()
cv2.destroyAllWindows()
