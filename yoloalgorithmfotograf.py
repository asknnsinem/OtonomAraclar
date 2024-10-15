from ultralytics import YOLO
import matplotlib.pyplot as plt

# Modeli yükle
model = YOLO('yolov8n.pt')

# Görüntüyü yükle ve tahmin yap
results = model('arabavideo.mp4')

# İlk sonucu al
result = results[0]

# Sonuçları plot et
result.plot()

# Matplotlib ile görüntüyü göster
plt.imshow(result.plot())  # Çizilen görüntüyü matplotlib ile ekranda göster
plt.axis('off')            # Ekseni kapat (isteğe bağlı)
plt.show()                 # Görüntüyü göster
