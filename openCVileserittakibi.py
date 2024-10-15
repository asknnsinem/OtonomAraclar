import cv2
import numpy as np

def region_of_interest(img, vertices):
    # Sadece ilgi alanını kesmek için
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    # Çizgileri görüntü üzerine çizmek için
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)

def process_frame(frame):
    # Grayscale dönüşümü
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection (kenar tespiti)
    edges = cv2.Canny(blur, 50, 150)

    # Yolun alt kısmını işlemek için bir üçgen ROI (Region of Interest) belirle
    height, width = frame.shape[:2]
    roi_vertices = [(0, height), (width // 2, height // 2), (width, height)]

    cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Hough Transformu ile çizgileri algıla
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

    # Çizgileri çerçeve üzerine çiz
    if lines is not None:
        draw_lines(frame, lines)

    return frame

# Video dosyasını veya kamerayı aç
cap = cv2.VideoCapture('arabavideosu2.mp4')  # Kendi video dosyanı veya 0 diyerek kamerayı kullanabilirsin

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Her kareyi işleyip şerit tespitini yap
    processed_frame = process_frame(frame)

    # Sonuçları göster
    cv2.imshow('Lane Detection', processed_frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
