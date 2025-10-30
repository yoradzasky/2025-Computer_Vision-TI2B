import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi Pose Detector
detector = PoseDetector(
    staticMode=False, 
    modelComplexity=1,
    enableSegmentation=False, 
    detectionCon=0.5, 
    trackCon=0.5
)

while True:
    # Tangkap setiap frame dari webcam
    success, img = cap.read()

    # Temukan pose manusia dalam frame
    img = detector.findPose(img)

    # Temukan landmark, bounding box, dan pusat tubuh dalam frame
    # Set draw=True untuk menggambar landmark dan bounding box pada gambar
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)

    # Periksa apakah ada landmark tubuh yang terdeteksi
    if lmList:
        # Dapatkan pusat bounding box di sekitar tubuh
        center = bboxInfo["center"]

        # Gambar lingkaran di pusat bounding box
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Hitung jarak antara landmark 11 dan 15 dan gambarkan pada gambar
        # Landmark 11 biasanya Bahu Kiri, 15 biasanya Pergelangan Tangan Kiri
        length, img, info = detector.findDistance(
            lmList[11][0:2], 
            lmList[15][0:2],
            img=img,
            color=(255, 0, 0),
            scale=10
        )

        # Hitung sudut antara landmark 11, 13, dan 15 dan gambarkan pada gambar
        # Landmark 11 (Bahu), 13 (Siku), 15 (Pergelangan Tangan) - untuk sudut siku
        angle, img = detector.findAngle(
            lmList[11][0:2], 
            lmList[13][0:2],
            lmList[15][0:2],
            img=img,
            color=(0, 0, 255),
            scale=10
        )

        # Periksa apakah sudut mendekati 50 derajat dengan offset 10
        isCloseAngle50 = detector.angleCheck(
            myAngle=angle,
            targetAngle=50,
            offset=10
        )

        # Cetak hasil pemeriksaan sudut
        print(isCloseAngle50)

    # Tampilkan frame
    cv2.imshow("Pose + Angle", img)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()