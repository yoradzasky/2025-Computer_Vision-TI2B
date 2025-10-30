import cv2
from cvzone.HandTrackingModule import HandDetector

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi HandDetector
detector = HandDetector(
    staticMode=False,      # Deteksi per frame
    maxHands=1,            # Maksimum 1 tangan
    modelComplexity=1,
    detectionCon=0.5,      # Ambang kepercayaan deteksi minimum
    minTrackCon=0.5        # Ambang kepercayaan pelacakan minimum
)

# Loop utama untuk pemrosesan video
while True:
    ok, img = cap.read()
    if not ok:
        break

    # Temukan tangan dalam frame
    # flipType=True membalikkan frame secara horizontal untuk UI 'mirror' yang lebih intuitif
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # Ambil tangan pertama yang terdeteksi
        hand = hands[0]  # Sebuah dictionary berisi "lmList", "bbox", dll.

        # Periksa jari mana yang terangkat (list panjang 5 berisi 0 atau 1)
        # Indeks: [jempol, telunjuk, tengah, manis, kelingking]
        fingers = detector.fingersUp(hand)
        
        # Hitung jumlah jari yang terangkat
        count = sum(fingers)

        # Tampilkan jumlah jari dan status jari pada frame
        text = f"Fingers: {count} {fingers}"
        cv2.putText(img, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Hands & Fingers", img)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Bersihkan sumber daya setelah loop selesai
cap.release()
cv2.destroyAllWindows()