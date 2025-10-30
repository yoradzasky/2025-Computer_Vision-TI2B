import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Fungsi untuk menghitung jarak Euclidean
def dist(a, b):
    """Menghitung jarak Euclidean antara dua titik (x, y)."""
    return np.linalg.norm(np.array(a) - np.array(b))

def classify_gesture(hand):
    """
    Mengklasifikasikan gestur tangan ("OK", "THUMBS_UP", "ROCK", "PAPER", "SCISSORS", "UNKNOWN")
    berdasarkan heuristik jarak relatif.
    Catatan: lmList berisi 21 titik (x,y,z) dalam piksel.
    """
    lm = hand["lmList"]

    # Ekstraksi koordinat X dan Y (indeks [0:2]) untuk titik penting
    wrist = np.array(lm[0][0:2])
    thumb_tip = np.array(lm[4][0:2])
    index_tip = np.array(lm[8][0:2])
    middle_tip = np.array(lm[12][0:2])
    ring_tip = np.array(lm[16][0:2])
    pinky_tip = np.array(lm[20][0:2])

    # Heuristik jarak relatif: Jarak rata-rata ujung 4 jari ke pergelangan tangan (wrist)
    # Tidak termasuk Jempol karena posisinya paling variabel
    r_mean = np.mean([
        dist(index_tip, wrist),
        dist(middle_tip, wrist),
        dist(ring_tip, wrist),
        dist(pinky_tip, wrist)
    ])
    
    # Heuristik Jarak untuk Jempol
    # Jarak ujung jari jempol ke ujung jari telunjuk
    dist_thumb_index = dist(thumb_tip, index_tip)

    # --- ATURAN KLASIFIKASI GESTUR ---

    # 1. OK Gesture
    # Jempol dan Telunjuk bertemu (jarak sangat kecil)
    if dist_thumb_index < 35:
        return "OK"

    # 2. THUMBS_UP (Jempol ke atas)
    # Ujung jempol berada jauh dari pergelangan tangan (vertikal/y kecil)
    # DAN Jauh dari pergelangan tangan daripada jari telunjuk yang dibengkokkan
    # Jari telunjuk (lm[8][1]) lebih tinggi (y lebih kecil) dari pergelangan tangan (lm[1][1]) 
    # di MediaPipe, tapi di sini kita pakai wrist (lm[0][1])
    if (thumb_tip[1] < wrist[1] - 40) and (dist(thumb_tip, wrist) > 0.8 * dist(index_tip, wrist)):
        return "THUMBS_UP"

    # 3. ROCK/PAPER/SCISSORS (Gestur Batu-Kertas-Gunting)
    
    # ROCK (Batu)
    # Rata-rata jarak jari ke pergelangan tangan kecil (semua jari ditekuk)
    if r_mean < 120:
        return "ROCK"

    # PAPER (Kertas)
    # Rata-rata jarak jari ke pergelangan tangan besar (semua jari lurus)
    if r_mean > 200:
        return "PAPER"

    # SCISSORS (Gunting)
    # Telunjuk dan jari tengah lurus (jarak > 180), Jari manis dan kelingking ditekuk (jarak < 160)
    if (dist(index_tip, wrist) > 180 and dist(middle_tip, wrist) > 180) and \
       (dist(ring_tip, wrist) < 160 and dist(pinky_tip, wrist) < 160):
        return "SCISSORS"

    # Jika tidak ada aturan yang cocok
    return "UNKNOWN"


# --- PROGRAM UTAMA ---

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi HandDetector
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.5,
    minTrackCon=0.5
)

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Temukan tangan dalam frame. flipType=True memberikan UI 'mirror'
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # Klasifikasikan gestur tangan pertama yang terdeteksi
        label = classify_gesture(hands[0])

        # Tampilkan label gestur
        cv2.putText(img, f"Gesture: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Tampilkan frame
    cv2.imshow("Hand Gestures (cvzone)", img)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()