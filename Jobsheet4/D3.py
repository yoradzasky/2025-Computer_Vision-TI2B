
import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Indeks Landmark Mata Kiri (didasarkan pada MediaPipe Face Mesh)
# vertikal (atas, bawah), horizontal (kiri, kanan)
L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133

# Fungsi untuk menghitung jarak Euclidean
def dist(p1, p2): 
    """Menghitung jarak Euclidean antara dua titik (x, y)."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi objek FaceMeshDetector
detector = FaceMeshDetector(
    staticMode=False,        # Deteksi per frame
    maxFaces=2,              # Maksimum 2 wajah
    minDetectionCon=0.5,     # Ambang kepercayaan deteksi minimum
    minTrackCon=0.5          # Ambang kepercayaan pelacakan minimum
)

# Variabel untuk menghitung kedipan sederhana
blink_count = 0
closed_frames = 0
is_closed = False

# Konstanta
CLOSED_FRAMES_THRESHOLD = 3   # Jumlah frame berturut-turut untuk dianggap kedipan
EYE_AR_THRESHOLD = 0.20       # Ambang Eye Aspect Ratio (EAR) untuk menilai mata tertutup

while True:
    ok, img = cap.read()
    if not ok:
        break

    # Temukan Face Mesh
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]  # Ambil wajah pertama (list of 468 (x,y) landmark)

        # Hitung jarak vertikal (v) dan horizontal (h) mata kiri
        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])

        # Hitung Eye Aspect Ratio (EAR)
        # Ditambah 1e-8 untuk menghindari pembagian dengan nol
        ear = v / (h + 1e-8)

        # Tampilkan EAR
        cv2.putText(img, f"EAR(L): {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Logika counter kedipan sederhana:
        # Jika EAR < EYE_AR_THRESHOLD
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            
            # Jika mata sudah tertutup selama ambang frame tertentu DAN ini adalah awal kedipan (is_closed=False)
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True # Set status ke 'sedang tertutup'
        else:
            # Mata terbuka
            closed_frames = 0
            is_closed = False # Set status ke 'sedang terbuka'

        # Tampilkan jumlah kedipan pada frame
        cv2.putText(img, f"Blink: {blink_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("FaceMesh + EAR", img)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
