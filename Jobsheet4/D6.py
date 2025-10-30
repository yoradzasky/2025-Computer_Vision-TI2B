import cv2
import numpy as np
from collections import deque
from cvzone.PoseModule import PoseDetector

# --- KONSTANTA & PENGATURAN ---
MODE = "squat"        # default mode, tekan 'm' untuk toggle ke "pushup"
KNEE_DOWN, KNEE_UP = 80, 160 # ambang sudut lutut squat (derajat)
DOWN_R, UP_R = 0.85, 1.00    # ambang rasio push-up (shoulder-wrist / shoulder-hip)
SAMPLE_OK = 4                # minimal frame konsisten sebelum ganti state

# --- INISIALISASI KAMERA & DETEKTOR ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

detector = PoseDetector(
    staticMode=False, 
    modelComplexity=1,
    enableSegmentation=False, 
    detectionCon=0.5, 
    trackCon=0.5
)

# --- VARIABEL STATE & COUNTER ---
count, state = 0, "up"
# Deque digunakan untuk 'debounce' (menghaluskan) perubahan state
debounce = deque(maxlen=6)

# --- FUNGSI BANTUAN ---
def ratio_pushup(lm):
    """
    Menghitung rasio untuk mendeteksi posisi push-up.
    lm: list landmark tubuh (lmList)
    Rasio: jarak(bahu–pergelangan_tangan) / jarak(bahu–pinggul)
    Landmark yang digunakan (contoh sisi kiri): 11 (bahuL), 15 (pergelangan_tanganL), 23 (pinggulL)
    """
    # Mengambil koordinat X, Y, Z (indeks [0:3] atau [1:4] jika lmList[i] berisi (id, x, y, z))
    # Asumsi lmList[i] adalah (x, y, z) jika lmList adalah list of np.array
    # Berdasarkan kode asli: lmList[i] = (id, x, y), jadi gunakan [0:2]
    # Namun, karena ratio_pushup(lm) mengambil lmList, dan kode aslinya menggunakan [1:3], saya sesuaikan.
    # Jika lmList[i] = [id, x, y, z], maka [1:4] adalah (x, y, z)
    # Jika lmList[i] = [id, x, y], maka [1:3] adalah (x, y)
    
    # Berdasarkan baris 22-24 kode asli, asumsikan elemen lmList adalah [id, x, y, z, vis]
    # dan koordinat yang diambil adalah [x, y, z] (indeks 1, 2, 3).
    
    sh = np.array(lm[11][1:4]) # Shoulder Left
    wr = np.array(lm[15][1:4]) # Wrist Left
    hp = np.array(lm[23][1:4]) # Hip Left

    # Menghitung jarak Euclidean (norm)
    dist_sw = np.linalg.norm(sh - wr)
    dist_sh = np.linalg.norm(sh - hp)
    
    # Menggunakan 1e-8 untuk menghindari pembagian dengan nol
    return dist_sw / (dist_sh + 1e-8)

# --- LOOP UTAMA ---
while True:
    ok, img = cap.read()
    if not ok: 
        break
    
    # Deteksi pose
    img = detector.findPose(img, draw=True)
    # Ambil landmark (id, x, y, z, vis)
    lmList, _ = detector.findPosition(img, draw=False) 
    
    flag = None

    if lmList:
        if MODE == "squat":
            # Hitung sudut lutut kiri (23: hip, 25: knee, 27: ankle)
            angL, img = detector.findAngle(
                lmList[23][0:2], # hip
                lmList[25][0:2], # knee
                lmList[27][0:2], # ankle
                img=img,
                color=(0, 0, 255),
                scale=10
            )

            # Hitung sudut lutut kanan (24: hip, 26: knee, 28: ankle)
            angR, img = detector.findAngle(
                lmList[24][0:2], # hip
                lmList[26][0:2], # knee
                lmList[28][0:2], # ankle
                img=img,
                color=(0, 255, 0),
                scale=10
            )

            # Sudut rata-rata
            ang = (angL + angR) / 2.0
            
            # Tentukan posisi
            if ang < KNEE_DOWN: 
                flag = "down"
            elif ang > KNEE_UP: 
                flag = "up"
            
            # Tampilkan sudut
            cv2.putText(img, f"Knee: {ang:5.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        else: # MODE == "pushup"
            # Hitung rasio push-up: (shoulder–wrist)/(shoulder–hip)
            r = ratio_pushup(lmList)
            
            # Tentukan posisi
            if r < DOWN_R: 
                flag = "down"
            elif r > UP_R: 
                flag = "up"
            
            # Tampilkan rasio
            cv2.putText(img, f"Ratio: {r:4.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Logika Debounce dan Counter
        debounce.append(flag)
        
        # Transisi dari 'up' ke 'down'
        if debounce.count("down") >= SAMPLE_OK and state == "up":
            state = "down"
            
        # Transisi dari 'down' ke 'up' (Repetisi selesai)
        if debounce.count("up") >= SAMPLE_OK and state == "down":
            state = "up"
            count += 1

    # Tampilkan informasi utama di layar
    cv2.putText(img, f"Mode: {MODE.upper()} Count: {count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"State: {state}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Tampilkan frame
    cv2.imshow("Pose Counter", img)
    
    # Kontrol input pengguna
    key = cv2.waitKey(1) & 0xFF
    
    # Keluar dari loop
    if key == ord('q'):
        break
    
    # Toggle mode
    if key == ord('m'):
        MODE = "pushup" if MODE == "squat" else "squat"
        # Reset counter dan state saat mode berganti
        count, state = 0, "up"
        debounce.clear()
        

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()