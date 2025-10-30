import cv2
import time

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Cek apakah kamera berhasil dibuka
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1/2.")

# Inisialisasi variabel untuk menghitung FPS
frames = 0
t0 = time.time()

# Loop utama untuk memproses frame
while True:
    # Baca frame dari kamera
    ok, frame = cap.read()
    
    # Keluar dari loop jika pembacaan frame gagal
    if not ok:
        break
    
    # Hitung frame untuk FPS
    frames += 1
    
    # Perbarui judul jendela dengan FPS setiap 1.0 detik
    if time.time() - t0 >= 1.0:
        cv2.setWindowTitle("Preview", f"Preview (FPS ~ {frames})")
        frames = 0
        t0 = time.time()
    
    # Tampilkan frame
    cv2.imshow("Preview", frame)
    
    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya setelah loop selesai
cap.release()
cv2.destroyAllWindows()