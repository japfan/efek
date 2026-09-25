"""
Glow Blade Filter - efek mirip trend TikTok (objek jadi "pedang" berkilau + partikel)
+ opsi partikel hati (PNG custom) yang melayang ke atas.

Cara kerja:
1. Deteksi tangan pakai MediaPipe Tasks API - HandLandmarker (posisi jempol & telunjuk).
   (versi mediapipe terbaru sudah membuang API lama `mp.solutions`, jadi pakai API baru ini)
2. Gambar garis "blade" antara dua titik itu, kasih efek glow (blur + additive blending).
3. Spawn partikel-partikel kecil (sparkle) di sepanjang blade yang muncul lalu memudar.
4. (opsional) Spawn partikel hati dari file PNG (assets/heart.png) yang melayang ke atas
   sambil membesar-mengecil (pulse) dan fade out.

Install dulu:
    pip install opencv-python mediapipe numpy

Jalankan:
    python glow_blade_filter.py
    (tekan ESC buat keluar)

Catatan: script ini otomatis download file model "hand_landmarker.task" (~beberapa MB)
dari Google saat pertama kali dijalankan, jadi butuh koneksi internet sekali di awal.
"""

import cv2
import numpy as np
import time
import random
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HEART_PATH = os.path.join(SCRIPT_DIR, "assets", "heart.png")

# index landmark tangan (sama seperti API lama): 4 = ujung jempol, 8 = ujung telunjuk
THUMB_TIP = 4
INDEX_TIP = 8

# ---- load sprite hati sekali di awal (RGBA, background transparan) ----
_heart_rgba = cv2.imread(HEART_PATH, cv2.IMREAD_UNCHANGED)  # (H, W, 4) BGRA
if _heart_rgba is None:
    print(f"Peringatan: {HEART_PATH} tidak ditemukan, partikel hati tidak akan tampil.")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Mengunduh model hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Selesai download model.")


def overlay_rgba(frame, sprite_bgra, cx, cy, target_size, alpha_mult=1.0):
    """Tempel sprite BGRA (dengan alpha channel) ke frame BGR di posisi (cx, cy),
    diresize ke target_size (lebar=tinggi, px), dengan alpha blending yang benar."""
    if sprite_bgra is None or target_size < 2:
        return

    size = int(target_size)
    sprite = cv2.resize(sprite_bgra, (size, size), interpolation=cv2.INTER_AREA)

    h, w = frame.shape[:2]
    x1, y1 = int(cx - size / 2), int(cy - size / 2)
    x2, y2 = x1 + size, y1 + size

    # clip area yang keluar dari frame
    sx1, sy1 = max(0, -x1), max(0, -y1)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return
    sx2 = sx1 + (x2c - x1c)
    sy2 = sy1 + (y2c - y1c)

    sprite_crop = sprite[sy1:sy2, sx1:sx2]
    roi = frame[y1c:y2c, x1c:x2c]

    alpha = (sprite_crop[:, :, 3:4].astype(np.float32) / 255.0) * alpha_mult
    fg = sprite_crop[:, :, :3].astype(np.float32)
    bg = roi.astype(np.float32)

    blended = fg * alpha + bg * (1 - alpha)
    frame[y1c:y2c, x1c:x2c] = blended.astype(np.uint8)


class HeartParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-15, 15)
        self.vy = random.uniform(-70, -35)   # melayang ke atas
        self.life = random.uniform(1.0, 1.8)
        self.age = 0.0
        self.base_size = random.uniform(24, 42)
        self.pulse_speed = random.uniform(4, 7)
        self.wobble_phase = random.uniform(0, 6.28)

    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt + np.sin(self.age * 3 + self.wobble_phase) * 0.6
        self.y += self.vy * dt
        return self.age < self.life

    def draw(self, frame):
        t = self.age / self.life
        fade_in = min(1.0, self.age / 0.15)          # muncul cepat
        fade_out = max(0.0, 1 - max(0.0, t - 0.6) / 0.4)  # menghilang di 40% akhir
        alpha = fade_in * fade_out
        pulse = 1 + 0.15 * np.sin(self.age * self.pulse_speed)
        size = self.base_size * pulse
        overlay_rgba(frame, _heart_rgba, self.x, self.y, size, alpha_mult=alpha)


class SparkleParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-25, 25)   # px/detik
        self.vy = random.uniform(-60, -15)  # px/detik, ke atas
        self.life = random.uniform(0.35, 0.8)
        self.age = 0.0
        self.size = random.randint(2, 5)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.age += dt
        return self.age < self.life

    def draw(self, frame):
        alpha = max(0.0, 1 - self.age / self.life)
        color = (int(255 * alpha), int(230 * alpha), int(120 * alpha))  # kuning-putih
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, color, -1)


def draw_glow_blade(frame, p1, p2, color=(60, 220, 255)):
    """Gambar garis dengan efek glow (BGR color, default kuning keemasan)."""
    overlay = np.zeros_like(frame, dtype=np.uint8)
    cv2.line(overlay, p1, p2, color, 14)
    overlay = cv2.GaussianBlur(overlay, (31, 31), 0)
    cv2.line(overlay, p1, p2, (255, 255, 255), 4)  # inti putih terang
    cv2.addWeighted(frame, 1.0, overlay, 1.3, 0, dst=frame)


def main():
    ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    particles = []
    heart_particles = []
    prev_time = time.time()
    start_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        now = time.time()
        dt = now - prev_time
        prev_time = now

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]  # tangan pertama yang terdeteksi
            thumb = lm[THUMB_TIP]
            index = lm[INDEX_TIP]
            p1 = (int(thumb.x * w), int(thumb.y * h))
            p2 = (int(index.x * w), int(index.y * h))

            draw_glow_blade(frame, p1, p2)

            # spawn partikel sparkle di sepanjang blade
            for t in np.linspace(0, 1, 6):
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                if random.random() < 0.5:
                    particles.append(SparkleParticle(x, y))

            # spawn partikel hati dari ujung telunjuk, sesekali aja (biar gak penuh)
            if random.random() < 0.15:
                heart_particles.append(HeartParticle(p2[0], p2[1]))

        particles = [p for p in particles if p.update(dt)]
        for p in particles:
            p.draw(frame)

        heart_particles = [p for p in heart_particles if p.update(dt)]
        for p in heart_particles:
            p.draw(frame)

        cv2.imshow("Glow Blade Filter - ESC untuk keluar", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
