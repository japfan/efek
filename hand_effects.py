"""
Hand Gesture Filter - Webcam Effect Real-Time
1. Tangan Kanan Nunjuk ke Atas -> Muncul efek Love (assets/heart.png) + UNLOCK Efek Bunga
2. Jempol Nunjuk ke Diri Sendiri -> Muncul efek Bunga-Bunga FULLSCREEN di BELAKANG SUBJEK
   (menggunakan AI Segmentation agar bunga melayang di belakang tubuh pengguna!)
3. Telunjuk Kiri & Kanan Bersentuhan lalu Melebar -> Membentuk shape Persegi Panjang Neon Dinamis
   yang bisa di-twist dan diputar bebas 360 derajat mengikuti jari!
4. Telapak Tangan Bersentuhan & Jari-Jari Renggang -> Objek Holografik 3D Kristal Prisma 9 Sisi
5. Pose Ketapel (Tangan Cabang 'V' + Tangan Cubit Tarik Karet) -> Tembak Peluru ke Kamera & Efek Embun Kaca (Di-usap untuk Bersih)
6. Dua Tangan Menarik dari Atas ke Bawah -> Filter Black & White (B&W) Sinematik dengan Animasi Tirai Push dari Atas!

Cara Menjalankan:
    python hand_effects.py

Kontrol Tombol:
    ESC / Q : Keluar dari aplikasi
    H / T   : Sembunyikan / tampilkan semua elemen teks UI (Clean View Mode)
    B       : Toggle manual filter Black & White (B&W)
    L       : Tampilkan / sembunyikan titik landmark tangan (debug)
    C       : Bersihkan partikel yang sedang aktif di layar
    R       : Reset seluruh efek
"""

import os
import cv2
import numpy as np
import time
import random
import math
import urllib.request
import threading

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ----------------- PATH & KONFIGURASI -----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "assets")

HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

SEG_MODEL_PATH = os.path.join(SCRIPT_DIR, "selfie_multiclass.tflite")
SEG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)

# Indeks Landmark MediaPipe (21 titik per tangan)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def ensure_models():
    """Download model hand landmarker dan selfie segmenter jika belum ada."""
    if not os.path.exists(HAND_MODEL_PATH):
        print(f"Mengunduh model hand_landmarker.task ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("Model hand_landmarker.task berhasil diunduh!")

    if not os.path.exists(SEG_MODEL_PATH):
        print(f"Mengunduh model selfie_multiclass.tflite untuk background segmentation...")
        urllib.request.urlretrieve(SEG_MODEL_URL, SEG_MODEL_PATH)
        print("Model selfie_multiclass.tflite berhasil diunduh!")


# ----------------- ASYNC BACKGROUND SEGMENTER -----------------
class AsyncSegmenter:
    """
    Menjalankan MediaPipe ImageSegmenter secara asynchronous di background thread
    agar proses webcam & rendering tetap berjalan lancar pada 30-60 FPS.
    """
    def __init__(self, model_path=SEG_MODEL_PATH):
        options = vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        self.latest_frame = None
        self.mask = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def update_frame(self, rgb_frame):
        with self.lock:
            self.latest_frame = rgb_frame.copy()

    def get_mask(self):
        with self.lock:
            return self.mask

    def _worker(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None

            if frame_to_process is not None:
                h, w = frame_to_process.shape[:2]
                # Downscale untuk inferensi segmentasi yang sangat cepat
                small = cv2.resize(frame_to_process, (256, 256), interpolation=cv2.INTER_AREA)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
                try:
                    res = self.segmenter.segment(mp_img)
                    cat = res.category_mask.numpy_view()
                    # Nilai > 0 adalah bagian tubuh pengguna (hair, skin, clothes, etc)
                    person_small = (cat > 0).astype(np.uint8) * 255
                    mask_full = cv2.resize(person_small, (w, h), interpolation=cv2.INTER_LINEAR)
                    with self.lock:
                        self.mask = mask_full
                except Exception as e:
                    pass
            else:
                time.sleep(0.01)

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.segmenter.close()


# ----------------- LOAD ASSETS -----------------
def load_sprites():
    """Load semua asset sprite (BGRA) dan buat versi pre-scaled untuk performa maksimal."""
    sprites = {
        "heart": cv2.imread(os.path.join(ASSETS_DIR, "heart.png"), cv2.IMREAD_UNCHANGED),
        "sakura": cv2.imread(os.path.join(ASSETS_DIR, "flower_sakura.png"), cv2.IMREAD_UNCHANGED),
        "bloom": cv2.imread(os.path.join(ASSETS_DIR, "flower_bloom.png"), cv2.IMREAD_UNCHANGED),
        "petal": cv2.imread(os.path.join(ASSETS_DIR, "flower_petal.png"), cv2.IMREAD_UNCHANGED),
    }
    for name, img in list(sprites.items()):
        if img is None:
            print(f"[Peringatan] Asset '{name}' tidak ditemukan di folder assets!")
        else:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                sprites[name] = img
            # Pre-scale ke ukuran wajar (80x80 / 64x64) agar resize per frame super cepat (menghemat ~15ms)
            target_dim = 64 if name == "petal" else 80
            sprites[f"{name}_small"] = cv2.resize(img, (target_dim, target_dim), interpolation=cv2.INTER_AREA)
    return sprites


# ----------------- OVERLAY RENDERING -----------------
def overlay_rgba(frame, sprite_bgra, cx, cy, target_size, angle=0.0, alpha_mult=1.0):
    """Tempelkan sprite RGBA/BGRA ke frame dengan rotasi, skala, dan fast integer alpha blending."""
    if sprite_bgra is None or target_size < 3 or alpha_mult <= 0.01:
        return

    size = int(target_size)
    sprite = cv2.resize(sprite_bgra, (size, size), interpolation=cv2.INTER_LINEAR)

    if abs(angle) > 1.0:
        M = cv2.getRotationMatrix2D((size * 0.5, size * 0.5), angle, 1.0)
        sprite = cv2.warpAffine(
            sprite,
            M,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    h, w = frame.shape[:2]
    x1, y1 = int(cx - size * 0.5), int(cy - size * 0.5)
    x2, y2 = x1 + size, y1 + size

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    if x2c <= x1c or y2c <= y1c:
        return

    sx1, sy1 = max(0, -x1), max(0, -y1)
    sx2 = sx1 + (x2c - x1c)
    sy2 = sy1 + (y2c - y1c)

    sprite_crop = sprite[sy1:sy2, sx1:sx2]
    roi = frame[y1c:y2c, x1c:x2c]

    # Fast integer alpha blending (3x lebih cepat dibanding float32)
    alpha = sprite_crop[:, :, 3:4]
    if alpha_mult < 0.99:
        a_scale = int(alpha_mult * 256)
        alpha = ((alpha.astype(np.uint16) * a_scale) >> 8).astype(np.uint8)

    a16 = alpha.astype(np.uint16)
    inv16 = 255 - a16
    blended = ((sprite_crop[:, :, :3].astype(np.uint16) * a16 + roi.astype(np.uint16) * inv16) >> 8).astype(np.uint8)
    roi[:] = blended


# ----------------- PARTICLE CLASSES -----------------
class HeartParticle:
    """Partikel Hati (Love) yang melayang ke atas dengan pulse dan wobble."""
    def __init__(self, x, y, sprite):
        self.x = float(x)
        self.y = float(y)
        self.sprite = sprite
        self.vx = random.uniform(-25, 25)
        self.vy = random.uniform(-110, -60)
        self.life = random.uniform(1.2, 2.0)
        self.age = 0.0
        self.base_size = random.uniform(32, 54)
        self.pulse_speed = random.uniform(4.0, 7.5)
        self.wobble_freq = random.uniform(2.5, 4.5)
        self.wobble_phase = random.uniform(0, 6.28)
        self.angle = random.uniform(-12, 12)
        self.rot_speed = random.uniform(-15, 15)

    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt + math.sin(self.age * self.wobble_freq + self.wobble_phase) * 1.2
        self.y += self.vy * dt
        self.angle += self.rot_speed * dt
        return self.age < self.life

    def draw(self, frame):
        t = self.age / self.life
        fade_in = min(1.0, self.age / 0.15)
        fade_out = max(0.0, 1.0 - max(0.0, t - 0.65) / 0.35)
        alpha = fade_in * fade_out

        pulse = 1.0 + 0.18 * math.sin(self.age * self.pulse_speed)
        size = self.base_size * pulse
        overlay_rgba(frame, self.sprite, self.x, self.y, size, angle=self.angle, alpha_mult=alpha)


class FlowerParticle:
    """Partikel Bunga Mekar (Sakura / Bloom) fullscreen yang mengembang dan berputar perlahan."""
    def __init__(self, x, y, sprite):
        self.x = float(x)
        self.y = float(y)
        self.sprite = sprite
        self.vx = random.uniform(-20, 20)
        self.vy = random.uniform(-30, 10)
        self.life = random.uniform(2.0, 3.2)
        self.age = 0.0
        self.target_size = random.uniform(55, 95)
        self.angle = random.uniform(0, 360)
        self.rot_speed = random.uniform(-40, 40)
        self.wobble_freq = random.uniform(1.5, 3.0)
        self.wobble_amp = random.uniform(15, 30)

    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt + math.sin(self.age * self.wobble_freq) * (self.wobble_amp * dt)
        self.y += self.vy * dt
        self.angle += self.rot_speed * dt
        return self.age < self.life

    def draw(self, frame):
        t = self.age / self.life
        if self.age < 0.35:
            scale = (self.age / 0.35) * 1.15
        elif self.age < 0.55:
            scale = 1.15 - ((self.age - 0.35) / 0.20) * 0.15
        else:
            scale = 1.0

        fade_out = max(0.0, 1.0 - max(0.0, t - 0.65) / 0.35)
        alpha = min(1.0, self.age / 0.15) * fade_out
        size = self.target_size * scale
        overlay_rgba(frame, self.sprite, self.x, self.y, size, angle=self.angle, alpha_mult=alpha)


class PetalParticle:
    """Partikel Kelopak Bunga (Petal) fullscreen yang melayang jatuh di latar belakang."""
    def __init__(self, x, y, sprite):
        self.x = float(x)
        self.y = float(y)
        self.sprite = sprite
        self.vx = random.uniform(-35, 35)
        self.vy = random.uniform(50, 120)  # Melayang jatuh ke bawah
        self.life = random.uniform(2.5, 4.2)
        self.age = 0.0
        self.size = random.uniform(25, 45)
        self.angle = random.uniform(0, 360)
        self.rot_speed = random.uniform(-80, 80)
        self.sway_freq = random.uniform(2.5, 5.0)
        self.sway_amp = random.uniform(45, 80)

    def update(self, dt):
        self.age += dt
        self.x += (self.vx + math.sin(self.age * self.sway_freq) * self.sway_amp) * dt
        self.y += self.vy * dt
        self.angle += self.rot_speed * dt
        return self.age < self.life

    def draw(self, frame):
        t = self.age / self.life
        alpha = min(1.0, self.age / 0.2) * max(0.0, 1.0 - max(0.0, t - 0.70) / 0.30)
        overlay_rgba(frame, self.sprite, self.x, self.y, self.size, angle=self.angle, alpha_mult=alpha)


class SparkleParticle:
    """Partikel kilau kecil berkilau."""
    def __init__(self, x, y, color=(255, 230, 150)):
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.vx = random.uniform(-35, 35)
        self.vy = random.uniform(-35, 35)
        self.life = random.uniform(0.3, 0.7)
        self.age = 0.0
        self.max_size = random.randint(3, 6)

    def update(self, dt):
        self.age += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        return self.age < self.life

    def draw(self, frame):
        alpha = max(0.0, 1.0 - self.age / self.life)
        size = int(self.max_size * math.sin((self.age / self.life) * math.pi))
        if size >= 1:
            col = (int(self.color[0] * alpha), int(self.color[1] * alpha), int(self.color[2] * alpha))
            cv2.circle(frame, (int(self.x), int(self.y)), size, col, -1)


# ----------------- LOGIKA DETEKSI GESTURE -----------------
def dist_sq(p1, p2):
    """Jarak kuadrat 2D antara dua landmark."""
    return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2


def is_finger_folded(lm, tip_idx, pip_idx, wrist_idx=WRIST):
    """Mengecek apakah suatu jari ditekuk/mengepal."""
    d_tip = dist_sq(lm[tip_idx], lm[wrist_idx])
    d_pip = dist_sq(lm[pip_idx], lm[wrist_idx])
    return d_tip < d_pip * 1.15 or lm[tip_idx].y > lm[pip_idx].y


def is_pointing_up(lm, is_right_hand):
    """Gesture 1: Tangan Kanan Nunjuk ke Atas."""
    if not is_right_hand:
        return False

    dx = lm[INDEX_TIP].x - lm[INDEX_MCP].x
    dy = lm[INDEX_TIP].y - lm[INDEX_MCP].y

    index_straight_up = (
        lm[INDEX_TIP].y < lm[INDEX_PIP].y
        and lm[INDEX_PIP].y < lm[INDEX_MCP].y
        and dy < -0.10
        and abs(dx) / (abs(dy) + 1e-6) < 0.75
    )
    if not index_straight_up:
        return False

    middle_folded = is_finger_folded(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_folded = is_finger_folded(lm, RING_TIP, RING_PIP)
    pinky_folded = is_finger_folded(lm, PINKY_TIP, PINKY_PIP)

    folded_count = sum([middle_folded, ring_folded, pinky_folded])
    return folded_count >= 2


def is_pointing_to_self_with_thumb(lm, is_right_hand):
    """Gesture 2: Jempol Nunjuk Diri Sendiri (ke arah dada/tubuh)."""
    index_folded = is_finger_folded(lm, INDEX_TIP, INDEX_PIP)
    middle_folded = is_finger_folded(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_folded = is_finger_folded(lm, RING_TIP, RING_PIP)
    pinky_folded = is_finger_folded(lm, PINKY_TIP, PINKY_PIP)

    folded_count = sum([index_folded, middle_folded, ring_folded, pinky_folded])
    if folded_count < 3:
        return False

    thumb_dist_wrist = dist_sq(lm[THUMB_TIP], lm[WRIST])
    thumb_mcp_dist_wrist = dist_sq(lm[THUMB_MCP], lm[WRIST])
    if thumb_dist_wrist < thumb_mcp_dist_wrist * 0.9:
        return False

    dx = lm[THUMB_TIP].x - lm[THUMB_MCP].x
    dy = lm[THUMB_TIP].y - lm[THUMB_MCP].y
    dz = lm[THUMB_TIP].z - lm[THUMB_MCP].z

    pointing_z = dz > 0.015 or lm[THUMB_TIP].z > lm[WRIST].z + 0.01
    pointing_inward = (is_right_hand and dx > 0.03) or ((not is_right_hand) and dx < -0.03)
    pointing_down_inward = dy > 0.04 and (abs(dx) > 0.02 or pointing_z)

    return pointing_z or pointing_inward or pointing_down_inward


# ----------------- FITUR 3: SHAPE PERSEGI PANJANG NEON BERPUTAR (TWIST) -----------------
def check_index_fingers_touching(hands_screen, touch_dist_threshold=55.0):
    """
    Mengecek apakah ujung jari telunjuk kedua tangan sedang bersentuhan.
    Mengembalikan: (is_touching, touch_point, index_dist)
    """
    if len(hands_screen) < 2:
        return False, None, 999.0

    hand_a_lm, to_screen_a = hands_screen[0]
    hand_b_lm, to_screen_b = hands_screen[1]

    idx_a = np.array(to_screen_a(hand_a_lm[INDEX_TIP]), dtype=np.float32)
    idx_b = np.array(to_screen_b(hand_b_lm[INDEX_TIP]), dtype=np.float32)

    dist = float(np.linalg.norm(idx_a - idx_b))
    touch_point = (idx_a + idx_b) * 0.5
    is_touching = bool(dist <= touch_dist_threshold)
    return is_touching, touch_point, dist


def get_two_handed_rectangle_corners(hands_screen):
    """
    Membentuk 4 sudut persegi panjang dinamis yang mengunci pada jari telunjuk dan jempol kedua tangan.
    Sudut: [idx_a, idx_b, thb_b, thb_a] sehingga shape membentuk loop keliling tertutup
    dan dapat diputar (twist) bebas 360 derajat tanpa pernah saling silang.
    """
    if len(hands_screen) < 2:
        return None, 0.0

    hands_sorted = sorted(hands_screen, key=lambda h: h[1](h[0][WRIST])[0])
    hand_a_lm, to_screen_a = hands_sorted[0]
    hand_b_lm, to_screen_b = hands_sorted[1]

    idx_a = np.array(to_screen_a(hand_a_lm[INDEX_TIP]), dtype=np.float32)
    thb_a = np.array(to_screen_a(hand_a_lm[THUMB_TIP]), dtype=np.float32)

    idx_b = np.array(to_screen_b(hand_b_lm[INDEX_TIP]), dtype=np.float32)
    thb_b = np.array(to_screen_b(hand_b_lm[THUMB_TIP]), dtype=np.float32)

    corners = [idx_a, idx_b, thb_b, thb_a]

    axis_vec = idx_b - idx_a
    angle_deg = math.degrees(math.atan2(axis_vec[1], axis_vec[0]))

    return corners, angle_deg


def draw_dynamic_rectangle(frame, corners, angle_deg, particles, theme_color=(255, 215, 60), show_text=True):
    """
    Menggambar shape persegi panjang neon bersih (tanpa garis silinder) yang responsif:
    - Multi-layer neon glow tebal + garis putih inti
    - Lapisan kaca semi-transparan di bagian dalam
    - Reticle cincin di 4 ujung jari
    - Corner brackets futuristik
    - Titik tengah crosshair dan teks derajat rotasi/twist
    - Sparkles di sepanjang garis keliling
    """
    pts = np.array(corners, dtype=np.int32)
    h, w = frame.shape[:2]

    # 1. Tint kaca semi-transparan di bagian dalam frame
    tint_overlay = frame.copy()
    tint_bgr = (int(theme_color[0] * 0.35), int(theme_color[1] * 0.35), int(theme_color[2] * 0.35))
    cv2.fillPoly(tint_overlay, [pts], tint_bgr)
    cv2.addWeighted(tint_overlay, 0.20, frame, 0.80, 0, dst=frame)

    # 2. Glow Neon Border tebal di sekeliling persegi panjang (Optimized Downscaled Bloom)
    glow_overlay = np.zeros_like(frame)
    cv2.polylines(glow_overlay, [pts], isClosed=True, color=theme_color, thickness=12, lineType=cv2.LINE_AA)
    small_glow = cv2.resize(glow_overlay, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    glow_blurred = cv2.GaussianBlur(small_glow, (15, 15), 0)
    glow_up = cv2.resize(glow_blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.addWeighted(frame, 1.0, glow_up, 1.35, 0, dst=frame)

    # Garis tepi putih terang di inti
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 3. Corner Brackets & Reticle di 4 Ujung Jari
    for i, pt in enumerate(pts):
        px, py = int(pt[0]), int(pt[1])

        cv2.circle(frame, (px, py), 9, theme_color, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)

        prev_pt = pts[(i - 1) % 4]
        next_pt = pts[(i + 1) % 4]

        v1 = prev_pt - pt
        v2 = next_pt - pt
        l1 = np.linalg.norm(v1) + 1e-5
        l2 = np.linalg.norm(v2) + 1e-5

        arm_len = 26
        b1 = (pt + (v1 / l1) * min(arm_len, l1 * 0.45)).astype(int)
        b2 = (pt + (v2 / l2) * min(arm_len, l2 * 0.45)).astype(int)

        cv2.line(frame, (px, py), (int(b1[0]), int(b1[1])), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(frame, (px, py), (int(b2[0]), int(b2[1])), (255, 255, 255), 3, cv2.LINE_AA)

    # 4. Center Crosshair di pusat persegi panjang
    mid_center = np.mean(pts, axis=0)
    mx, my = int(mid_center[0]), int(mid_center[1])
    cv2.circle(frame, (mx, my), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (mx, my), 11, theme_color, 1, cv2.LINE_AA)

    if show_text:
        deg_text = f"Rotasi: {int(angle_deg % 360)} deg"
        cv2.putText(frame, deg_text, (mx - 45, my - 16), cv2.FONT_HERSHEY_PLAIN, 0.95, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Partikel kilau di sepanjang tepian
    for i in range(4):
        p_start = pts[i]
        p_end = pts[(i + 1) % 4]
        if random.random() < 0.45:
            t = random.random()
            sx = int(p_start[0] + (p_end[0] - p_start[0]) * t)
            sy = int(p_start[1] + (p_end[1] - p_start[1]) * t)
            particles.append(SparkleParticle(sx, sy, color=theme_color))


# ----------------- FITUR 4: OBJEK 3D KRISTAL PRISMA (TELAPAK BERSENTUHAN & JARI MERENGGANG) -----------------
def is_cupped_hands_gesture(hands_data):
    """
    Gesture 4: Telapak tangan saling bersentuhan di bagian bawah tetapi jari-jari merenggang
    (seperti membentuk wadah teratai / mangkuk penopang kristal).
    Input hands_data: list dari tuple (lm, is_right_hand, to_screen)
    Mengembalikan: (is_cupped, crystal_data)
        crystal_data: dict berisi titik-titik jangkar tangan langsung:
            'p_bottom': titik temu pergelangan tangan (apex bawah kristal)
            'idx_l': ujung telunjuk tangan kiri (sudut kiri atas depan)
            'idx_r': ujung telunjuk tangan kanan (sudut kanan atas depan)
            'pky_l': ujung kelingking/sisi kiri (sudut samping kiri)
            'pky_r': ujung kelingking/sisi kanan (sudut samping kanan)
    """
    if len(hands_data) < 2:
        return False, None

    # Urutkan tangan kiri dan tangan kanan di layar
    hands_sorted = sorted(hands_data, key=lambda h: h[2](h[0][WRIST])[0])
    h_left, h_right = hands_sorted[0], hands_sorted[1]

    lm_l, to_s_l = h_left[0], h_left[2]
    lm_r, to_s_r = h_right[0], h_right[2]

    wrist_l = np.array(to_s_l(lm_l[WRIST]), dtype=np.float32)
    wrist_r = np.array(to_s_r(lm_r[WRIST]), dtype=np.float32)

    # 1. Jarak pergelangan/pangkal telapak tangan harus dekat (bersentuhan di bawah)
    wrist_dist = float(np.linalg.norm(wrist_l - wrist_r))
    if wrist_dist > 125.0:
        return False, None

    # 2. Ujung-ujung jari harus merenggang terbuka (bukan mengepal)
    idx_l = np.array(to_s_l(lm_l[INDEX_TIP]), dtype=np.float32)
    idx_r = np.array(to_s_r(lm_r[INDEX_TIP]), dtype=np.float32)
    index_dist = float(np.linalg.norm(idx_l - idx_r))

    pinky_l = np.array(to_s_l(lm_l[PINKY_TIP]), dtype=np.float32)
    pinky_r = np.array(to_s_r(lm_r[PINKY_TIP]), dtype=np.float32)
    span_width = float(np.linalg.norm(pinky_l - pinky_r))

    # Syarat: jari-jari terbuka lebar, jarak ujung jari setidaknya 1.5x jarak pergelangan tangan
    if index_dist < wrist_dist * 1.5 or index_dist < 100.0 or span_width < 120.0:
        return False, None

    # Pastikan jari telunjuk tidak ditekuk ke dalam telapak
    if is_finger_folded(lm_l, INDEX_TIP, INDEX_PIP) and is_finger_folded(lm_r, INDEX_TIP, INDEX_PIP):
        return False, None

    p_bottom = (wrist_l + wrist_r) * 0.5

    # Urutkan seluruh 10 ujung jari (5 jari tangan kiri + 5 jari tangan kanan)
    # Urutan anatomis: Kelingking(20), Jari Manis(16), Jari Tengah(12), Telunjuk(8), Jempol(4)
    tips_order = [PINKY_TIP, RING_TIP, MIDDLE_TIP, INDEX_TIP, THUMB_TIP]
    pts_l = [np.array(to_s_l(lm_l[i]), dtype=np.float32) for i in tips_order]
    pts_r = [np.array(to_s_r(lm_r[i]), dtype=np.float32) for i in reversed(tips_order)]

    # Pastikan orientasi titik tangan kiri dari paling kiri (luar) ke arah tengah
    if pts_l[0][0] > pts_l[-1][0]:
        pts_l.reverse()
    # Pastikan orientasi titik tangan kanan dari tengah ke paling kanan (luar)
    if pts_r[0][0] > pts_r[-1][0]:
        pts_r.reverse()

    ordered_fingers = pts_l + pts_r

    crystal_data = {
        "p_bottom": p_bottom,
        "fingers": np.array(ordered_fingers, dtype=np.float32),
        "idx_l": idx_l,
        "idx_r": idx_r,
        "pky_l": pinky_l,
        "pky_r": pinky_r,
    }
    return True, crystal_data


def draw_3d_crystal(frame, crystal_data, particles):
    """
    Menggambar 3D Faceted Crystal / Prisma Berlian yang TERKUNCI LANGSUNG pada tiap jari:
    - Tanpa kotak bergaris strip di atas (sesuai permintaan user).
    - Memiliki facet / sisi kristal untuk SETIAP jari (total 9 facet fanning out dari apex bawah ke seluruh 10 jari).
    - Apex bawah menancap pada titik sentuh pergelangan tangan (p_bottom).
    - Garis wireframe neon glow menyala di setiap rusuk jari dan tepian atas (rim).
    - Simpul cincin neon di setiap ujung jari.
    - Sparkles memancar dari kristal.
    """
    h, w = frame.shape[:2]
    p_bottom = crystal_data["p_bottom"]
    fingers = crystal_data["fingers"]
    n_fingers = len(fingers)
    if n_fingers < 2:
        return

    num_facets = n_fingers - 1

    # Palet warna gemstone kristal untuk setiap sisi jari (BGR):
    # Simetris dari sisi kiri ke sisi kanan:
    facet_palette = [
        (45, 185, 115),   # 0: Zamrud / Emerald Green (Kelingking - Manis Kiri)
        (210, 140, 30),   # 1: Azure Cyan (Manis - Tengah Kiri)
        (225, 95, 35),    # 2: Royal Blue (Tengah - Telunjuk Kiri)
        (185, 45, 115),   # 3: Deep Amethyst / Purple (Telunjuk - Jempol Kiri)
        (235, 80, 20),    # 4: Sapphire Core Lapis (Jempol Kiri - Jempol Kanan, Pusat)
        (185, 45, 115),   # 5: Deep Amethyst / Purple (Jempol - Telunjuk Kanan)
        (225, 95, 35),    # 6: Royal Blue (Telunjuk - Tengah Kanan)
        (210, 140, 30),   # 7: Azure Cyan (Tengah - Manis Kanan)
        (35, 160, 230),   # 8: Golden Amber (Manis - Kelingking Kanan)
    ]

    # 1. Gambar facet kristal translusen per setiap pasang jari
    overlay = frame.copy()
    for i in range(num_facets):
        pts_facet = np.array([p_bottom, fingers[i], fingers[i + 1]], dtype=np.int32)
        color = facet_palette[i % len(facet_palette)]
        cv2.fillPoly(overlay, [pts_facet], color)

        # Kilauan refraksi / specular facet di bagian dalam
        p_sub_1 = fingers[i] * 0.72 + p_bottom * 0.28
        p_sub_2 = fingers[i + 1] * 0.72 + p_bottom * 0.28
        p_sub_bot = p_bottom * 0.50 + (fingers[i] + fingers[i + 1]) * 0.25
        pts_specular = np.array([p_sub_bot, p_sub_1, p_sub_2], dtype=np.int32)
        spec_color = tuple(min(255, int(c * 0.65 + 95)) for c in color)
        cv2.fillPoly(overlay, [pts_specular], spec_color)

    # Blend translusen agar tetap terlihat efek kaca/kristal bercahaya
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, dst=frame)

    # 2. Glowing Neon Wireframe di setiap rusuk jari (Radial + Rim)
    edge_overlay = np.zeros_like(frame)

    # Rusuk radial dari apex bawah ke setiap ujung jari
    for i in range(n_fingers):
        pt_bot = (int(p_bottom[0]), int(p_bottom[1]))
        pt_tip = (int(fingers[i][0]), int(fingers[i][1]))
        cv2.line(edge_overlay, pt_bot, pt_tip, (0, 240, 255), 7, cv2.LINE_AA)
        cv2.line(frame, pt_bot, pt_tip, (255, 255, 255), 2, cv2.LINE_AA)

    # Rusuk rim atas yang menghubungkan antar ujung jari
    for i in range(num_facets):
        pt1 = (int(fingers[i][0]), int(fingers[i][1]))
        pt2 = (int(fingers[i + 1][0]), int(fingers[i + 1][1]))
        cv2.line(edge_overlay, pt1, pt2, (0, 240, 255), 7, cv2.LINE_AA)
        cv2.line(frame, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

    # Optimized downscaled bloom pass untuk wireframe glow (menghemat ~4ms)
    small_edge = cv2.resize(edge_overlay, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    edge_glow_small = cv2.GaussianBlur(small_edge, (11, 11), 0)
    edge_glow = cv2.resize(edge_glow_small, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.addWeighted(frame, 1.0, edge_glow, 1.40, 0, dst=frame)

    # 3. Cincin Jangkar Bercahaya di Ujung-Ujung Tiap Jari & Apex Bawah
    all_anchors = [p_bottom] + [fingers[i] for i in range(n_fingers)]
    for pt in all_anchors:
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(frame, (px, py), 7, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # 4. Sparkles memancar dari titik-titik ujung jari
    if random.random() < 0.65:
        sp = random.choice(all_anchors)
        particles.append(
            SparkleParticle(
                sp[0] + random.uniform(-12, 12),
                sp[1] + random.uniform(-12, 12),
                color=random.choice([(255, 255, 255), (0, 240, 255), (255, 215, 60), (220, 100, 255)]),
            )
        )


# ================= GESTURE 5: EFEK KETAPEL (SLINGSHOT TO CAMERA & WIPE BLUR) =================

def detect_slingshot_hands(hands_data):
    """
    Mendeteksi apakah terdapat 2 tangan yang membentuk pose ketapel:
    - Satu tangan sebagai Cabang / Garpu Ketapel (Fork):
      Bisa berupa pose 'V' (telunjuk & tengah berdiri, jari lain terlipat) ATAU pose 'L' (jempol & telunjuk terbuka).
    - Satu tangan sebagai Penarik Karet (Pinch):
      Ujung jempol dan telunjuk saling menjepit (jarak < 55 px).
    Mengembalikan: (is_slingshot, fork_data, pinch_data) atau (False, None, None)
    """
    if len(hands_data) < 2:
        return False, None, None

    for i in range(len(hands_data)):
        for j in range(len(hands_data)):
            if i == j:
                continue
            h_fork = hands_data[i]
            h_pinch = hands_data[j]

            lm_f, is_r_f, to_s_f = h_fork
            lm_p, is_r_p, to_s_p = h_pinch

            # 1. Periksa apakah h_fork adalah garpu ketapel murni
            # HANYA Pose 'V' / Peace (Telunjuk & Tengah terentang terbuka, Manis & Kelingking terlipat)
            # *Pose 'L' sengaja ditiadakan agar TIDAK BENTROK dengan Persegi Panjang (Gesture 3)*
            v_shape = (
                not is_finger_folded(lm_f, INDEX_TIP, INDEX_PIP)
                and not is_finger_folded(lm_f, MIDDLE_TIP, MIDDLE_PIP)
                and is_finger_folded(lm_f, RING_TIP, RING_PIP)
                and is_finger_folded(lm_f, PINKY_TIP, PINKY_PIP)
            )

            if not v_shape:
                continue

            # Tentukan dua ujung cabang ketapel (Telunjuk & Jari Tengah)
            pt_idx = np.array(to_s_f(lm_f[INDEX_TIP]), dtype=np.float32)
            pt_mid = np.array(to_s_f(lm_f[MIDDLE_TIP]), dtype=np.float32)

            prong_dist = float(np.linalg.norm(pt_idx - pt_mid))
            # Cabang 'V' harus terbuka renggang
            if prong_dist < 30.0:
                continue

            fork_center = (pt_idx + pt_mid) * 0.5
            fork_base = np.array(to_s_f(lm_f[WRIST]), dtype=np.float32)

            # 2. Periksa apakah h_pinch adalah tangan penjepit yang rapat
            p_thumb = np.array(to_s_p(lm_p[THUMB_TIP]), dtype=np.float32)
            p_index = np.array(to_s_p(lm_p[INDEX_TIP]), dtype=np.float32)
            pinch_dist = float(np.linalg.norm(p_thumb - p_index))

            # Cubitan harus benar-benar rapat (< 42 px)
            is_pinching = pinch_dist < 42.0
            pinch_pt = (p_thumb + p_index) * 0.5

            stretch_dist = float(np.linalg.norm(pinch_pt - fork_center))

            fork_data = {
                "p1": pt_idx,
                "p2": pt_mid,
                "center": fork_center,
                "base": fork_base,
            }
            pinch_data = {
                "pt": pinch_pt,
                "is_pinching": is_pinching,
                "pinch_dist": pinch_dist,
                "stretch_dist": stretch_dist,
            }

            return True, fork_data, pinch_data

    return False, None, None


class SlingshotManager:
    """
    Mengelola seluruh siklus hidup Efek Ketapel & Usap Kamera:
    - AIMING: Membidik & meregangkan karet ketapel bercahaya neon.
    - FLYING: Peluru melesat membesar menuju lensa kamera (3D zoom & speed lines).
    - IMPACT: Guncangan layar (screen shake), flash, dan munculnya efek blur / embun (fog).
    - WIPING: Deteksi gerakan tangan mengusap kamera untuk menghapus embun secara real-time.
    """

    def __init__(self):
        self.state = "IDLE"  # IDLE, AIMING, FLYING
        self.aim_fork = None
        self.aim_pinch = None
        self.last_aim_time = 0.0
        self.max_stretch = 0.0

        # Bullet animation
        self.bullet_start = None
        self.bullet_target = None
        self.bullet_start_time = 0.0
        self.bullet_duration = 0.26  # detik

        # Fog / Blur state
        self.fog_active = False
        self.fog_mask = None
        self.screen_shake = 0
        self.white_flash = 0.0
        self.prev_wipes = {}
        self.clean_shimmer_timer = 0.0

        # Preallocated buffers untuk rendering blur super cepat
        self.cached_shape = None
        self.white_tint_small = None
        self.p1 = None
        self.p2 = None

    def reset(self):
        self.state = "IDLE"
        self.fog_active = False
        self.fog_mask = None
        self.max_stretch = 0.0
        self.screen_shake = 0
        self.white_flash = 0.0
        self.prev_wipes.clear()

    def update_and_render(self, frame, hands_data, foreground_particles, dt, suppress_aiming=False, show_text=True):
        h, w = frame.shape[:2]
        now = time.time()
        detected_text = None
        is_busy = False

        if self.cached_shape != (h, w):
            self.cached_shape = (h, w)
            sw, sh = max(1, w // 4), max(1, h // 4)
            self.white_tint_small = np.full((sh, sw, 3), 245, dtype=np.uint8)
            self.p1 = np.empty((h, w, 3), dtype=np.uint8)
            self.p2 = np.empty((h, w, 3), dtype=np.uint8)

        # ---------------- 1. KETIKA FOG / EMBUN SEDANG AKTIF ----------------
        if self.fog_active:
            is_busy = True
            if self.fog_mask is None or self.fog_mask.shape != (h, w):
                self.fog_mask = np.full((h, w), 255, dtype=np.uint8)

            detected_text = "Usap Kamera untuk Bersihkan! 🖐️"

            # Deteksi sapuan telapak tangan pengguna untuk membersihkan embun
            current_hand_ids = set()
            for hand_idx, (lm, is_r, to_s) in enumerate(hands_data):
                current_hand_ids.add(hand_idx)
                palm_pt = np.array(to_s(lm[MIDDLE_MCP]), dtype=np.float32)
                wrist_pt = np.array(to_s(lm[WRIST]), dtype=np.float32)

                prev_pt = self.prev_wipes.get(hand_idx, None)
                if prev_pt is not None:
                    # Gambar sapuan garis tebal dari posisi sebelumnya ke sekarang
                    cv2.line(
                        self.fog_mask,
                        (int(prev_pt[0]), int(prev_pt[1])),
                        (int(palm_pt[0]), int(palm_pt[1])),
                        0,
                        thickness=140,
                    )

                # Sapuan lingkaran di telapak tangan dan pangkal pergelangan
                cv2.circle(self.fog_mask, (int(palm_pt[0]), int(palm_pt[1])), 80, 0, -1)
                mid_wrist = (palm_pt + wrist_pt) * 0.5
                cv2.circle(self.fog_mask, (int(mid_wrist[0]), int(mid_wrist[1])), 70, 0, -1)

                # Ujung jari juga ikut mengusap
                for tip_id in [INDEX_TIP, MIDDLE_TIP, THUMB_TIP, RING_TIP, PINKY_TIP]:
                    t_pt = to_s(lm[tip_id])
                    cv2.circle(self.fog_mask, (int(t_pt[0]), int(t_pt[1])), 45, 0, -1)

                self.prev_wipes[hand_idx] = palm_pt

                # Partikel tetesan embun yang terhapus
                if random.random() < 0.40:
                    foreground_particles.append(
                        SparkleParticle(
                            palm_pt[0] + random.uniform(-40, 40),
                            palm_pt[1] + random.uniform(-40, 40),
                            color=(255, 255, 255),
                        )
                    )

            # Hapus ID tangan yang sudah tidak tampak
            stale_ids = set(self.prev_wipes.keys()) - current_hand_ids
            for sid in stale_ids:
                del self.prev_wipes[sid]

            # Hitung persentase embun tersisa secara cepat
            small_mask = cv2.resize(self.fog_mask, (160, 90))
            fog_ratio = float(np.count_nonzero(small_mask > 30)) / (160 * 90)

            if fog_ratio < 0.07:
                # Kaca sudah bersih tuntas!
                self.fog_active = False
                self.fog_mask = None
                self.clean_shimmer_timer = now + 1.5
                for _ in range(32):
                    foreground_particles.append(
                        SparkleParticle(
                            random.uniform(40, w - 40),
                            random.uniform(40, h - 40),
                            color=random.choice([(0, 240, 255), (255, 255, 255), (255, 215, 60)]),
                        )
                    )
            else:
                # Render efek blur/frosted glass kamera berkecepatan tinggi (SIMD-accelerated)
                sw, sh = max(1, w // 4), max(1, h // 4)
                small_frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
                blurred_sm = cv2.blur(small_frame, (15, 15))
                frosted_sm = cv2.addWeighted(blurred_sm, 0.88, self.white_tint_small, 0.12, 0)
                frosted_glass = cv2.resize(frosted_sm, (w, h), interpolation=cv2.INTER_LINEAR)

                # Soft mask cepat via downscale-blur-upscale untuk tepian sapuan halus alami
                sm_mask = cv2.resize(self.fog_mask, (sw, sh), interpolation=cv2.INTER_AREA)
                sm_mask = cv2.blur(sm_mask, (7, 7))
                soft_mask = cv2.resize(sm_mask, (w, h), interpolation=cv2.INTER_LINEAR)

                # SIMD Blending super cepat menggunakan C++ OpenCV (< 1.5 ms)
                mask_3c = cv2.merge([soft_mask, soft_mask, soft_mask])
                inv_mask_3c = cv2.bitwise_not(mask_3c)
                cv2.multiply(frosted_glass, mask_3c, self.p1, scale=1.0 / 255.0)
                cv2.multiply(frame, inv_mask_3c, self.p2, scale=1.0 / 255.0)
                cv2.add(self.p1, self.p2, dst=frame)

                # Tampilkan info persentase embun tersisa di bawah
                if show_text:
                    pct_clean = int((1.0 - fog_ratio) * 100)
                    cv2.putText(frame, f"Kebersihan Lensa: {pct_clean}%", (w // 2 - 120, h - 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, f"Kebersihan Lensa: {pct_clean}%", (w // 2 - 120, h - 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 240, 255), 1, cv2.LINE_AA)

            return detected_text, is_busy

        # Bersihkan riwayat usap jika fog tidak aktif
        self.prev_wipes.clear()

        if self.clean_shimmer_timer > now and show_text:
            cv2.putText(frame, "Lensa Kamera Bersih! ✨", (w // 2 - 160, 110), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "Lensa Kamera Bersih! ✨", (w // 2 - 160, 110), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 220), 1, cv2.LINE_AA)

        # ---------------- 2. KETIKA PELURU SEDANG MELESAT KE KAMERA (FLYING) ----------------
        if self.state == "FLYING":
            is_busy = True
            progress = min(1.0, (now - self.bullet_start_time) / self.bullet_duration)
            detected_text = "Peluru Ketapel Melesat! 🚀"

            # Interpolasi posisi menuju lensa kamera (tengah layar)
            t_ease = progress * progress  # Percepatan perspektif mendekat
            curr_pos = self.bullet_start * (1.0 - progress) + self.bullet_target * progress
            curr_radius = int(18 + (340 - 18) * t_ease)

            # Efek speed lines radial memancar dari pusat kamera
            center_pt = (w // 2, h // 2)
            for _ in range(8):
                angle = random.uniform(0, 2 * math.pi)
                r_in = random.uniform(curr_radius * 0.5, curr_radius * 1.2)
                r_out = r_in + random.uniform(80, 220)
                p1 = (int(center_pt[0] + math.cos(angle) * r_in), int(center_pt[1] + math.sin(angle) * r_in))
                p2 = (int(center_pt[0] + math.cos(angle) * r_out), int(center_pt[1] + math.sin(angle) * r_out))
                cv2.line(frame, p1, p2, (0, 240, 255), 2, cv2.LINE_AA)

            # Peluru energi bercahaya
            bx, by = int(curr_pos[0]), int(curr_pos[1])
            # Outer glow
            glow_surf = np.zeros_like(frame)
            cv2.circle(glow_surf, (bx, by), curr_radius, (0, 200, 255), -1, cv2.LINE_AA)
            glow_surf = cv2.GaussianBlur(glow_surf, (25, 25), 0)
            cv2.addWeighted(frame, 1.0, glow_surf, 0.85, 0, dst=frame)

            # Inner bright sphere
            cv2.circle(frame, (bx, by), max(5, int(curr_radius * 0.65)), (100, 240, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), max(2, int(curr_radius * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)

            # Trail particles
            for _ in range(4):
                foreground_particles.append(
                    SparkleParticle(
                        bx + random.uniform(-curr_radius * 0.5, curr_radius * 0.5),
                        by + random.uniform(-curr_radius * 0.5, curr_radius * 0.5),
                        color=random.choice([(0, 240, 255), (255, 255, 255), (255, 215, 60)]),
                    )
                )

            if progress >= 1.0:
                # TABRAKAN DI LENSA KAMERA (IMPACT!)
                self.state = "IDLE"
                self.fog_active = True
                self.fog_mask = np.full((h, w), 255, dtype=np.uint8)
                self.screen_shake = 10
                self.white_flash = 0.50

            return detected_text, is_busy

        # ---------------- 3. DETEKSI POSE KETAPEL (AIMING / PULLING) ----------------
        if not suppress_aiming:
            is_slingshot, fork_data, pinch_data = detect_slingshot_hands(hands_data)

            if is_slingshot and pinch_data["is_pinching"]:
                stretch_dist = pinch_data["stretch_dist"]

                if stretch_dist > 75.0:
                    self.state = "AIMING"
                    is_busy = True
                    self.aim_fork = fork_data
                    self.aim_pinch = pinch_data
                    self.last_aim_time = now
                    self.max_stretch = max(self.max_stretch, stretch_dist)

                    pct_power = min(100, int((stretch_dist - 75.0) / 180.0 * 100))
                    detected_text = f"Ketapel [Tarik: {pct_power}% 🏹]"

                    # Visual Cabang Ketapel (Fork)
                    p1 = (int(fork_data["p1"][0]), int(fork_data["p1"][1]))
                    p2 = (int(fork_data["p2"][0]), int(fork_data["p2"][1]))
                    p_center = (int(fork_data["center"][0]), int(fork_data["center"][1]))
                    p_base = (int(fork_data["base"][0]), int(fork_data["base"][1]))

                    # Batang ketapel
                    cv2.line(frame, p_base, p_center, (30, 160, 220), 6, cv2.LINE_AA)
                    cv2.line(frame, p_center, p1, (40, 200, 255), 5, cv2.LINE_AA)
                    cv2.line(frame, p_center, p2, (40, 200, 255), 5, cv2.LINE_AA)

                    # Cincin cabang
                    cv2.circle(frame, p1, 8, (0, 240, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, p2, 8, (0, 240, 255), -1, cv2.LINE_AA)

                    # Karet Ketapel Elastis (Neon Glowing Rubber Bands)
                    pinch_pt = (int(pinch_data["pt"][0]), int(pinch_data["pt"][1]))

                    band_overlay = np.zeros_like(frame)
                    cv2.line(band_overlay, p1, pinch_pt, (0, 215, 255), 7, cv2.LINE_AA)
                    cv2.line(frame, p1, pinch_pt, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.line(band_overlay, p2, pinch_pt, (0, 215, 255), 7, cv2.LINE_AA)
                    cv2.line(frame, p2, pinch_pt, (255, 255, 255), 2, cv2.LINE_AA)

                    small_band = cv2.resize(band_overlay, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
                    band_glow_sm = cv2.GaussianBlur(small_band, (9, 9), 0)
                    band_glow = cv2.resize(band_glow_sm, (w, h), interpolation=cv2.INTER_LINEAR)
                    cv2.addWeighted(frame, 1.0, band_glow, 1.30, 0, dst=frame)

                    # Peluru Energi di Kantung Ketapel
                    bullet_r = int(12 + (stretch_dist / 250.0) * 10)
                    cv2.circle(frame, pinch_pt, bullet_r + 4, (0, 240, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, pinch_pt, bullet_r, (40, 180, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, pinch_pt, int(bullet_r * 0.4), (255, 255, 255), -1, cv2.LINE_AA)

                    # Percikan energi di peluru
                    if random.random() < 0.60:
                        foreground_particles.append(
                            SparkleParticle(
                                pinch_pt[0] + random.uniform(-15, 15),
                                pinch_pt[1] + random.uniform(-15, 15),
                                color=(0, 240, 255),
                            )
                        )

                    return detected_text, is_busy

        # ---------------- 4. DETEKSI PELEPASAN PELURU (RELEASE / FIRE) ----------------
        if self.state == "AIMING":
            # Jika sebelumnya membidik dan teregang cukup jauh (> 90 px), dan sekarang cubitan dibuka atau dilepas
            if self.aim_pinch is not None and self.max_stretch >= 90.0:
                # RELEASE!
                self.state = "FLYING"
                is_busy = True
                self.bullet_start = self.aim_pinch["pt"].copy()
                self.bullet_target = np.array([w // 2, h // 2], dtype=np.float32)
                self.bullet_start_time = now
                self.max_stretch = 0.0
                detected_text = "Peluru Ketapel Ditembakkan! 🚀"
                return detected_text, is_busy
            else:
                self.state = "IDLE"
                self.max_stretch = 0.0

        # Screen Shake jika ada
        if self.screen_shake > 0:
            dx = random.randint(-self.screen_shake, self.screen_shake)
            dy = random.randint(-self.screen_shake, self.screen_shake)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            cv2.warpAffine(frame, M, (w, h), dst=frame, borderMode=cv2.BORDER_REFLECT)
            self.screen_shake -= 1

        # White flash jika ada (zero-allocation scale)
        if self.white_flash > 0.0:
            cv2.convertScaleAbs(frame, dst=frame, alpha=1.0 - self.white_flash, beta=self.white_flash * 255)
            self.white_flash = max(0.0, self.white_flash - 0.12)

        return detected_text, is_busy


# ----------------- GESTURE 6: BLACK & WHITE FILTER MANAGER (PUSH FROM TOP) -----------------
class BlackWhiteFilterManager:
    """
    Mengelola Efek Filter Black & White (B&W) Sinematik dengan Animasi Tirai Push dari Atas:
    - Deteksi gesture: kedua tangan menarik dari atas ke bawah (two-hand pull down).
    - Transisi: animasi tirai pemisah / laser push bergerak halus dari y=0 ke y=h (atas ke bawah).
    - Filter: Black & White monokrom sinematik berkecepatan tinggi (< 1.2 ms).
    - Toggle: menarik ke bawah lagi akan mengembalikan filter warna (push dari atas juga).
    """

    def __init__(self):
        self.is_bw = False
        self.in_transition = False
        self.target_bw = False
        self.transition_start_time = 0.0
        self.transition_duration = 0.65  # detik (animasi responsif dan mulus)
        self.last_trigger_time = 0.0
        self.hand_history = []  # list of (timestamp, y_left, y_right, x_left, x_right)

    def reset(self):
        self.is_bw = False
        self.in_transition = False
        self.target_bw = False
        self.hand_history.clear()
        self.last_trigger_time = 0.0

    def trigger_toggle(self, now, is_blur_active=True):
        """Memicu transisi push filter dari atas (toggle B&W <-> Color). Hanya jika blur aktif."""
        if not is_blur_active:
            return
        self.in_transition = True
        self.target_bw = not self.is_bw
        self.transition_start_time = now
        self.last_trigger_time = now

    def update_and_detect_gesture(self, hands_data, now, is_blur_active=False, is_other_gesture_busy=False):
        """
        Mendeteksi apakah kedua tangan melakukan gerakan menarik dari atas ke bawah.
        Syarat: Efek B&W HANYA aktif jika efek blur (embun kamera dari ketapel) sedang aktif!
        """
        detected_text = None

        # Jika blur tidak aktif, B&W tidak dapat diaktifkan dan langsung di-reset
        if not is_blur_active:
            if self.is_bw or self.in_transition:
                self.reset()
            self.hand_history.clear()
            return detected_text

        if self.in_transition or is_other_gesture_busy:
            self.hand_history.clear()
            return detected_text

        if len(hands_data) < 2:
            self.hand_history.clear()
            return detected_text

        # Pisahkan kedua tangan berdasarkan posisi horizontal (kiri vs kanan)
        h1, h2 = hands_data[0], hands_data[1]
        lm1, _, to_s1 = h1
        lm2, _, to_s2 = h2

        p1_wrist = to_s1(lm1[WRIST])
        p2_wrist = to_s2(lm2[WRIST])

        if p1_wrist[0] <= p2_wrist[0]:
            h_left, h_right = h1, h2
            w_left, w_right = p1_wrist, p2_wrist
        else:
            h_left, h_right = h2, h1
            w_left, w_right = p2_wrist, p1_wrist

        # Kedua tangan harus memiliki jarak horizontal yang cukup (bukan bertumpuk/bersentuhan)
        dx = w_right[0] - w_left[0]
        if dx < 90:
            self.hand_history.clear()
            return detected_text

        lm_l, _, to_sl = h_left
        lm_r, _, to_sr = h_right

        mcp_l = to_sl(lm_l[MIDDLE_MCP])
        mcp_r = to_sr(lm_r[MIDDLE_MCP])

        y_left = (w_left[1] + mcp_l[1]) * 0.5
        y_right = (w_right[1] + mcp_r[1]) * 0.5
        x_left = (w_left[0] + mcp_l[0]) * 0.5
        x_right = (w_right[0] + mcp_r[0]) * 0.5

        # Simpan ke riwayat pergerakan tangan
        self.hand_history.append((now, y_left, y_right, x_left, x_right))

        # Pangkas riwayat yang lebih lama dari 0.45 detik
        self.hand_history = [entry for entry in self.hand_history if now - entry[0] <= 0.45]

        # Evaluasi gerakan menarik ke bawah jika ada riwayat minimal ~0.10 detik (3-4 frame)
        if len(self.hand_history) >= 3 and (now - self.hand_history[0][0]) >= 0.10:
            t_old, yl_old, yr_old, xl_old, xr_old = self.hand_history[0]
            dt = now - t_old

            dy_l = y_left - yl_old
            dy_r = y_right - yr_old

            v_l = dy_l / dt
            v_r = dy_r / dt

            # Syarat gerakan menarik ke bawah:
            # 1. Kedua tangan bergerak ke bawah secara bersamaan (> 55 px)
            # 2. Kecepatan ke bawah cukup tinggi (> 150 px/detik)
            # 3. Cooldown minimal 1.2 detik sejak aktivasi terakhir
            if (
                dy_l > 55.0
                and dy_r > 55.0
                and v_l > 150.0
                and v_r > 150.0
                and (now - self.last_trigger_time) > 1.20
            ):
                self.trigger_toggle(now)
                self.hand_history.clear()
                action_name = "B&W 🎬" if self.target_bw else "Color 🌈"
                detected_text = f"Tarik Bawah: Push Filter {action_name} ⬇️"
                return detected_text

        return detected_text

    def render_and_apply(self, frame, foreground_particles, now, is_blur_active=False):
        """
        Menerapkan filter Black & White dan merender animasi push dari atas.
        Hanya diaplikasikan jika efek blur (embun kamera) sedang aktif!
        """
        h, w = frame.shape[:2]
        detected_text = None

        if not is_blur_active:
            if self.is_bw or self.in_transition:
                self.reset()
            return detected_text

        if self.in_transition:
            raw_t = min(1.0, (now - self.transition_start_time) / self.transition_duration)
            # Smoothstep curve untuk gerakan tirai yang alami
            t = raw_t * raw_t * (3.0 - 2.0 * raw_t)
            split_y = int(t * h)

            # Siapkan versi Black & White monokrom sinematik
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.convertScaleAbs(bw, dst=bw, alpha=1.06, beta=-6)

            if self.target_bw:
                # Push B&W dari atas ke bawah:
                # Bagian 0 -> split_y: Black & White
                # Bagian split_y -> h: Warna Asli
                frame[:split_y, :] = bw[:split_y, :]
                detected_text = f"Push Filter B&W: {int(raw_t * 100)}% ⬇️"
            else:
                # Push Warna Asli dari atas ke bawah (menggantikan B&W):
                # Bagian 0 -> split_y: Warna Asli
                # Bagian split_y -> h: Black & White
                frame[split_y:, :] = bw[split_y:, :]
                detected_text = f"Push Filter Color: {int(raw_t * 100)}% ⬇️"

            # Render Garis Pembatas Laser / Tirai di posisi split_y
            if 0 < split_y < h:
                # Laser beam putih di tengah
                cv2.line(frame, (0, split_y), (w, split_y), (255, 255, 255), 3, cv2.LINE_AA)
                # Glow neon cyan di sisi atas dan bawah
                if split_y > 1:
                    cv2.line(frame, (0, split_y - 1), (w, split_y - 1), (0, 240, 255), 1, cv2.LINE_AA)
                if split_y < h - 1:
                    cv2.line(frame, (0, split_y + 1), (w, split_y + 1), (0, 240, 255), 1, cv2.LINE_AA)

                # Percikan laser sparkle di sepanjang garis tirai yang bergerak turun
                for _ in range(2):
                    foreground_particles.append(
                        SparkleParticle(
                            random.uniform(20, w - 20),
                            float(split_y + random.uniform(-3, 3)),
                            color=random.choice([(255, 255, 255), (0, 240, 255), (255, 215, 60)]),
                        )
                    )

            if raw_t >= 1.0:
                self.in_transition = False
                self.is_bw = self.target_bw

        elif self.is_bw:
            # Filter Black & White aktif secara penuh
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.convertScaleAbs(bw, dst=frame, alpha=1.06, beta=-6)
            detected_text = "Filter: Black & White [🎬]"

        return detected_text


# ----------------- DRAW HUD -----------------
def draw_hud(
    frame,
    active_gesture_text,
    fps,
    gesture1_unlocked,
    frame_active,
    crystal_active=False,
    slingshot_busy=False,
    fog_active=False,
    is_bw=False,
    is_bw_transition=False,
    show_text=True,
):
    """Menampilkan status gesture, status unlock bunga, dan FPS dengan tampilan negatif/kontras tinggi."""
    if not show_text:
        return

    h, w = frame.shape[:2]

    # Pill Kiri: Status Gesture & Mode
    pill_text = f"Mode: {active_gesture_text}" if active_gesture_text else "Arahkan tangan ke kamera..."
    pill_color = (40, 40, 40)
    text_color = (255, 255, 255)

    if "B&W" in pill_text or "Black & White" in pill_text or "Push Filter" in pill_text:
        pill_color = (25, 25, 25)
        text_color = (250, 250, 250)
    elif "Kristal" in pill_text or "Crystal" in pill_text:
        pill_color = (80, 40, 100)  # Kristal Holografik Ungu-Emas
        text_color = (255, 235, 140)
    elif "Ketapel" in pill_text or "Peluru" in pill_text:
        pill_color = (25, 75, 120)  # Neon Cyan-Blue Ketapel
        text_color = (120, 245, 255)
    elif "Usap" in pill_text or "Lensa" in pill_text:
        pill_color = (60, 60, 60)   # Foggy Grey
        text_color = (255, 255, 255)
    elif "Persegi" in pill_text or "Twist" in pill_text or "Frame" in pill_text:
        pill_color = (80, 60, 20)
        text_color = (255, 240, 150)
    elif "Love" in pill_text:
        pill_color = (50, 20, 120)
        text_color = (180, 200, 255)
    elif "Bunga" in pill_text:
        if "Terkunci" in pill_text:
            pill_color = (30, 30, 80)
            text_color = (200, 180, 255)
        else:
            pill_color = (60, 80, 20)
            text_color = (200, 255, 220)

    overlay = frame.copy()
    left_w = min(530, int(w * 0.55))
    cv2.rectangle(overlay, (16, 16), (left_w, 72), pill_color, -1, cv2.LINE_AA)

    # Pill Kanan: FPS & Shortcut dengan gaya warna Negatif / Kontras Tinggi
    right_w = min(460, int(w * 0.40))
    rx1 = w - right_w - 16
    rx2 = w - 16
    cv2.rectangle(overlay, (rx1, 16), (rx2, 72), (15, 15, 15), -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Border badge
    cv2.rectangle(frame, (16, 16), (left_w, 72), (140, 140, 140), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (rx1, 16), (rx2, 72), (0, 255, 255), 1, cv2.LINE_AA)

    # Teks Pill Kiri
    cv2.putText(frame, pill_text, (28, 44), cv2.FONT_HERSHEY_DUPLEX, 0.58, text_color, 1, cv2.LINE_AA)
    status_sub = []
    status_sub.append("Bunga: [UNLOCKED 🌸]" if gesture1_unlocked else "Bunga: [TERKUNCI 🔒]")
    if fog_active:
        if is_bw_transition:
            status_sub.append("Filter: [PUSH ⬇️]")
        elif is_bw:
            status_sub.append("Filter: [B&W 🎬]")
        else:
            status_sub.append("Filter: [B&W Siap ⬇️]")
        status_sub.append("Kamera: [BEREMBUN 🌫️]")
    elif slingshot_busy:
        status_sub.append("Ketapel: [AKTIF 🏹]")
    elif crystal_active:
        status_sub.append("Kristal 3D: [AKTIF 💎]")
    else:
        status_sub.append("Persegi: [AKTIF 📐]" if frame_active else "Persegi: [👉👈]")
    status_line = " | ".join(status_sub)
    cv2.putText(frame, status_line, (28, 64), cv2.FONT_HERSHEY_PLAIN, 0.85, (170, 255, 200), 1, cv2.LINE_AA)

    # Teks Pill Kanan - FPS Warna Negatif (High-Contrast Inverted Neon + Dark Outline)
    fps_val = int(round(fps))
    if 29.5 <= fps <= 30.5:
        fps_val = 30
    # Warna dinamis menyala: hijau neon (lancar), kuning (sedang), merah (turun)
    fps_color = (0, 255, 120) if fps_val >= 28 else (0, 215, 255) if fps_val >= 18 else (0, 80, 255)
    fps_text = f"FPS: {fps_val}"

    # Double pass: outline hitam tebal + teks warna neon terang (warna negatif kontras)
    cv2.putText(frame, fps_text, (rx1 + 16, 42), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (rx1 + 16, 42), cv2.FONT_HERSHEY_DUPLEX, 0.65, fps_color, 1, cv2.LINE_AA)

    # Indikator titik status FPS
    dot_x = rx1 + 16 + int(len(fps_text) * 14) + 12
    cv2.circle(frame, (dot_x, 37), 5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, (dot_x, 37), 4, fps_color, -1, cv2.LINE_AA)

    # Tombol shortcut kontrol
    cv2.putText(
        frame,
        "ESC: Out | H: Hide Text | B: B&W | R: Reset",
        (rx1 + 16, 64),
        cv2.FONT_HERSHEY_PLAIN,
        0.82,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


# ----------------- MAIN PROGRAM -----------------
def main():
    ensure_models()
    sprites = load_sprites()

    # Inisialisasi MediaPipe Tasks HandLandmarker
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # Inisialisasi Background Segmenter untuk efek bunga di belakang subjek
    segmenter = AsyncSegmenter(SEG_MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Gagal membuka webcam! Pastikan kamera terpasang dan izin aktif.")
        segmenter.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Dua wadah partikel terpisah:
    # 1. flower_particles: partikel bunga fullscreen yang dirender di BELAKANG subjek
    # 2. foreground_particles: partikel love/hati dan kilau telunjuk di DEPAN tangan
    flower_particles = []
    foreground_particles = []

    TARGET_FPS = 30.0
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS

    prev_time = time.perf_counter()
    prev_frame_end = time.perf_counter()
    start_time = time.perf_counter()
    fps_smooth = 30.0
    show_landmarks = False
    show_ui_text = True

    gesture1_unlocked = False
    frame_active = False

    smoothed_corners = None
    smoothed_angle = None
    frame_smoothing_factor = 0.38

    smoothed_crystal_data = None
    crystal_smoothing_factor = 0.35

    slingshot_manager = SlingshotManager()
    bw_filter_manager = BlackWhiteFilterManager()

    print("\n=======================================================")
    print("Filter Berhasil Dijalankan (Locked 30 FPS)!")
    print("1. Tangan KANAN Nunjuk ke ATAS -> Efek Hati (Love ❤️) + UNLOCK Efek Bunga")
    print("2. JEMPOL Nunjuk DIRI SENDIRI   -> Efek Bunga FULLSCREEN di BELAKANG SUBJEK 🌸")
    print("3. TELUNJUK Kiri & Kanan SENTUH -> Persegi Panjang Muncul & Bisa di-TWIST 360° 📐")
    print("4. TELAPAK BERSENTUHAN & MEKAR  -> Objek 3D Kristal Prisma Holografik 💎")
    print("5. EFEK KETAPEL & USAP KAMERA   -> Ketapel ke Kamera (Frosted Blur) & Usap untuk Bersihkan! 🏹🖐️")
    print("6. DUA TANGAN TARIK KE BAWAH    -> Filter Black & White (B&W) Sinematik Push dari Atas 🎬⬇️")
    print("Tombol: 'ESC' keluar, 'H' hide/tampil teks, 'B' toggle B&W, 'R' reset, 'L' landmark, 'C' clear.")
    print("=======================================================\n")

    while cap.isOpened():
        loop_start = time.perf_counter()
        ok, raw_frame = cap.read()
        if not ok:
            break

        h, w, _ = raw_frame.shape

        # Downscale frame khusus untuk input inferensi AI HandLandmarker (menghemat ~12ms tanpa mengurangi akurasi koordinat)
        ai_w, ai_h = 640, 360
        det_frame = cv2.resize(raw_frame, (ai_w, ai_h), interpolation=cv2.INTER_LINEAR)
        rgb_det = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_det)
        timestamp_ms = int((time.perf_counter() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        display_frame = cv2.flip(raw_frame, 1)

        # Kirim frame mirrored ke segmenter secara On-Demand (hanya jika efek bunga sedang/akan aktif)
        if flower_particles or gesture1_unlocked:
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            segmenter.update_frame(display_rgb)

        now = time.perf_counter()
        dt = max(0.001, now - prev_time)
        prev_time = now

        detected_gestures = []
        hands_data = []

        crystal_active = False

        if result.hand_landmarks:
            for i, lm in enumerate(result.hand_landmarks):
                is_right_hand = False
                if result.handedness and i < len(result.handedness):
                    hand_cat = result.handedness[i][0].category_name
                    is_right_hand = (hand_cat == "Right")

                def make_to_screen(w_frame, h_frame):
                    return lambda pt: (int((1.0 - pt.x) * w_frame), int(pt.y * h_frame))

                to_screen = make_to_screen(w, h)
                hands_data.append((lm, is_right_hand, to_screen))

                if show_landmarks:
                    for pt in lm:
                        sx, sy = to_screen(pt)
                        cv2.circle(display_frame, (sx, sy), 3, (0, 255, 255), -1)

        # Cek apakah telunjuk sedang bersentuhan atau gesture persegi/kristal sedang aktif
        is_index_touching = False
        if len(hands_data) >= 2:
            hands_for_frame = [(h_item[0], h_item[2]) for h_item in hands_data[:2]]
            is_index_touching, _, _ = check_index_fingers_touching(hands_for_frame, touch_dist_threshold=55.0)

        suppress_aiming = frame_active or is_index_touching or crystal_active

        # ================= GESTURE 5: EFEK KETAPEL (SLINGSHOT TO CAMERA & WIPE BLUR) =================
        slingshot_text, is_slingshot_busy = slingshot_manager.update_and_render(
            display_frame, hands_data, foreground_particles, dt, suppress_aiming=suppress_aiming, show_text=show_ui_text
        )
        if slingshot_text:
            detected_gestures.append(slingshot_text)

        # Jika sedang membidik ketapel, peluru melesat, atau layar berembun (fog), nonaktifkan gesture lain sementara
        if not is_slingshot_busy:
            # ================= GESTURE 4: OBJEK 3D KRISTAL PRISMA (TELAPAK BERSENTUHAN & JARI MEKAR) =================
            if len(hands_data) >= 2:
                is_cupped, raw_crystal = is_cupped_hands_gesture(hands_data)
                if is_cupped and raw_crystal is not None:
                    crystal_active = True
                    if smoothed_crystal_data is None:
                        smoothed_crystal_data = {k: v.copy() for k, v in raw_crystal.items()}
                    else:
                        for k in ["p_bottom", "idx_l", "idx_r", "pky_l", "pky_r", "fingers"]:
                            if k in raw_crystal:
                                smoothed_crystal_data[k] = (
                                    smoothed_crystal_data[k] * (1.0 - crystal_smoothing_factor)
                                    + raw_crystal[k] * crystal_smoothing_factor
                                )
                    detected_gestures.append("3D Kristal Prisma [💎]")
                    draw_3d_crystal(display_frame, smoothed_crystal_data, foreground_particles)
                else:
                    smoothed_crystal_data = None

            # ================= GESTURE 3: SHAPE PERSEGI PANJANG (BERSENTUHAN LALU MELEBAR) =================
            if not crystal_active and len(hands_data) >= 2:
                hands_for_frame = [(h_item[0], h_item[2]) for h_item in hands_data[:2]]

                is_touching, touch_pt, touch_dist = check_index_fingers_touching(hands_for_frame, touch_dist_threshold=52.0)

                if is_touching:
                    if not frame_active:
                        frame_active = True
                        for _ in range(16):
                            foreground_particles.append(
                                SparkleParticle(
                                    touch_pt[0] + random.uniform(-15, 15),
                                    touch_pt[1] + random.uniform(-15, 15),
                                    color=(255, 255, 255),
                                )
                            )

                if frame_active:
                    raw_corners, raw_angle = get_two_handed_rectangle_corners(hands_for_frame)

                    if raw_corners is not None:
                        raw_corners_np = np.array(raw_corners, dtype=np.float32)

                        if smoothed_corners is None:
                            smoothed_corners = raw_corners_np.copy()
                            smoothed_angle = raw_angle
                        else:
                            smoothed_corners = (
                                smoothed_corners * (1.0 - frame_smoothing_factor)
                                + raw_corners_np * frame_smoothing_factor
                            )
                            diff_ang = (raw_angle - smoothed_angle + 180.0) % 360.0 - 180.0
                            smoothed_angle += diff_ang * frame_smoothing_factor

                        detected_gestures.append(f"Persegi Twist [{int(smoothed_angle % 360)} deg]")
                        draw_dynamic_rectangle(
                            display_frame,
                            smoothed_corners,
                            smoothed_angle,
                            foreground_particles,
                            theme_color=(255, 215, 60),
                            show_text=show_ui_text,
                        )
                else:
                    smoothed_corners = None
                    smoothed_angle = None
            else:
                if not crystal_active:
                    frame_active = False
                    smoothed_corners = None
                    smoothed_angle = None

            # Jika TIDAK sedang menampilkan 3D Kristal atau Persegi Panjang, periksa Gesture 1 dan 2
            if not crystal_active and not frame_active:
                for lm, is_right_hand, to_screen in hands_data:
                    index_tip_screen = to_screen(lm[INDEX_TIP])

                    # ================= GESTURE 1 =================
                    # Tangan Kanan Nunjuk ke Atas -> Muncul Love & Unlock Bunga
                    if is_pointing_up(lm, is_right_hand):
                        gesture1_unlocked = True
                        detected_gestures.append("Tangan Kanan Nunjuk Atas [Love]")

                        heart_spr = sprites.get("heart_small", sprites.get("heart"))
                        if heart_spr is not None and random.random() < 0.45:
                            foreground_particles.append(
                                HeartParticle(
                                    index_tip_screen[0] + random.uniform(-10, 10),
                                    index_tip_screen[1] + random.uniform(-10, 10),
                                    heart_spr,
                                )
                            )

                        if random.random() < 0.40:
                            foreground_particles.append(
                                SparkleParticle(
                                    index_tip_screen[0] + random.uniform(-8, 8),
                                    index_tip_screen[1] + random.uniform(-8, 8),
                                    color=(255, 180, 220),
                                )
                            )

                    # ================= GESTURE 2 =================
                    # Jempol Nunjuk Diri Sendiri -> Efek Bunga FULLSCREEN di BELAKANG SUBJEK
                    elif is_pointing_to_self_with_thumb(lm, is_right_hand):
                        if gesture1_unlocked:
                            detected_gestures.append("Jempol Nunjuk Diri [Bunga Latar Belakang 🌸]")

                            # Spawn kelopak bunga jatuh dari atas secara fullscreen
                            petal_spr = sprites.get("petal_small", sprites.get("petal"))
                            if petal_spr is not None and random.random() < 0.45:
                                flower_particles.append(
                                    PetalParticle(
                                        random.uniform(0, w),
                                        random.uniform(-40, 20),
                                        petal_spr,
                                    )
                                )

                            # Spawn bunga mekar (sakura / bloom) acak di latar belakang
                            flower_choice = random.choice([
                                sprites.get("sakura_small", sprites.get("sakura")),
                                sprites.get("bloom_small", sprites.get("bloom")),
                            ])
                            if flower_choice is not None and random.random() < 0.20:
                                flower_particles.append(
                                    FlowerParticle(
                                        random.uniform(30, w - 30),
                                        random.uniform(30, h - 30),
                                        flower_choice,
                                    )
                                )

                            # Sparkles keemasan di latar belakang
                            if random.random() < 0.35:
                                flower_particles.append(
                                    SparkleParticle(
                                        random.uniform(0, w),
                                        random.uniform(0, h),
                                        color=(255, 235, 160),
                                    )
                                )
                        else:
                            detected_gestures.append("Jempol Nunjuk Diri [Terkunci: Nunjuk Dulu ☝️]")

        # ================= GESTURE 6: DUA TANGAN MENARIK DARI ATAS KE BAWAH (PUSH FILTER B&W) =================
        # Syarat mutlak: Efek B&W HANYA aktif jika efek blur (embun kamera dari ketapel) sedang aktif!
        is_blur_active = slingshot_manager.fog_active
        is_other_busy = (slingshot_manager.state in ("AIMING", "FLYING")) or crystal_active or frame_active or is_index_touching
        bw_gesture_text = bw_filter_manager.update_and_detect_gesture(
            hands_data, now, is_blur_active=is_blur_active, is_other_gesture_busy=is_other_busy
        )
        if bw_gesture_text:
            detected_gestures.append(bw_gesture_text)

        # ================= RENDERING BUNGA DI BELAKANG SUBJEK =================
        # Update partikel bunga (dibatasi 28 partikel agar FPS stabil terkunci di 30)
        flower_particles = [p for p in flower_particles if p.update(dt)]
        if len(flower_particles) > 28:
            flower_particles = flower_particles[-28:]

        # Jika ada partikel bunga yang aktif, render di background di belakang subjek
        if flower_particles:
            person_mask = segmenter.get_mask()

            # Buat kanvas background dan gambar semua partikel bunga
            bg_frame = display_frame.copy()
            for p in flower_particles:
                p.draw(bg_frame)

            if person_mask is not None:
                # Fast C++ compositing via cv2.copyTo (0.88ms vs 35ms float32!)
                if person_mask.dtype != np.uint8:
                    mask_u8 = (person_mask * 255).astype(np.uint8)
                else:
                    mask_u8 = person_mask
                # Tempelkan tubuh pengguna (display_frame) di atas bunga (bg_frame)
                cv2.copyTo(display_frame, mask_u8, bg_frame)
                display_frame = bg_frame
            else:
                display_frame = bg_frame

        # ================= GESTURE 6: APLIKASIKAN FILTER BLACK & WHITE (PUSH DARI ATAS) =================
        # Hanya diaplikasikan jika efek blur sedang aktif!
        bw_render_text = bw_filter_manager.render_and_apply(
            display_frame, foreground_particles, now, is_blur_active=is_blur_active
        )
        if bw_render_text and not bw_gesture_text:
            detected_gestures.append(bw_render_text)

        # ================= RENDERING FOREGROUND PARTICLES =================
        # Partikel love / sparkle di depan tangan pengguna (dibatasi 24 partikel)
        foreground_particles = [p for p in foreground_particles if p.update(dt)]
        if len(foreground_particles) > 24:
            foreground_particles = foreground_particles[-24:]

        for p in foreground_particles:
            p.draw(display_frame)

        # Gambar UI HUD
        status_text = " + ".join(detected_gestures) if detected_gestures else ""
        draw_hud(
            display_frame,
            status_text,
            fps_smooth,
            gesture1_unlocked,
            frame_active,
            crystal_active,
            slingshot_busy=is_slingshot_busy,
            fog_active=slingshot_manager.fog_active,
            is_bw=bw_filter_manager.is_bw,
            is_bw_transition=bw_filter_manager.in_transition,
            show_text=show_ui_text,
        )

        cv2.imshow("Hand Gesture Filter - Love, Bunga, Persegi, 3D Kristal, Ketapel, & Filter B&W", display_frame)

        # Precise Frame Pacer: Sinkronisasi tepat ke ritme 30.0 FPS
        elapsed = time.perf_counter() - loop_start
        sleep_left = TARGET_FRAME_TIME - elapsed
        if sleep_left > 0.002:
            wait_ms = max(1, int(sleep_left * 1000))
        else:
            wait_ms = 1

        key = cv2.waitKey(wait_ms) & 0xFF

        # Pengukuran FPS stabil & konsisten
        now = time.perf_counter()
        actual_frame_dt = max(0.001, now - prev_frame_end)
        prev_frame_end = now
        inst_fps = 1.0 / actual_frame_dt
        fps_smooth = fps_smooth * 0.88 + inst_fps * 0.12

        if key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("h"), ord("H"), ord("t"), ord("T")):
            show_ui_text = not show_ui_text
        elif key in (ord("b"), ord("B")):
            bw_filter_manager.trigger_toggle(now, is_blur_active=slingshot_manager.fog_active)
        elif key in (ord("l"), ord("L")):
            show_landmarks = not show_landmarks
        elif key in (ord("c"), ord("C")):
            flower_particles.clear()
            foreground_particles.clear()
        elif key in (ord("r"), ord("R")):
            gesture1_unlocked = False
            frame_active = False
            smoothed_corners = None
            smoothed_angle = None
            smoothed_crystal_data = None
            slingshot_manager.reset()
            bw_filter_manager.reset()
            flower_particles.clear()
            foreground_particles.clear()

    segmenter.close()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
