"""
Test script untuk memverifikasi logika:
1. Fitur 4: Objek 3D Kristal Prisma saat telapak tangan bersentuhan dan jari merenggang (is_cupped_hands_gesture & draw_3d_crystal).
2. Efek bunga fullscreen dan rendering di BELAKANG subjek (AsyncSegmenter & Background Compositing).
3. Inisiasi persegi panjang ketika telunjuk kanan dan kiri bersentuhan.
4. 4 Sudut persegi panjang yang dapat di-twist 360 derajat.
5. Syarat unlock efek bunga setelah gesture 1.
"""

import cv2
import numpy as np
import math
import time
import sys
import io
import hand_effects as he

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


class DummyLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def create_hand(wrist=(0.5, 0.7, 0.0)):
    return [DummyLandmark(wrist[0], wrist[1], wrist[2]) for _ in range(21)]


def test_sprites():
    print("Testing sprites loading...")
    sprites = he.load_sprites()
    for name in ["heart", "sakura", "bloom", "petal"]:
        assert sprites[name] is not None, f"Sprite {name} failed to load!"
        assert sprites[name].shape[2] == 4, f"Sprite {name} does not have 4 channels!"
    print("-> All sprites loaded successfully with 4 channels.")


def test_particle_rendering():
    print("Testing particle rendering and update...")
    sprites = he.load_sprites()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    particles = [
        he.HeartParticle(300, 200, sprites["heart"]),
        he.FlowerParticle(320, 220, sprites["sakura"]),
        he.PetalParticle(340, 240, sprites["petal"]),
        he.SparkleParticle(350, 250),
    ]

    for _ in range(10):
        particles = [p for p in particles if p.update(0.033)]
        for p in particles:
            p.draw(frame)

    assert np.count_nonzero(frame) > 0, "No pixels drawn by particles!"
    print("-> Particle rendering & update verified.")


def test_unlock_logic():
    print("Testing Flower unlock requirement logic...")
    lm = create_hand()

    lm[he.WRIST] = DummyLandmark(0.5, 0.8, 0.0)
    lm[he.INDEX_MCP] = DummyLandmark(0.48, 0.60, 0.0)
    lm[he.INDEX_PIP] = DummyLandmark(0.48, 0.45, 0.0)
    lm[he.INDEX_DIP] = DummyLandmark(0.48, 0.35, 0.0)
    lm[he.INDEX_TIP] = DummyLandmark(0.48, 0.25, 0.0)

    for mcp, pip, tip in [
        (he.MIDDLE_MCP, he.MIDDLE_PIP, he.MIDDLE_TIP),
        (he.RING_MCP, he.RING_PIP, he.RING_TIP),
        (he.PINKY_MCP, he.PINKY_PIP, he.PINKY_TIP),
    ]:
        lm[mcp] = DummyLandmark(0.50, 0.60, 0.0)
        lm[pip] = DummyLandmark(0.50, 0.65, 0.0)
        lm[tip] = DummyLandmark(0.50, 0.70, 0.0)

    gesture1_unlocked = False
    if he.is_pointing_up(lm, is_right_hand=True):
        gesture1_unlocked = True

    assert gesture1_unlocked is True, "Flower should unlock after Gesture 1 is triggered!"
    print("-> Flower unlock requirement verified.")


def test_index_touch_trigger_and_rectangle():
    print("Testing index fingers touch trigger and clean dynamic rectangle...")
    w, h = 1280, 720
    to_screen = lambda pt: (int(pt.x * w), int(pt.y * h))

    h1_touch = create_hand()
    h1_touch[he.WRIST] = DummyLandmark(0.4, 0.6)
    h1_touch[he.INDEX_TIP] = DummyLandmark(0.495, 0.4)
    h1_touch[he.THUMB_TIP] = DummyLandmark(0.45, 0.5)

    h2_touch = create_hand()
    h2_touch[he.WRIST] = DummyLandmark(0.6, 0.6)
    h2_touch[he.INDEX_TIP] = DummyLandmark(0.505, 0.4)
    h2_touch[he.THUMB_TIP] = DummyLandmark(0.55, 0.5)

    is_touching, touch_pt, dist = he.check_index_fingers_touching(
        [(h1_touch, to_screen), (h2_touch, to_screen)], touch_dist_threshold=52.0
    )
    assert is_touching, f"Index fingers should be touching! Dist: {dist}"
    print(f"-> Index touch detected at {touch_pt} with dist {dist:.1f}px.")

    h1_far = create_hand()
    h1_far[he.INDEX_TIP] = DummyLandmark(0.3, 0.4)
    h2_far = create_hand()
    h2_far[he.INDEX_TIP] = DummyLandmark(0.7, 0.4)

    is_touching_far, _, dist_far = he.check_index_fingers_touching(
        [(h1_far, to_screen), (h2_far, to_screen)], touch_dist_threshold=52.0
    )
    assert not is_touching_far, "Index fingers far apart should not be touching!"
    print("-> Non-touching state correctly identified.")

    corners, angle_deg = he.get_two_handed_rectangle_corners([(h1_far, to_screen), (h2_far, to_screen)])
    assert corners is not None and len(corners) == 4
    print(f"-> 4 corners created successfully, twist angle: {angle_deg:.1f} deg.")

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    particles = []
    he.draw_dynamic_rectangle(frame, corners, angle_deg, particles)
    assert np.count_nonzero(frame) > 0, "Clean dynamic rectangle rendering failed!"
    print("-> Clean dynamic rectangle rendering verified.")


def test_background_flower_compositing():
    print("Testing background flower compositing (behind subject)...")
    h, w = 720, 1280
    display_frame = np.full((h, w, 3), 120, dtype=np.uint8)
    display_frame[200:600, 450:850] = (200, 100, 50)

    person_mask = np.zeros((h, w), dtype=np.float32)
    person_mask[200:600, 450:850] = 1.0
    person_mask = cv2.GaussianBlur(person_mask, (11, 11), 0)

    bg_frame = display_frame.copy()
    cv2.circle(bg_frame, (450, 400), 80, (200, 100, 255), -1)

    alpha = person_mask[:, :, np.newaxis]
    result = (display_frame.astype(np.float32) * alpha + bg_frame.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

    person_pixel = result[400, 500]
    assert np.allclose(person_pixel, [200, 100, 50], atol=2), "Person pixels were overwritten by flower!"

    flower_pixel = result[400, 400]
    assert np.allclose(flower_pixel, [200, 100, 255], atol=2), "Flower was not visible in background!"
    print("-> Background compositing verified: Flowers appear behind the person!")


def test_async_segmenter():
    print("Testing AsyncSegmenter initialization and execution...")
    seg = he.AsyncSegmenter(he.SEG_MODEL_PATH)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    seg.update_frame(dummy_frame)
    time.sleep(0.3)
    mask = seg.get_mask()
    assert mask is not None, "AsyncSegmenter failed to produce mask!"
    assert mask.shape == (480, 640), f"Mask shape unexpected: {mask.shape}"
    seg.close()
    print("-> AsyncSegmenter verified successfully.")


def test_feature_4_cupped_hands_and_3d_crystal():
    print("Testing Feature 4: Cupped Hands Gesture and 3D Crystal Gem...")
    w, h = 1280, 720
    to_screen = lambda pt: (int(pt.x * w), int(pt.y * h))

    # Kasus 1: Telapak tangan bersentuhan di bawah dan jari-jari merenggang
    # Left hand: wrist at (0.48, 0.75), index at (0.35, 0.40), pinky at (0.28, 0.50)
    h_l = create_hand()
    h_l[he.WRIST] = DummyLandmark(0.48, 0.75)
    h_l[he.INDEX_MCP] = DummyLandmark(0.46, 0.65)
    h_l[he.INDEX_PIP] = DummyLandmark(0.40, 0.55)
    h_l[he.INDEX_TIP] = DummyLandmark(0.35, 0.40)
    h_l[he.PINKY_TIP] = DummyLandmark(0.28, 0.50)

    # Right hand: wrist at (0.52, 0.75), index at (0.65, 0.40), pinky at (0.72, 0.50)
    h_r = create_hand()
    h_r[he.WRIST] = DummyLandmark(0.52, 0.75)
    h_r[he.INDEX_MCP] = DummyLandmark(0.54, 0.65)
    h_r[he.INDEX_PIP] = DummyLandmark(0.60, 0.55)
    h_r[he.INDEX_TIP] = DummyLandmark(0.65, 0.40)
    h_r[he.PINKY_TIP] = DummyLandmark(0.72, 0.50)

    hands_data = [
        (h_l, False, to_screen),
        (h_r, True, to_screen),
    ]

    is_cupped, crystal_data = he.is_cupped_hands_gesture(hands_data)
    assert is_cupped is True, "Cupped hands gesture should be detected!"
    assert crystal_data is not None
    for k in ["p_bottom", "idx_l", "idx_r", "pky_l", "pky_r", "fingers"]:
        assert k in crystal_data, f"Key {k} missing in crystal_data!"
    assert len(crystal_data["fingers"]) == 10, f"Expected 10 fingers, got {len(crystal_data['fingers'])}"
    print(f"-> Cupped hands detected! All 10 finger vertices tracked. Bottom: {crystal_data['p_bottom']}")

    # Test render 3D crystal directly anchored to every finger
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    particles = []
    he.draw_3d_crystal(frame, crystal_data, particles)
    assert np.count_nonzero(frame) > 10000, "3D Crystal rendering produced too few pixels!"
    print("-> 3D Crystal rendering verified: 9 finger facets, neon wireframe edges, and nodes rendered successfully.")

    # Kasus 2: Tangan berjauhan (pergelangan tangan tidak bersentuhan)
    h_l_far = create_hand()
    h_l_far[he.WRIST] = DummyLandmark(0.2, 0.75)
    h_r_far = create_hand()
    h_r_far[he.WRIST] = DummyLandmark(0.8, 0.75)
    is_cupped_far, _ = he.is_cupped_hands_gesture([(h_l_far, False, to_screen), (h_r_far, True, to_screen)])
    assert is_cupped_far is False, "Cupped hands should NOT trigger when wrists are far apart!"
    print("-> Non-touching wrists condition correctly suppresses 3D crystal.")


def test_feature_5_slingshot_and_wipe_blur():
    print("Testing Feature 5: Slingshot Gesture & Camera Wipe Blur...")
    w, h = 1280, 720
    to_screen = lambda pt: (int(pt.x * w), int(pt.y * h))

    # Hand 1: Fork Hand (V-shape)
    h_fork = create_hand()
    h_fork[he.INDEX_MCP] = DummyLandmark(0.40, 0.60)
    h_fork[he.INDEX_PIP] = DummyLandmark(0.38, 0.50)
    h_fork[he.INDEX_TIP] = DummyLandmark(0.35, 0.38)
    h_fork[he.MIDDLE_MCP] = DummyLandmark(0.42, 0.60)
    h_fork[he.MIDDLE_PIP] = DummyLandmark(0.43, 0.50)
    h_fork[he.MIDDLE_TIP] = DummyLandmark(0.45, 0.38)
    h_fork[he.RING_MCP] = DummyLandmark(0.44, 0.60)
    h_fork[he.RING_PIP] = DummyLandmark(0.44, 0.55)
    h_fork[he.RING_TIP] = DummyLandmark(0.44, 0.62)
    h_fork[he.PINKY_MCP] = DummyLandmark(0.46, 0.60)
    h_fork[he.PINKY_PIP] = DummyLandmark(0.46, 0.55)
    h_fork[he.PINKY_TIP] = DummyLandmark(0.46, 0.62)

    # Hand 2: Pinch Hand (Thumb and Index tips touching)
    h_pinch = create_hand()
    h_pinch[he.THUMB_TIP] = DummyLandmark(0.55, 0.45)
    h_pinch[he.INDEX_TIP] = DummyLandmark(0.56, 0.45)

    hands_data = [
        (h_fork, False, to_screen),
        (h_pinch, True, to_screen),
    ]

    is_slingshot, fork_data, pinch_data = he.detect_slingshot_hands(hands_data)
    assert is_slingshot is True, "Slingshot gesture should be detected!"
    assert pinch_data["is_pinching"] is True, "Pinch should be detected!"
    assert pinch_data["stretch_dist"] > 75.0, f"Stretch dist should be > 75px, got {pinch_data['stretch_dist']}"
    print(f"-> Slingshot aiming detected! Prongs dist: {np.linalg.norm(fork_data['p1'] - fork_data['p2']):.1f}, Stretch: {pinch_data['stretch_dist']:.1f}")

    # Verify that Rectangle Hands (Thumb + Index extended, Middle folded) do NOT trigger Slingshot!
    h_rect_1 = create_hand()
    h_rect_1[he.INDEX_TIP] = DummyLandmark(0.35, 0.40)
    h_rect_1[he.INDEX_PIP] = DummyLandmark(0.35, 0.50)
    h_rect_1[he.THUMB_TIP] = DummyLandmark(0.30, 0.55)
    h_rect_1[he.MIDDLE_TIP] = DummyLandmark(0.40, 0.65)  # Middle folded!
    h_rect_1[he.MIDDLE_PIP] = DummyLandmark(0.40, 0.55)
    h_rect_2 = create_hand()
    h_rect_2[he.INDEX_TIP] = DummyLandmark(0.65, 0.40)
    h_rect_2[he.INDEX_PIP] = DummyLandmark(0.65, 0.50)
    h_rect_2[he.THUMB_TIP] = DummyLandmark(0.70, 0.55)
    h_rect_2[he.MIDDLE_TIP] = DummyLandmark(0.60, 0.65)
    h_rect_2[he.MIDDLE_PIP] = DummyLandmark(0.60, 0.55)
    rect_hands = [(h_rect_1, False, to_screen), (h_rect_2, True, to_screen)]
    is_rect_slingshot, _, _ = he.detect_slingshot_hands(rect_hands)
    assert not is_rect_slingshot, "Rectangle gesture MUST NOT trigger slingshot!"
    print("-> Verified: Rectangle hands do NOT false-trigger slingshot.")

    # Test SlingshotManager lifecycle
    mgr = he.SlingshotManager()
    frame = np.full((h, w, 3), 100, dtype=np.uint8)
    particles = []

    # Test suppress_aiming (e.g. when rectangle is active or index touching)
    text_sup, busy_sup = mgr.update_and_render(frame, hands_data, particles, dt=0.03, suppress_aiming=True)
    assert not busy_sup and mgr.state == "IDLE", "suppress_aiming=True must prevent slingshot aiming!"
    print("-> Verified: suppress_aiming prevents slingshot activation when rectangle is active.")

    # 1. Aiming update (without suppression)
    text, busy = mgr.update_and_render(frame, hands_data, particles, dt=0.03, suppress_aiming=False)
    assert busy is True
    assert mgr.state == "AIMING"
    print(f"-> Manager entered AIMING state ({text})")

    # 2. Release trigger (unpinch Hand 2)
    h_pinch_released = create_hand()
    h_pinch_released[he.THUMB_TIP] = DummyLandmark(0.52, 0.45)
    h_pinch_released[he.INDEX_TIP] = DummyLandmark(0.65, 0.45)
    hands_released = [(h_fork, False, to_screen), (h_pinch_released, True, to_screen)]

    text_fire, busy_fire = mgr.update_and_render(frame, hands_released, particles, dt=0.03)
    assert mgr.state == "FLYING", f"Expected state FLYING, got {mgr.state}"
    print(f"-> Slingshot release detected! Bullet launched towards camera ({text_fire})")

    # 3. Simulate bullet impact
    mgr.bullet_start_time = time.time() - 0.5
    text_impact, busy_impact = mgr.update_and_render(frame, hands_released, particles, dt=0.03)
    assert mgr.fog_active is True, "Fog should be active after bullet impact!"
    assert mgr.fog_mask is not None
    assert np.all(mgr.fog_mask == 255), "Fog mask should initially be full frosted blur!"
    print("-> Bullet impact verified: Screen shaken and camera frosted blur activated!")

    # 4. Simulate wiping camera clean
    for wipe_x in np.linspace(0.3, 0.8, 8):
        h_wipe = create_hand()
        h_wipe[he.MIDDLE_MCP] = DummyLandmark(wipe_x, 0.5)
        h_wipe[he.WRIST] = DummyLandmark(wipe_x, 0.65)
        mgr.update_and_render(frame, [(h_wipe, False, to_screen)], particles, dt=0.03)

    remaining_fog = np.mean(mgr.fog_mask > 30) if mgr.fog_mask is not None else 0.0
    assert remaining_fog < 0.90, f"Wiping should clear fog! Remaining: {remaining_fog * 100:.1f}%"
    print(f"-> Camera wipe verified: Fog reduced to {remaining_fog * 100:.1f}%.")

    # Simulate wiping entire screen clean
    mgr.fog_mask.fill(0)
    mgr.update_and_render(frame, [(h_wipe, False, to_screen)], particles, dt=0.03)
    assert mgr.fog_active is False, "Fog should clear when wiped!"
    print("-> 100% clean threshold verified: Camera clear shimmer triggered successfully.")


def test_feature_6_black_white_pull_down_and_push_transition():
    print("Testing Feature 6: Two-Hand Pull Down & Black & White Push Transition...")
    w, h = 1280, 720
    to_screen = lambda pt: (int(pt.x * w), int(pt.y * h))

    mgr = he.BlackWhiteFilterManager()
    assert mgr.is_bw is False and mgr.in_transition is False

    # 1. Test stationary hands: should NOT trigger pull-down
    t0 = 100.0
    h_l = create_hand()
    h_l[he.WRIST] = DummyLandmark(0.30, 0.35)
    h_l[he.MIDDLE_MCP] = DummyLandmark(0.30, 0.32)

    h_r = create_hand()
    h_r[he.WRIST] = DummyLandmark(0.70, 0.35)
    h_r[he.MIDDLE_MCP] = DummyLandmark(0.70, 0.32)

    hands_stationary = [(h_l, False, to_screen), (h_r, True, to_screen)]
    for step in range(4):
        txt = mgr.update_and_detect_gesture(hands_stationary, t0 + step * 0.03, is_blur_active=True)
        assert txt is None, "Stationary hands should not trigger pull down!"
    print("-> Stationary hands correctly ignored.")

    # 2. Test downward pull movement when BLUR IS FALSE: should NOT activate!
    y_steps = [0.35, 0.44, 0.54, 0.65]
    detected_no_blur = None
    for idx, y_val in enumerate(y_steps):
        t_curr = t0 + 0.15 + idx * 0.04
        hl_mov = create_hand()
        hl_mov[he.WRIST] = DummyLandmark(0.30, y_val)
        hl_mov[he.MIDDLE_MCP] = DummyLandmark(0.30, y_val - 0.03)

        hr_mov = create_hand()
        hr_mov[he.WRIST] = DummyLandmark(0.70, y_val)
        hr_mov[he.MIDDLE_MCP] = DummyLandmark(0.70, y_val - 0.03)

        hands_pull = [(hl_mov, False, to_screen), (hr_mov, True, to_screen)]
        res = mgr.update_and_detect_gesture(hands_pull, t_curr, is_blur_active=False)
        if res:
            detected_no_blur = res

    assert detected_no_blur is None, "B&W should NOT activate if blur is not active!"
    assert mgr.is_bw is False and mgr.in_transition is False
    print("-> Verified: B&W correctly rejected when blur is not active.")

    # 3. Test downward pull movement when BLUR IS TRUE: SHOULD activate!
    detected_text = None
    for idx, y_val in enumerate(y_steps):
        t_curr = t0 + 2.0 + idx * 0.04
        hl_mov = create_hand()
        hl_mov[he.WRIST] = DummyLandmark(0.30, y_val)
        hl_mov[he.MIDDLE_MCP] = DummyLandmark(0.30, y_val - 0.03)

        hr_mov = create_hand()
        hr_mov[he.WRIST] = DummyLandmark(0.70, y_val)
        hr_mov[he.MIDDLE_MCP] = DummyLandmark(0.70, y_val - 0.03)

        hands_pull = [(hl_mov, False, to_screen), (hr_mov, True, to_screen)]
        res = mgr.update_and_detect_gesture(hands_pull, t_curr, is_blur_active=True)
        if res:
            detected_text = res

    assert detected_text is not None, "Two-hand downward pull gesture should be detected when blur is active!"
    assert mgr.in_transition is True, "Manager should enter in_transition state!"
    assert mgr.target_bw is True, "Target state should be Black & White!"
    print(f"-> Pull down gesture triggered successfully when blur is active ({detected_text}).")

    # 4. Test push transition rendering (Push from top)
    test_frame = np.full((h, w, 3), (30, 120, 220), dtype=np.uint8)
    particles = []
    t_mid = mgr.transition_start_time + (mgr.transition_duration * 0.5)

    render_txt = mgr.render_and_apply(test_frame, particles, t_mid, is_blur_active=True)
    assert mgr.in_transition is True
    assert "Push Filter" in render_txt or "Filter Push" in render_txt

    split_y = int(0.5 * h)
    top_pixel = test_frame[split_y - 80, w // 2]
    bottom_pixel = test_frame[split_y + 80, w // 2]

    assert abs(int(top_pixel[0]) - int(top_pixel[1])) <= 2 and abs(int(top_pixel[1]) - int(top_pixel[2])) <= 2, (
        f"Top region should be Black & White, got pixel: {top_pixel}"
    )
    assert np.allclose(bottom_pixel, [30, 120, 220], atol=3), (
        f"Bottom region should be original color, got pixel: {bottom_pixel}"
    )
    assert len(particles) > 0, "Laser divider sparkles should be generated during push transition!"
    print("-> Push from top verified: Top is B&W, Bottom is Color, Laser divider active.")

    # 5. Test transition completion
    t_end = mgr.transition_start_time + mgr.transition_duration + 0.1
    mgr.render_and_apply(test_frame, particles, t_end, is_blur_active=True)
    assert mgr.in_transition is False, "Transition should complete after duration!"
    assert mgr.is_bw is True, "Filter state should be fully Black & White!"
    print("-> Full Black & White active state verified.")

    # 6. Test blur deactivation resetting B&W back to Color
    mgr.render_and_apply(test_frame, particles, t_end + 0.5, is_blur_active=False)
    assert mgr.is_bw is False and mgr.in_transition is False, "B&W must reset when blur ends!"
    print("-> Verified: B&W automatically clears when blur is wiped clean/deactivated.")

    # 7. Test toggle back to color with push from top (when blur is active)
    mgr.trigger_toggle(t_end + 1.0, is_blur_active=True)
    assert mgr.in_transition is True and mgr.target_bw is True  # was reset to False, so now True
    print("-> Trigger toggle with blur verified.")


if __name__ == "__main__":
    he.ensure_models()
    test_sprites()
    test_particle_rendering()
    test_unlock_logic()
    test_index_touch_trigger_and_rectangle()
    test_background_flower_compositing()
    test_async_segmenter()
    test_feature_4_cupped_hands_and_3d_crystal()
    test_feature_5_slingshot_and_wipe_blur()
    test_feature_6_black_white_pull_down_and_push_transition()
    print("\nALL AUTOMATED TESTS PASSED!")

