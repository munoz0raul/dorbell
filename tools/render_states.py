"""Headless render of every Dragonwing Doorbell screen to PNGs for visual QA.

Imports doorbell.py's drawing layer directly (stubbing cv2 display calls) and
writes one PNG per state into tools/_shots/. No camera or models required.
"""
import os, sys, importlib.util
import numpy as np
import cv2

# Stub display/window calls so importing doorbell.py never opens a window.
cv2.namedWindow = lambda *a, **k: None
cv2.setWindowProperty = lambda *a, **k: None
cv2.imshow = lambda *a, **k: None

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "doorbell.py")
spec = importlib.util.spec_from_file_location("doorbell", APP)
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)

OUT = os.path.join(HERE, "_shots")
os.makedirs(OUT, exist_ok=True)
W, H = db.W, db.H


def fake_frame():
    """A synthetic 'camera' frame with a face-like blob so overlays have context."""
    f = np.full((480, 640, 3), 40, np.uint8)
    f[:] = db.INDIGO_DEEP
    cv2.ellipse(f, (320, 240), (90, 120), 0, 0, 360, (150, 170, 190), -1)
    cv2.circle(f, (290, 215), 12, (60, 60, 70), -1)
    cv2.circle(f, (350, 215), 12, (60, 60, 70), -1)
    return f


class FakeFace:
    def __init__(self, cx=160, cy=120, w=100, h=120, score=0.92):
        self.bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)
        self.kps = np.array([[cx - 25, cy - 20], [cx + 25, cy - 20], [cx, cy],
                             [cx - 20, cy + 25], [cx + 20, cy + 25]], np.float32)
        self.landmark_2d_106 = None
        self.det_score = score
        self.embedding = np.ones(512, np.float32)


def shot(name, canvas):
    cv2.imwrite(os.path.join(OUT, name), canvas)
    print(" ", name)


def main():
    canvas = np.zeros((H, W, 3), np.uint8)
    t = 1.3
    faces = [FakeFace()]
    frame = fake_frame()

    db.draw_loading(canvas, t); shot("01_loading.png", canvas)
    db.draw_standby(canvas, t, 2); shot("02_standby.png", canvas)

    db.draw_camera_state(canvas, frame, faces, db.INF_W, db.INF_H,
                         db.DETECT, "RECOGNIZING", db.DETECT)
    db.draw_hud(canvas, db.State.DETECTION, "Face detected", 2, 15.3, "npu")
    shot("03_detection.png", canvas)

    alice = FakeFace(); alice.rec_name = "Alice"; alice.rec_sim = 0.71
    db.draw_camera_state(canvas, frame, [alice], db.INF_W, db.INF_H,
                         db.UNLOCK, "UNLOCKED   Alice", db.UNLOCK, per_face=True)
    db.draw_hud(canvas, db.State.RECOGNIZED, "Welcome, Alice!", 2, 15.3, "npu")
    shot("04_recognized.png", canvas)

    db.draw_camera_state(canvas, frame, faces, db.INF_W, db.INF_H,
                         db.LOCK, "ACCESS DENIED", db.LOCK, per_face=True)
    db.draw_hud(canvas, db.State.UNKNOWN, "Unknown person", 2, 12.1, "cpu")
    shot("05_unknown.png", canvas)

    # Mixed crowd: one known (green + name) + one unknown (red) in the same frame.
    known = FakeFace(cx=120, cy=120, w=90, h=110); known.rec_name = "raul"; known.rec_sim = 0.62
    stranger = FakeFace(cx=240, cy=130, w=80, h=100); stranger.rec_name = None; stranger.rec_sim = 0.21
    wide = fake_frame()
    db.draw_camera_state(canvas, wide, [known, stranger], db.INF_W, db.INF_H,
                         db.UNLOCK, "UNLOCKED   raul", db.UNLOCK, per_face=True)
    db.draw_hud(canvas, db.State.RECOGNIZED, "Welcome, raul!", 1, 15.3, "npu")
    shot("13_mixed_recognition.png", canvas)

    db.draw_camera_state(canvas, frame, faces, db.INF_W, db.INF_H,
                         db.VIOLET, "ENROLLING   john-doe", db.VIOLET, reg_progress=(14, 30))
    db.draw_hud(canvas, db.State.REGISTRATION, "Enrolling john-doe", 2, 15.0, "npu")
    shot("06_registration.png", canvas)

    db.draw_qr_authorized(canvas, 1.0, "john-doe"); shot("07_qr_authorized.png", canvas)
    db.draw_qr_countdown(canvas, 2.0, "john-doe", 3); shot("08_qr_countdown.png", canvas)

    # Enroll gate — the "O" the visitor places their face in. One face centered in
    # the ring (ready), during the get-ready countdown and during capture.
    gcx, gcy, _, _ = db.gate_ellipse_px(db.INF_W, db.INF_H)
    in_gate = [FakeFace(cx=gcx, cy=gcy)]
    db.draw_enroll_gate(canvas, frame, in_gate, db.INF_W, db.INF_H, "john-doe",
                        phase="countdown", countdown=3, t=t)
    db.draw_hud(canvas, db.State.QR_AUTHORIZED, "Get ready", 2, 15.0, "npu")
    shot("10_enroll_gate_countdown.png", canvas)

    db.draw_enroll_gate(canvas, frame, in_gate, db.INF_W, db.INF_H, "john-doe",
                        phase="capture", reg_progress=(14, 30), t=t)
    db.draw_hud(canvas, db.State.REGISTRATION, "Enrolling john-doe", 2, 15.0, "npu")
    shot("11_enroll_gate_capture.png", canvas)

    # Two faces in the ring → capture paused, "one person at a time".
    two_in = [FakeFace(cx=gcx - 30, cy=gcy), FakeFace(cx=gcx + 30, cy=gcy)]
    db.draw_enroll_gate(canvas, frame, two_in, db.INF_W, db.INF_H, "john-doe",
                        phase="capture", reg_progress=(14, 30), blocked=True, t=t)
    db.draw_hud(canvas, db.State.REGISTRATION, "Enrolling john-doe", 2, 15.0, "npu")
    shot("12_enroll_gate_blocked.png", canvas)

    # Standby + full HUD + stats panel = the real composite the demo shows
    db.draw_standby(canvas, t, 2)
    db.draw_hud(canvas, db.State.STANDBY, "SYSTEM READY", 2, 0.0, "cpu")
    db.draw_stats_panel(canvas, [12, 45, 8, 90, 30, 15, 60, 5], 33.1,
                        41.0, 2400, 5800, 15.3, 15.0, "npu", True)
    shot("09_standby_full.png", canvas)

    print(f"\nWrote {len(os.listdir(OUT))} shots to {OUT}")


if __name__ == "__main__":
    main()
