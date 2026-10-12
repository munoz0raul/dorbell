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
    def __init__(self):
        self.bbox = np.array([110, 70, 210, 190], np.float32)
        self.kps = np.array([[135, 110], [185, 110], [160, 140], [140, 165], [180, 165]], np.float32)
        self.landmark_2d_106 = None
        self.det_score = 0.92


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

    db.draw_camera_state(canvas, frame, faces, db.INF_W, db.INF_H,
                         db.UNLOCK, "UNLOCKED   Alice", db.UNLOCK)
    db.draw_access_banner(canvas, "UNLOCKED  ALICE", db.UNLOCK, 1.3)
    db.draw_hud(canvas, db.State.RECOGNIZED, "Welcome, Alice!", 2, 15.3, "npu")
    shot("04_recognized.png", canvas)

    db.draw_camera_state(canvas, frame, faces, db.INF_W, db.INF_H,
                         db.LOCK, "ACCESS DENIED", db.LOCK)
    db.draw_access_banner(canvas, "ACCESS DENIED", db.LOCK, 1.3)
    db.draw_hud(canvas, db.State.UNKNOWN, "Unknown person", 2, 12.1, "cpu")
    shot("05_unknown.png", canvas)

    db.draw_camera_state(canvas, frame, faces, db.INF_W, db.INF_H,
                         db.VIOLET, "ENROLLING   john-doe", db.VIOLET, reg_progress=(6, 10))
    db.draw_hud(canvas, db.State.REGISTRATION, "Enrolling john-doe", 2, 15.0, "npu")
    shot("06_registration.png", canvas)

    db.draw_qr_authorized(canvas, 1.0, "john-doe"); shot("07_qr_authorized.png", canvas)
    db.draw_qr_countdown(canvas, 2.0, "john-doe", 3); shot("08_qr_countdown.png", canvas)

    # Standby + full HUD + stats panel = the real composite the demo shows
    db.draw_standby(canvas, t, 2)
    db.draw_hud(canvas, db.State.STANDBY, "SYSTEM READY", 2, 0.0, "cpu")
    db.draw_stats_panel(canvas, [12, 45, 8, 90, 30, 15, 60, 5], 33.1,
                        41.0, 2400, 5800, 15.3, 15.0, "npu", True)
    shot("09_standby_full.png", canvas)

    print(f"\nWrote {len(os.listdir(OUT))} shots to {OUT}")


if __name__ == "__main__":
    main()
