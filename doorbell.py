"""
Dragonwing Doorbell — doorbell.py
Autonomous face-recognition doorbell demo for the Qualcomm IQ-8275 (QCS8275 / HTP v75).

Stage 2 target: IQ8 + Ubuntu 24.04 + USB webcam + HDMI display.
  - InsightFace buffalo_sc, det_size=(320,320) for low latency on aarch64 CPU
  - V4L2 camera capture, synthetic-frame fallback when no camera is present
  - Dual inference backends, switchable live:
      CPU  — InsightFace / onnxruntime (CPUExecutionProvider)
      NPU  — HTP v75 via qnn-net-run batch engine (the only working NPU path on this board)
  - Auto-discovers Xwayland auth from the running GNOME session
  - Dragonwing-branded full-screen UI drawn on an OpenCV canvas

Usage:
  python3 doorbell.py                  # CPU backend
  python3 doorbell.py --backend npu    # NPU (HTP v75) via qnn-net-run batch engine

Controls (click the window first):
  B   — toggle CPU / NPU backend live
  K   — force "known face" result   (demo)
  U   — force "unknown face" result (demo)
  R   — start registration manually (fallback for the QR trigger)
  ESC — back to Standby
  Q   — quit

Auto behaviour:
  Face detected        → Standby becomes Detection, recognition runs each frame
  No face for 5s       → returns to Standby
  LinkedIn QR detected → triggers Registration (captures 10 frames, 2-min TTL)
"""

import cv2
import numpy as np
import time
import threading
import random
import re
import os
import glob
import argparse
import collections
import subprocess
import tempfile
import shutil
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List


def _setup_display():
    """Auto-discover Xwayland auth from the running GNOME session."""
    if not os.environ.get('DISPLAY'):
        os.environ['DISPLAY'] = ':0'
    if not os.environ.get('XAUTHORITY'):
        matches = glob.glob('/run/user/1000/.mutter-Xwaylandauth.*')
        if matches:
            os.environ['XAUTHORITY'] = matches[0]
            print(f"[Display] XAUTHORITY={matches[0]}")
        else:
            print("[Display] WARNING: no Xwayland auth found")

_setup_display()


# ── FPS counter ───────────────────────────────────────────────────────────────

class FpsCounter:
    """Rolling average FPS over the last N timestamps."""
    def __init__(self, window: int = 30):
        self._times: collections.deque = collections.deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


# ── NPU inference engine (qnn-net-run batch) ──────────────────────────────────
#
# Pre/post-processing helpers shared by CPU and NPU paths.

ONNX_DIR = os.path.expanduser("~/.insightface/models/buffalo_sc")
CTX_DIR  = os.path.expanduser("~/qnn_context")

_DET_STRIDES    = [8, 16, 32]
_DET_FEAT_MAPS  = [40, 20, 10]   # 320/8, 320/16, 320/32
_DET_NUM_ANCHORS = 2


def _det_preprocess(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (320, 320))
    img = img[:, :, ::-1].astype(np.float32)
    img = (img - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]


def _det_postprocess(outputs, orig_h, orig_w, det_thresh=0.5, nms_thresh=0.4):
    scores_list = outputs[0:3]
    boxes_list  = outputs[3:6]
    kps_list    = outputs[6:9]

    sx = orig_w / 320.0
    sy = orig_h / 320.0
    boxes_out, scores_out, kps_out = [], [], []

    for i, stride in enumerate(_DET_STRIDES):
        fm = _DET_FEAT_MAPS[i]
        scores = scores_list[i].reshape(-1)
        boxes  = boxes_list[i].reshape(-1, 4)
        kps    = kps_list[i].reshape(-1, 5, 2)

        ys, xs = np.meshgrid(np.arange(fm), np.arange(fm), indexing='ij')
        cx = np.repeat((xs.flatten() + 0.5) * stride, _DET_NUM_ANCHORS)
        cy = np.repeat((ys.flatten() + 0.5) * stride, _DET_NUM_ANCHORS)

        mask = scores >= det_thresh
        if not np.any(mask):
            continue

        sc  = scores[mask]
        bx  = boxes[mask]
        kp  = kps[mask]
        acx = cx[mask]
        acy = cy[mask]

        x1 = (acx - bx[:, 0] * stride) * sx
        y1 = (acy - bx[:, 1] * stride) * sy
        x2 = (acx + bx[:, 2] * stride) * sx
        y2 = (acy + bx[:, 3] * stride) * sy

        kp_dec = np.zeros_like(kp)
        kp_dec[:, :, 0] = (acx[:, None] + kp[:, :, 0] * stride) * sx
        kp_dec[:, :, 1] = (acy[:, None] + kp[:, :, 1] * stride) * sy

        boxes_out.append(np.stack([x1, y1, x2, y2], axis=1))
        scores_out.append(sc)
        kps_out.append(kp_dec)

    if not boxes_out:
        return []

    boxes_all  = np.concatenate(boxes_out)
    scores_all = np.concatenate(scores_out)
    kps_all    = np.concatenate(kps_out)

    keep = cv2.dnn.NMSBoxes(
        boxes_all.tolist(), scores_all.tolist(), det_thresh, nms_thresh)
    if len(keep) == 0:
        return []

    faces = []
    for k in keep.flatten():
        faces.append({'bbox': boxes_all[k], 'kps': kps_all[k], 'score': scores_all[k]})
    return faces


def _rec_preprocess(bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr, (112, 112))
    img = img[:, :, ::-1].astype(np.float32)
    img = (img - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]



QNN_BACKEND  = "/usr/lib/libQnnHtp.so"
DET_CTX_PATH = os.path.join(CTX_DIR, "det_500m_qcs8275.bin")
REC_CTX_PATH = os.path.join(CTX_DIR, "w600k_mbf_qcs8275.bin")

# Det output tensor shapes (float32 element counts):
#   outputs 0-2: scores  (3200, 800, 200)
#   outputs 3-5: boxes×4 (12800, 3200, 800)
#   outputs 6-8: kps×10  (32000, 8000, 2000)
_DET_OUT_SHAPES = [
    (3200, 1), (800, 1), (200, 1),
    (3200, 4), (800, 4), (200, 4),
    (3200, 10), (800, 10), (200, 10),
]
_REC_EMB_SIZE = 512   # w600k_mbf output: 1×512


def _qnn_run_batch(ctx_bin: str, inp_raws: list, out_dir: str, tmpdir: str) -> bool:
    """
    Write input_list.txt for inp_raws, run qnn-net-run, return True on success.
    Outputs land in out_dir/Result_0/ .. Result_{N-1}/.
    """
    model_tag = "det" if "det" in ctx_bin else "rec"
    list_path = os.path.join(tmpdir, f"list_{model_tag}.txt")
    with open(list_path, "w") as f:
        for p in inp_raws:
            f.write(p + "\n")
    cmd = [
        "qnn-net-run",
        f"--backend={QNN_BACKEND}",
        f"--retrieve_context={ctx_bin}",
        f"--input_list={list_path}",
        f"--output_dir={out_dir}",
        "--perf_profile=burst",
        "--log_level=error",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception as e:
        print(f"[QNN] qnn-net-run failed: {e}")
        return False


def _read_det_outputs(result_dir: str):
    """Read 9 output tensors from Result_N/ and return list of np arrays."""
    outs = []
    for i, shape in enumerate(_DET_OUT_SHAPES):
        path = os.path.join(result_dir, f"output_{i}.raw")
        arr = np.fromfile(path, dtype=np.float32).reshape(shape)
        outs.append(arr)
    return outs


def _read_rec_output(result_dir: str) -> np.ndarray:
    path = os.path.join(result_dir, "output_0.raw")
    return np.fromfile(path, dtype=np.float32)   # shape (512,)


class NpuFaceEngine(threading.Thread):
    """
    Real HTP (Hexagon Tensor Processor) NPU engine using qnn-net-run CLI.

    Design: queue-based batch inference.
    - Main thread calls submit(frame) — frames accumulate in a deque (up to MAX_BATCH).
    - Background thread waits until MAX_BATCH frames queued OR BATCH_TIMEOUT seconds
      elapses, then calls qnn-net-run once for the whole batch.

    Timing breakdown (measured on QCS8275):
      Fixed spawn overhead:  ~270ms  (FastRPC + model load, per qnn-net-run call)
      Per-inference HTP cost: ~11ms det / ~4ms rec
      With batch=16: 270ms + 16×11ms = 446ms det → 28ms/frame → ~36 FPS det
      Full pipeline (det+rec): ~(270+16×11) + (270+1×4) ≈ 716ms/16 frames → ~22 FPS

    CPU usage when active: ~5-10% (process blocked in subprocess.run kernel wait)
    vs CPU engine: ~95% (ORT spreads work across all 8 cores).
    """
    MAX_BATCH     = 16
    BATCH_TIMEOUT = 0.35    # seconds — collect frames for up to this long, then run

    def __init__(self, on_ready):
        super().__init__(daemon=True)
        self._queue: collections.deque = collections.deque(maxlen=self.MAX_BATCH)
        self._queue_lock = threading.Lock()
        self._faces: list = []
        self._faces_lock = threading.Lock()
        self.running = True
        self.on_ready = on_ready
        self.fps_counter = FpsCounter(window=30)
        self._tmpdir: Optional[str] = None

    def submit(self, frame: np.ndarray):
        with self._queue_lock:
            self._queue.append(frame.copy())

    def get_faces(self) -> list:
        with self._faces_lock:
            return list(self._faces)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _infer_det_batch(self, frames: list) -> list:
        """Run det model on a list of frames, return list of raw_face lists."""
        det_work = os.path.join(self._tmpdir, "det_work")
        if os.path.exists(det_work):
            shutil.rmtree(det_work)
        os.makedirs(det_work)

        inp_paths = []
        for i, f in enumerate(frames):
            pre = _det_preprocess(f)
            p = os.path.join(self._tmpdir, f"det_in_{i}.raw")
            pre.tofile(p)
            inp_paths.append(p)

        ok = _qnn_run_batch(DET_CTX_PATH, inp_paths, det_work, self._tmpdir)
        if not ok:
            return [[] for _ in frames]

        results = []
        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            result_dir = os.path.join(det_work, f"Result_{i}")
            try:
                outs = _read_det_outputs(result_dir)
                results.append(_det_postprocess(outs, h, w))
            except Exception as e:
                print(f"[QNN] det read error for frame {i}: {e}")
                results.append([])
        return results

    def _infer_rec_batch(self, crops: list) -> list:
        """Run rec model on a list of 112×112 crops, return list of embedding arrays."""
        if not crops:
            return []
        rec_work = os.path.join(self._tmpdir, "rec_work")
        if os.path.exists(rec_work):
            shutil.rmtree(rec_work)
        os.makedirs(rec_work)

        inp_paths = []
        for i, crop in enumerate(crops):
            pre = _rec_preprocess(crop)
            p = os.path.join(self._tmpdir, f"rec_in_{i}.raw")
            pre.tofile(p)
            inp_paths.append(p)

        ok = _qnn_run_batch(REC_CTX_PATH, inp_paths, rec_work, self._tmpdir)
        if not ok:
            return [None] * len(crops)

        embeddings = []
        for i in range(len(crops)):
            result_dir = os.path.join(rec_work, f"Result_{i}")
            try:
                emb = _read_rec_output(result_dir)
                norm = np.linalg.norm(emb)
                embeddings.append(emb / norm if norm > 1e-6 else emb)
            except Exception as e:
                print(f"[QNN] rec read error for crop {i}: {e}")
                embeddings.append(None)
        return embeddings

    def _process_batch(self, frames: list):
        """Full pipeline for a batch of frames; updates self._faces with latest."""
        if not frames:
            return

        t_batch_start = time.perf_counter()

        # Detection pass — one qnn-net-run call for the whole batch
        raw_faces_list = self._infer_det_batch(frames)

        # Gather rec crops from the *latest* frame (last in batch)
        latest_idx = len(frames) - 1
        latest_frame = frames[latest_idx]
        h, w = latest_frame.shape[:2]
        raw_faces = raw_faces_list[latest_idx]

        crops = []
        valid_fds = []
        for fd in raw_faces:
            x1, y1, x2, y2 = fd['bbox']
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                crops.append(latest_frame[y1:y2, x1:x2])
                valid_fds.append(fd)

        # Recognition pass — batch all crops in one qnn-net-run call
        embeddings = self._infer_rec_batch(crops)

        result_faces = []
        for fd, emb in zip(valid_fds, embeddings):
            if emb is None:
                continue
            face = type('Face', (), {
                'bbox': fd['bbox'], 'kps': fd['kps'],
                'landmark_2d_106': None,
                'det_score': fd['score'], 'embedding': emb,
            })()
            result_faces.append(face)

        with self._faces_lock:
            self._faces = result_faces

        # Spread FPS ticks evenly across the real batch wall-clock duration
        # so the rolling FpsCounter shows actual throughput, not a burst spike.
        t_now = time.perf_counter()
        batch_dur = max(t_now - t_batch_start, 1e-6)
        n = len(frames)
        for i in range(n):
            self.fps_counter._times.append(t_batch_start + batch_dur * i / n)

    # ── thread entry ─────────────────────────────────────────────────────────

    def run(self):
        det_ok = os.path.exists(DET_CTX_PATH)
        rec_ok = os.path.exists(REC_CTX_PATH)
        if not det_ok or not rec_ok:
            print(f"[NPU] ERROR: context binaries missing")
            print(f"  det: {DET_CTX_PATH}  exists={det_ok}")
            print(f"  rec: {REC_CTX_PATH}  exists={rec_ok}")
            self.on_ready()
            return

        self._tmpdir = tempfile.mkdtemp(prefix="qnn_doorbell_")
        print(f"[NPU] qnn-net-run engine ready  tmpdir={self._tmpdir}")
        print(f"[NPU] det_ctx={DET_CTX_PATH}")
        print(f"[NPU] rec_ctx={REC_CTX_PATH}")
        self.on_ready()

        batch_start = time.perf_counter()

        while self.running:
            with self._queue_lock:
                queue_len = len(self._queue)

            elapsed = time.perf_counter() - batch_start
            batch_full  = queue_len >= self.MAX_BATCH
            timeout_hit = elapsed >= self.BATCH_TIMEOUT and queue_len > 0
            idle        = elapsed >= self.BATCH_TIMEOUT and queue_len == 0

            if batch_full or timeout_hit:
                with self._queue_lock:
                    batch = list(self._queue)
                    self._queue.clear()
                self._process_batch(batch)
                batch_start = time.perf_counter()
            elif idle:
                batch_start = time.perf_counter()
                time.sleep(0.01)
            else:
                time.sleep(0.010)

        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def stop(self):
        self.running = False


# QR codes whose URLs match this pattern trigger registration.
# Name is extracted from the LinkedIn username segment.
QR_URL_PATTERN = re.compile(r'linkedin\.com/in/([^/?]+)', re.IGNORECASE)


# ── State machine ─────────────────────────────────────────────────────────────

class State(Enum):
    LOADING       = auto()
    STANDBY       = auto()
    DETECTION     = auto()
    RECOGNIZED    = auto()
    UNKNOWN       = auto()
    QR_AUTHORIZED = auto()  # 5s "AUTHORIZED!" anim + 5s countdown
    REGISTRATION  = auto()


@dataclass
class FaceEntry:
    name: str
    added_at: float
    ttl: float = 120.0
    embeddings: List[np.ndarray] = field(default_factory=list)

    def is_expired(self) -> bool:
        return time.time() - self.added_at > self.ttl

    def remaining(self) -> float:
        return max(0.0, self.ttl - (time.time() - self.added_at))

    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)


@dataclass
class AppState:
    state: State = State.LOADING
    face_db: dict = field(default_factory=dict)
    reg_frames: int = 0
    reg_target: int = 10
    reg_name: str = ""
    last_match: Optional[str] = None
    status_msg: str = "Loading models..."
    last_face_time: float = 0.0
    qr_authorized_at: float = 0.0   # when QR_AUTHORIZED state began
    reg_start_at: float = 0.0       # when REGISTRATION capture began
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_face(self, name: str, embeddings: list):
        entry = FaceEntry(name=name, added_at=time.time())
        entry.embeddings = [e.copy() for e in embeddings]
        self.face_db[name] = entry

    def prune_expired(self):
        expired = [k for k, v in self.face_db.items() if v.is_expired()]
        for k in expired:
            del self.face_db[k]
            print(f"[DB] Expired: {k}")


# ── QR Detector ───────────────────────────────────────────────────────────────

class QRDetector:
    def __init__(self):
        self._detector = cv2.QRCodeDetector()
        self._cooldown_until = 0.0  # avoid re-triggering immediately

    def detect(self, frame: np.ndarray) -> Optional[str]:
        """Return a registration name if a known QR is found, else None."""
        if time.time() < self._cooldown_until:
            return None
        data, pts, _ = self._detector.detectAndDecode(frame)
        if not data:
            return None
        m = QR_URL_PATTERN.search(data)
        if m:
            self._cooldown_until = time.time() + 5.0  # 5s cooldown
            username = m.group(1).split('?')[0]
            return username
        return None


# ── Recognition ───────────────────────────────────────────────────────────────

RECOGNITION_THRESHOLD = 0.35
NO_FACE_TIMEOUT = 5.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def recognize_face(embedding: np.ndarray, face_db: dict) -> Optional[str]:
    best_name, best_sim = None, -1.0
    for name, entry in face_db.items():
        mean_emb = entry.mean_embedding()
        if mean_emb is None:
            continue
        sim = cosine_sim(embedding, mean_emb)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name if best_sim >= RECOGNITION_THRESHOLD else None


# ── Face engine (background thread) ───────────────────────────────────────────

class FaceEngine(threading.Thread):
    def __init__(self, on_ready):
        super().__init__(daemon=True)
        self._pending: Optional[np.ndarray] = None
        self._pending_lock = threading.Lock()
        self._faces: list = []
        self._faces_lock = threading.Lock()
        self.running = True
        self.on_ready = on_ready
        self.fa = None
        self.fps_counter = FpsCounter(window=30)

    def submit(self, frame: np.ndarray):
        with self._pending_lock:
            self._pending = frame  # always keep latest, drop older

    def get_faces(self) -> list:
        with self._faces_lock:
            return list(self._faces)

    def run(self):
        from insightface.app import FaceAnalysis
        self.fa = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
        self.fa.prepare(ctx_id=0, det_size=(320, 320))
        self.on_ready()
        while self.running:
            with self._pending_lock:
                frame = self._pending
                self._pending = None
            if frame is not None:
                faces = self.fa.get(frame)
                with self._faces_lock:
                    self._faces = faces
                self.fps_counter.tick()
            else:
                time.sleep(0.01)

    def stop(self):
        self.running = False


# ── Camera thread ─────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.running = True
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print("[Camera] No camera — using synthetic frames")

    def run(self):
        t = 0.0
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
            else:
                h, w = 480, 640
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                t += 0.02
                for y in range(h):
                    b = int(20 + 60 * abs(np.sin(t + y * 0.005)))
                    frame[y, :, 0] = b
                noise = np.random.randint(0, 12, (h, w, 3), dtype=np.uint8)
                frame = cv2.add(frame, noise)
                with self._lock:
                    self._frame = frame
            time.sleep(1 / 30)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


# ── Registration driver ───────────────────────────────────────────────────────

class RegistrationDriver(threading.Thread):
    def __init__(self, app_state: AppState, face_engine: FaceEngine, on_done):
        super().__init__(daemon=True)
        self.app_state = app_state
        self.face_engine = face_engine
        self.on_done = on_done
        self._embeddings: List[np.ndarray] = []

    def run(self):
        while True:
            with self.app_state.lock:
                if self.app_state.state != State.REGISTRATION:
                    return
                n = self.app_state.reg_frames
                target = self.app_state.reg_target
                name = self.app_state.reg_name

            faces = self.face_engine.get_faces()
            captured = False
            for face in faces:
                if face.embedding is not None:
                    self._embeddings.append(face.embedding.copy())
                    captured = True
                    break

            if captured:
                with self.app_state.lock:
                    self.app_state.reg_frames += 1
                    done = self.app_state.reg_frames >= self.app_state.reg_target
                    name = self.app_state.reg_name
                if done:
                    self.on_done(name, self._embeddings)
                    return

            time.sleep(0.3)


# ── Drawing ───────────────────────────────────────────────────────────────────

W, H = 1280, 720

# ── Dragonwing theme ────────────────────────────────────────────────────────
# Palette sampled from the Dragonwing brand references (assets/1.png–3.png).
# Ground is deep indigo #31017D; the accent is electric violet #5B0CD9.
# OpenCV is BGR, so every tuple below is (B, G, R).

INDIGO      = (125,   1,  49)   # #31017D  brand ground
INDIGO_DEEP = ( 90,   1,  35)   # darker indigo for gradient base / panels
INDIGO_LIFT = (150,  10,  66)   # lifted indigo for gradient top / fills
VIOLET      = (217,  12,  91)   # #5B0CD9  primary accent / brand chrome
VIOLET_LO   = (150,  30,  70)   # dim violet — inactive lines, secondary rules
VIOLET_HI   = (245, 120, 170)   # bright violet highlight / glow
INK         = (245, 240, 255)   # near-white primary text
MIST        = (190, 155, 205)   # secondary text (tinted from the violet hue, never gray)

# Access-control semantics stay universal, retuned to read on indigo.
UNLOCK      = (120, 220,  90)   # known / unlocked — green
LOCK        = ( 70,  70, 235)   # unknown / locked — red
DETECT      = (235, 190,  60)   # recognizing — cyan-amber that sits on indigo

# Legacy aliases kept so any out-of-scope reference still resolves.
GREEN, RED, AMBER, PURPLE, WHITE, DIM, DARK = (
    UNLOCK, LOCK, DETECT, VIOLET, INK, VIOLET_LO, INDIGO_DEEP)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

# Precomputed vertical indigo gradient (INDIGO_DEEP at top → INDIGO_LIFT at bottom).
_GRADIENT = None


def _build_gradient() -> np.ndarray:
    col = np.zeros((H, 1, 3), dtype=np.uint8)
    for y in range(H):
        f = y / (H - 1)
        col[y, 0] = [int(INDIGO_DEEP[c] + (INDIGO_LIFT[c] - INDIGO_DEEP[c]) * f)
                     for c in range(3)]
    return np.repeat(col, W, axis=1)


def brand_gradient(canvas: np.ndarray):
    """Fill the canvas with the vertical indigo brand gradient."""
    global _GRADIENT
    if _GRADIENT is None or _GRADIENT.shape[:2] != canvas.shape[:2]:
        _GRADIENT = _build_gradient()
    canvas[:] = _GRADIENT


def draw_wave_field(canvas: np.ndarray, t: float, intensity: float = 1.0):
    """
    The swept contour-line motif from assets/1.png: a fan of fine curves
    sweeping across the field. Rendered in dim violet over the indigo ground;
    animates by phase-shifting the sweep. Cheap — polylines over a sparse grid.
    """
    cx, cy = int(W * 0.80), int(H * 0.42)   # off-centre focus, like the reference
    n_lines = 22
    for i in range(n_lines):
        spread = 0.6 + i * 0.12                     # each line fans out further
        phase = t * 0.25 + i * 0.14
        pts = []
        for s in range(0, W + 40, 40):
            x = s
            # curve that bows toward the focus and drifts with phase
            dx = (x - cx) / W
            y = cy + int(np.sin(dx * 3.1 + phase) * (120 + 60 * spread) * spread * 0.5)
            pts.append((x, y))
        shade = 0.20 + 0.5 * (i / n_lines)
        col = tuple(int(INDIGO[c] + (VIOLET_LO[c] - INDIGO[c]) * shade * intensity)
                    for c in range(3))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, col, 1, cv2.LINE_AA)


def _rot(px, py, cx, cy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return (int(cx + (px - cx) * c - (py - cy) * s),
            int(cy + (px - cx) * s + (py - cy) * c))


def draw_dragon_mark(canvas: np.ndarray, cx: int, cy: int, r: int,
                     color: tuple, bg: Optional[tuple] = None, glow: bool = False):
    """
    The Dragonwing dragon-head glyph from assets/2.png: a crescent sweep
    enclosing a hooked head with a negative-space eye. Built from filled
    ellipses/polygons so it scales cleanly on the canvas.
    """
    if glow:
        halo = canvas.copy()
        cv2.circle(halo, (cx, cy), int(r * 1.15), VIOLET, -1, cv2.LINE_AA)
        cv2.addWeighted(halo, 0.18, canvas, 0.82, 0, canvas)

    bg = bg if bg is not None else INDIGO_DEEP

    # Outer body: a thick crescent (large disc with an offset disc carved out).
    cv2.circle(canvas, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(cx - r * 0.42), int(cy - r * 0.30)),
               int(r * 0.86), bg, -1, cv2.LINE_AA)

    # Head/beak hook sweeping up and to the right.
    beak = np.array([
        _rot(cx + r * 0.10, cy - r * 0.86, cx, cy, 0),
        _rot(cx + r * 0.95, cy - r * 0.55, cx, cy, 0),
        _rot(cx + r * 0.55, cy - r * 0.18, cx, cy, 0),
        _rot(cx + r * 0.18, cy - r * 0.40, cx, cy, 0),
    ], dtype=np.int32)
    cv2.fillConvexPoly(canvas, beak, color, cv2.LINE_AA)

    # Flared tail curl sweeping down and to the right — the official-mark
    # geometry that closes the head + body into a near-circular emblem.
    tail_th = max(2, int(r * 0.17))
    cv2.ellipse(canvas, (int(cx + r * 0.12), int(cy + r * 0.44)),
                (int(r * 0.74), int(r * 0.54)), 0, -34, 122,
                color, tail_th, cv2.LINE_AA)

    # Inner curl (the eye socket), then the negative-space eye.
    cv2.circle(canvas, (int(cx + r * 0.30), int(cy + r * 0.12)),
               int(r * 0.34), bg, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(cx + r * 0.34), int(cy + r * 0.06)),
               int(r * 0.14), color, -1, cv2.LINE_AA)


def put_centered(img, text, cy, scale, color, thickness=1):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, cy), FONT, scale, color, thickness, cv2.LINE_AA)


def put_tracked(img, text, cx, cy, scale, color, thickness, spacing):
    """Centered text with manual letter-spacing — used for the DRAGONWING wordmark."""
    sizes = [cv2.getTextSize(ch, FONT, scale, thickness)[0][0] for ch in text]
    total = sum(sizes) + spacing * (len(text) - 1)
    x = cx - total // 2
    for ch, w in zip(text, sizes):
        cv2.putText(img, ch, (x, cy), FONT, scale, color, thickness, cv2.LINE_AA)
        x += w + spacing


def draw_wordmark(canvas: np.ndarray, cx: int, cy: int, scale: float = 1.0):
    """DRAGONWING tracked wordmark + DOORBELL subtitle lockup."""
    put_tracked(canvas, "DRAGONWING", cx, cy, 0.95 * scale, INK, 2, int(10 * scale))
    put_tracked(canvas, "D O O R B E L L", cx, cy + int(34 * scale),
                0.5 * scale, VIOLET_HI, 1, int(2 * scale))


def legibility_halo(canvas: np.ndarray, cx: int, cy: int, w: int, h: int,
                    strength: float = 0.72):
    """
    The one permitted 'shadow' (DESIGN.md Legibility-Halo Rule): a soft
    Deep-Indigo pad behind critical text or the mark when it sits over the
    camera feed or the wave field. Elliptical radial falloff, never a hard
    rectangle — for contrast, never decoration.
    """
    x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
    x1, y1 = min(W, cx + w // 2), min(H, cy + h // 2)
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    rh, rw = region.shape[:2]

    # Elliptical alpha: 1 at centre, feathering to 0 at the edge.
    yy, xx = np.ogrid[0:rh, 0:rw]
    nx = (xx - rw / 2) / (rw / 2)
    ny = (yy - rh / 2) / (rh / 2)
    d = np.sqrt(nx * nx + ny * ny)
    alpha = np.clip(1.0 - d, 0.0, 1.0) ** 1.6 * strength
    alpha = alpha[..., None]

    pad = np.array(INDIGO_DEEP, dtype=np.float32)
    region = region * (1.0 - alpha) + pad * alpha
    canvas[y0:y1, x0:x1] = region.astype(np.uint8)


def draw_access_banner(canvas: np.ndarray, text: str, color: tuple, t: float):
    """
    The signature verdict component (DESIGN.md Access Banner + Flat-Verdict
    Rule): a full-width, flat, square-cornered Deep-Indigo block carrying the
    verdict in display-scale type — the loudest thing on screen. No gradient,
    glow, or halo on the block; its certainty is pure solid color and size.
    The type pulses subtly on appearance.
    """
    band_h = 188
    y0 = (H - band_h) // 2
    y1 = y0 + band_h

    # Flat Deep-Indigo block, full width, square corners.
    canvas[y0:y1, 0:W] = INDIGO_DEEP
    # Thin signal-color rules top and bottom seat the block without softening it.
    cv2.line(canvas, (0, y0), (W, y0), color, 3, cv2.LINE_AA)
    cv2.line(canvas, (0, y1), (W, y1), color, 3, cv2.LINE_AA)

    # Display-scale verdict, subtly pulsing on appearance.
    pulse = 1.0 - 0.5 * np.exp(-t * 5.0) * abs(np.sin(t * 9.0))
    scale = 2.9 * pulse
    thickness = 6
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    tx = (W - tw) // 2
    ty = y0 + (band_h + th) // 2
    cv2.putText(canvas, text, (tx, ty), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_face_overlays(canvas: np.ndarray, faces: list, color: tuple,
                        sx: float, sy: float, ox: int, oy: int):
    """
    Draw bounding box, 5 key-points, and 106 landmarks.
    faces are in inference-frame coords; sx/sy/ox/oy map to canvas coords.
    """
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(canvas,
                      (int(x1 * sx + ox), int(y1 * sy + oy)),
                      (int(x2 * sx + ox), int(y2 * sy + oy)),
                      color, 2)
        if face.kps is not None:
            for x, y in face.kps:
                cv2.circle(canvas, (int(x * sx + ox), int(y * sy + oy)), 5, color, -1)
        lm = getattr(face, 'landmark_2d_106', None)
        if lm is not None:
            for x, y in lm:
                cv2.circle(canvas, (int(x * sx + ox), int(y * sy + oy)), 2, color, -1)
        # confidence score
        score = getattr(face, 'det_score', None)
        if score is not None:
            cv2.putText(canvas, f"{score:.2f}",
                        (int(x1 * sx + ox), int(y1 * sy + oy) - 6),
                        FONT, 0.45, color, 1, cv2.LINE_AA)


def draw_loading(canvas: np.ndarray, t: float):
    brand_gradient(canvas)
    draw_wave_field(canvas, t, intensity=0.7)
    cx, cy = W // 2, H // 2 - 40
    pulse = 0.5 + 0.5 * np.sin(t * 2.2)
    draw_dragon_mark(canvas, cx, cy, 86, VIOLET, glow=True)

    # Orbiting progress arc around the mark
    ang = (t * 2.0) % (2 * np.pi)
    deg = int(np.degrees(ang))
    cv2.ellipse(canvas, (cx, cy), (120, 120), 0, deg, deg + 90,
                VIOLET_HI, 3, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 120, VIOLET_LO, 1, cv2.LINE_AA)

    dots = "." * (int(t * 2) % 4)
    glow_c = tuple(int(MIST[c] + (VIOLET_HI[c] - MIST[c]) * pulse) for c in range(3))
    legibility_halo(canvas, cx, cy + 215, 440, 170, strength=0.6)
    draw_wordmark(canvas, cx, cy + 180, 1.1)
    put_centered(canvas, f"Initializing neural engines{dots}", cy + 250, 0.6, glow_c, 1)
    put_centered(canvas, "InsightFace buffalo_sc    HTP v75", cy + 282, 0.46, VIOLET_LO, 1)


def draw_standby(canvas: np.ndarray, t: float, db_count: int):
    brand_gradient(canvas)
    draw_wave_field(canvas, t, intensity=1.0)

    # Large dragon-mark watermark anchored off-centre, echoing the reference art.
    draw_dragon_mark(canvas, int(W * 0.5), int(H * 0.44), 150, INDIGO_LIFT)

    cx, cy = W // 2, H // 2
    pulse = 0.5 + 0.5 * np.sin(t * 2.4)

    # Pulsing violet ring enclosing a lock glyph
    r_outer = int(104 + 14 * pulse)
    ring_c = tuple(int(VIOLET_LO[c] + (VIOLET[c] - VIOLET_LO[c]) * pulse) for c in range(3))
    cv2.circle(canvas, (cx, cy), r_outer, ring_c, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r_outer - 16, VIOLET_LO, 1, cv2.LINE_AA)

    # Padlock glyph, violet body on indigo
    lw, lh = 46, 36
    lx, ly = cx - lw // 2, cy - lh // 2 + 8
    cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), VIOLET, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), VIOLET_HI, 1, cv2.LINE_AA)
    shackle_c = tuple(int(VIOLET[c] + (VIOLET_HI[c] - VIOLET[c]) * pulse) for c in range(3))
    cv2.ellipse(canvas, (cx, ly), (lw // 3, lh // 2), 0, 180, 360, shackle_c, 4, cv2.LINE_AA)
    cv2.circle(canvas, (cx, ly + lh // 2 - 2), 5, INDIGO_DEEP, -1, cv2.LINE_AA)

    legibility_halo(canvas, cx, cy + 142, 430, 110, strength=0.62)
    put_centered(canvas, "DOORBELL READY", cy + 128, 0.95, INK, 2)
    put_centered(canvas, "Awaiting visitor", cy + 162, 0.56, MIST, 1)

    # DB badge, top-right — sits below the HUD bar to avoid the FPS badge
    badge = f"{db_count} enrolled" if db_count else "no faces enrolled"
    (tw, _), _ = cv2.getTextSize(badge, FONT, 0.5, 1)
    cv2.circle(canvas, (W - tw - 30, 50), 4, UNLOCK if db_count else VIOLET_LO, -1, cv2.LINE_AA)
    cv2.putText(canvas, badge, (W - tw - 18, 55), FONT, 0.5, MIST, 1, cv2.LINE_AA)


def composite_frame(canvas: np.ndarray, frame: np.ndarray):
    """Letterbox-fit frame into canvas over the indigo ground, return (sx, sy, ox, oy)."""
    fh, fw = frame.shape[:2]
    scale = min(W / fw, H / fh)
    dw, dh = int(fw * scale), int(fh * scale)
    ox, oy = (W - dw) // 2, (H - dh) // 2
    brand_gradient(canvas)
    canvas[oy:oy + dh, ox:ox + dw] = cv2.resize(frame, (dw, dh))
    return scale, scale, ox, oy


def draw_camera_state(canvas: np.ndarray, frame: np.ndarray, faces: list,
                       inf_w: int, inf_h: int,
                       border: tuple, label: str, label_color: tuple,
                       reg_progress: Optional[tuple] = None):
    scale, _, ox, oy = composite_frame(canvas, frame)

    fh, fw = frame.shape[:2]
    # Inference frame is inf_w × inf_h; faces map onto the full canvas.
    lm_sx = W / inf_w
    lm_sy = H / inf_h
    draw_face_overlays(canvas, faces, border, lm_sx, lm_sy, 0, 0)

    # Corner brackets in the state color (violet when neutral, green/red/amber per state)
    fx2, fy2 = ox + int(fw * scale), oy + int(fh * scale)
    bl = 54
    for bx, by, sx2, sy2 in [(ox, oy, 1, 1), (fx2, oy, -1, 1),
                             (ox, fy2, 1, -1), (fx2, fy2, -1, -1)]:
        cv2.line(canvas, (bx, by), (bx + sx2 * bl, by), border, 3, cv2.LINE_AA)
        cv2.line(canvas, (bx, by), (bx, by + sy2 * bl), border, 3, cv2.LINE_AA)

    # Bottom status bar — indigo wash + small dragon lockup + state label
    panel_y = H - 84
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, panel_y), (W, H), INDIGO_DEEP, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.line(canvas, (0, panel_y), (W, panel_y), VIOLET_LO, 1, cv2.LINE_AA)
    draw_dragon_mark(canvas, 54, panel_y + 42, 24, VIOLET, bg=INDIGO_DEEP)

    # Status dot + label, left-aligned after the mark
    cv2.circle(canvas, (104, panel_y + 42), 6, label_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label.strip(), (122, panel_y + 50),
                FONT, 0.85, label_color, 2, cv2.LINE_AA)

    if reg_progress:
        n, total = reg_progress
        bar_w = int(W * 0.5)
        bar_x = W - bar_w - 40
        bar_y = panel_y + 36
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12),
                      INDIGO_LIFT, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12),
                      VIOLET_LO, 1, cv2.LINE_AA)
        fill = int(bar_w * n / total) if total else 0
        if fill > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + 12),
                          VIOLET, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{n}/{total}", (bar_x + bar_w + 10, bar_y + 11),
                    FONT, 0.5, MIST, 1, cv2.LINE_AA)


def draw_qr_authorized(canvas: np.ndarray, elapsed: float, name: str):
    """Phase 1 (0-5s): QR accepted — violet burst + dragon mark + AUTHORIZED."""
    brand_gradient(canvas)
    draw_wave_field(canvas, elapsed, intensity=0.6)
    cx, cy = W // 2, H // 2 - 20

    # Expanding violet ring burst
    for i in range(4):
        phase = elapsed * 2.2 - i * 0.45
        frac = phase % 1.0
        r = int((70 + 150 * frac) * (1.0 - frac * 0.4))
        alpha = max(0.0, 1.0 - frac)
        color = tuple(int(INDIGO[c] + (VIOLET_HI[c] - INDIGO[c]) * alpha) for c in range(3))
        if r > 0:
            cv2.circle(canvas, (cx, cy), r, color, max(1, int(4 * alpha)), cv2.LINE_AA)

    # Dragon mark bursts to full presence
    pulse = 0.6 + 0.4 * abs(np.sin(elapsed * 5))
    draw_dragon_mark(canvas, cx, cy, int(72 * pulse), VIOLET, glow=True)

    # Green confirmation check, ring'd — the one semantic (unlock) accent
    scale_in = min(1.0, elapsed * 2.5)
    cv2.circle(canvas, (cx, cy), int(72 * pulse) + 14, UNLOCK, 2, cv2.LINE_AA)

    legibility_halo(canvas, cx, cy + 180, 460, 120, strength=0.55)
    put_centered(canvas, "AUTHORIZED", cy + 150, 1.6 * scale_in, INK, 3)
    put_centered(canvas, name, cy + 196, 0.72 * scale_in, VIOLET_HI, 1)


def draw_qr_countdown(canvas: np.ndarray, elapsed: float, name: str, countdown: int):
    """Phase 2 (5-10s): get-ready countdown before capture begins."""
    brand_gradient(canvas)
    draw_wave_field(canvas, elapsed + 3.0, intensity=0.6)
    cx, cy = W // 2, H // 2 - 20

    # Depleting violet sweep ring
    frac = max(0.0, min(1.0, 1.0 - elapsed / 5.0))
    cv2.ellipse(canvas, (cx, cy), (118, 118), -90, 0, 360, INDIGO_LIFT, 8, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (118, 118), -90, 0, int(360 * frac), VIOLET, 8, cv2.LINE_AA)

    # Big countdown digit
    digit = str(max(0, countdown))
    txt_scale = 4.6 + 0.6 * abs(np.sin(elapsed * 3.14))
    legibility_halo(canvas, cx, cy + 20, 240, 240, strength=0.5)
    put_centered(canvas, digit, cy + 42, txt_scale, INK, 6)

    legibility_halo(canvas, cx, cy + 185, 420, 110, strength=0.55)
    put_centered(canvas, "GET READY", cy + 168, 0.9, VIOLET_HI, 2)
    put_centered(canvas, f"Enrolling  {name}", cy + 206, 0.56, MIST, 1)


def draw_hud(canvas: np.ndarray, state: State, status: str, db_count: int,
             fps: float = 0.0, backend: str = "cpu"):
    hint = "B toggle CPU/NPU   K known   U unknown   R enroll   ESC standby   Q quit"

    # Top bar
    cv2.rectangle(canvas, (0, 0), (W, 24), INDIGO_DEEP, -1)
    cv2.line(canvas, (0, 24), (W, 24), VIOLET_LO, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{state.name:<13}", (10, 17),
                FONT_MONO, 1.1, VIOLET_HI, 1, cv2.LINE_AA)
    cv2.putText(canvas, status, (150, 17), FONT_MONO, 1.0, MIST, 1, cv2.LINE_AA)

    # Bottom hint bar
    cv2.rectangle(canvas, (0, H - 22), (W, H), INDIGO_DEEP, -1)
    cv2.putText(canvas, hint, (10, H - 7), FONT_MONO, 0.95, VIOLET_LO, 1, cv2.LINE_AA)

    # FPS badge — top right. Violet when NPU active, mist when CPU.
    is_npu = "npu" in backend
    fps_color = VIOLET_HI if is_npu else MIST
    label = f"{backend.upper()}  {fps:4.1f} FPS"
    (tw, _), _ = cv2.getTextSize(label, FONT_MONO, 1.1, 1)
    bx = W - tw - 16
    cv2.circle(canvas, (bx - 10, 13), 4, UNLOCK if is_npu else VIOLET_LO, -1, cv2.LINE_AA)
    cv2.putText(canvas, label, (bx, 17), FONT_MONO, 1.1, fps_color, 1, cv2.LINE_AA)


# ── System stats monitor ───────────────────────────────────────────────────────

class StatsMonitor(threading.Thread):
    """Polls CPU % and RAM every 0.5 s in a background thread."""
    def __init__(self):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._cpu_per_core: list = []
        self._cpu_total: float = 0.0
        self._ram_pct: float = 0.0
        self._ram_used_mb: float = 0.0
        self._ram_total_mb: float = 0.0
        self.running = True

    def run(self):
        import psutil
        psutil.cpu_percent(percpu=True)   # discard first sample (always 0)
        while self.running:
            time.sleep(0.5)
            cores = psutil.cpu_percent(percpu=True)
            total = sum(cores) / len(cores) if cores else 0.0
            m = psutil.virtual_memory()
            with self._lock:
                self._cpu_per_core = cores
                self._cpu_total = total
                self._ram_pct = m.percent
                self._ram_used_mb  = m.used  / 1024 / 1024
                self._ram_total_mb = m.total / 1024 / 1024

    def snapshot(self):
        with self._lock:
            return (list(self._cpu_per_core), self._cpu_total,
                    self._ram_pct, self._ram_used_mb, self._ram_total_mb)

    def stop(self):
        self.running = False


def _bar(w: int, pct: float, color: tuple) -> np.ndarray:
    """Horizontal meter: indigo track, colored fill."""
    bar = np.empty((8, w, 3), dtype=np.uint8)
    bar[:] = INDIGO_LIFT
    fill = max(1, int(w * min(pct, 100.0) / 100.0))
    bar[:, :fill] = color
    return bar


def draw_stats_panel(canvas: np.ndarray,
                     cpu_cores: list, cpu_total: float,
                     ram_pct: float, ram_used_mb: float, ram_total_mb: float,
                     cpu_fps: float, npu_fps: float,
                     active_backend: str, npu_ready: bool):
    """
    Translucent indigo telemetry panel, bottom-right:
      - CPU / NPU FPS rows (active one highlighted in violet)
      - Per-core CPU bars (load heat is a meter, kept green→red)
      - RAM bar
    """
    PW, PH = 262, 212
    px = W - PW - 12
    py = H - PH - 34           # above the hint bar

    overlay = canvas.copy()
    cv2.rectangle(overlay, (px - 8, py - 8), (px + PW + 8, py + PH + 8),
                  INDIGO_DEEP, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (px - 8, py - 8), (px + PW + 8, py + PH + 8),
                  VIOLET_LO, 1, cv2.LINE_AA)

    y = py + 16
    F = cv2.FONT_HERSHEY_PLAIN

    # ── Backend FPS rows ─────────────────────────────────────────────────────
    for label, fps, is_active, ready in [
        ("CPU", cpu_fps, active_backend == "cpu", True),
        ("NPU", npu_fps, active_backend == "npu", npu_ready),
    ]:
        if is_active:
            cv2.rectangle(canvas, (px - 4, y - 12), (px + PW + 4, y + 6),
                          INDIGO_LIFT, -1)
            cv2.rectangle(canvas, (px - 4, y - 12), (px - 2, y + 6), VIOLET, -1)

        if not ready:
            fps_str, col = "loading", VIOLET_LO
        else:
            fps_str = f"{fps:5.1f} FPS"
            col = VIOLET_HI if is_active else MIST

        if is_active:
            tri = np.array([(px, y - 8), (px, y), (px + 7, y - 4)], np.int32)
            cv2.fillConvexPoly(canvas, tri, VIOLET, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 14, y), F, 1.1, col, 1, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(fps_str, F, 1.1, 1)
        cv2.putText(canvas, fps_str, (px + PW - sw, y), F, 1.1, col, 1, cv2.LINE_AA)
        y += 20

    y += 6
    cv2.line(canvas, (px, y), (px + PW, y), VIOLET_LO, 1, cv2.LINE_AA)
    y += 12

    # ── Per-core CPU bars ─────────────────────────────────────────────────────
    cv2.putText(canvas, f"CPU CORES   avg {cpu_total:.0f}%",
                (px, y), F, 0.95, MIST, 1, cv2.LINE_AA)
    y += 12

    cols_n = 4
    bar_w = (PW - (cols_n - 1) * 4) // cols_n
    for i, pct in enumerate(cpu_cores):
        bx = px + (i % cols_n) * (bar_w + 4)
        by = y + (i // cols_n) * 18
        heat = min(255, int(pct * 2.55))
        bar_col = (0, 255 - heat, heat)       # green→red as load rises
        canvas[by:by + 8, bx:bx + bar_w] = _bar(bar_w, pct, bar_col)
        cv2.putText(canvas, f"C{i}", (bx, by + 18), F, 0.75, VIOLET_LO, 1, cv2.LINE_AA)

    row_count = (len(cpu_cores) + cols_n - 1) // cols_n
    y += row_count * 18 + 10

    cv2.line(canvas, (px, y), (px + PW, y), VIOLET_LO, 1, cv2.LINE_AA)
    y += 12

    # ── RAM bar ───────────────────────────────────────────────────────────────
    cv2.putText(canvas, f"RAM   {ram_used_mb:.0f} / {ram_total_mb:.0f} MB   {ram_pct:.0f}%",
                (px, y), F, 0.95, MIST, 1, cv2.LINE_AA)
    y += 10
    ram_col = (0, int(255 * (1 - ram_pct / 100)), int(255 * ram_pct / 100))
    canvas[y:y + 8, px:px + PW] = _bar(PW, ram_pct, ram_col)


# ── Main ──────────────────────────────────────────────────────────────────────

INF_W, INF_H = 320, 240  # inference frame size on board (aarch64 CPU)


def main():
    parser = argparse.ArgumentParser(description="Doorbell demo — InsightFace face recognition")
    parser.add_argument("--backend", choices=["cpu", "npu"], default="cpu",
                        help="Starting backend (default: cpu). Press B to toggle live.")
    args = parser.parse_args()

    app = AppState()
    camera = CameraThread()
    camera.start()

    # Track readiness of each engine separately
    _ready = {"cpu": False, "npu": False}

    def on_cpu_ready():
        _ready["cpu"] = True
        with app.lock:
            if app.state == State.LOADING:
                app.state = State.STANDBY
                app.status_msg = "SYSTEM READY"
        print("[CPU] InsightFace ready")

    def on_npu_ready():
        _ready["npu"] = True
        with app.lock:
            if app.state == State.LOADING:
                app.state = State.STANDBY
                app.status_msg = "SYSTEM READY"
        print("[NPU] qnn-net-run batch engine ready")

    cpu_engine = FaceEngine(on_ready=on_cpu_ready)
    npu_engine = NpuFaceEngine(on_ready=on_npu_ready)
    stats = StatsMonitor()
    cpu_engine.start()
    npu_engine.start()
    stats.start()

    # Mutable backend state — toggled by B key
    active = {"backend": args.backend}

    def active_engine():
        return npu_engine if active["backend"] == "npu" else cpu_engine

    def finish_registration(name: str, embeddings: list):
        with app.lock:
            app.add_face(name, embeddings)
            app.state = State.RECOGNIZED
            app.last_match = name
            app.status_msg = f"Registered! Welcome, {name}"
        print(f"[REG] Done — {name} added with {len(embeddings)} embeddings (TTL 120s)")

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    t0 = time.time()
    qr_detector = QRDetector()
    inf_frame_cache: Optional[np.ndarray] = None
    verdict_since = {"state": None, "at": 0.0}   # pulse clock for the access banner

    cv2.namedWindow("Dragonwing Doorbell", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Dragonwing Doorbell", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("=== DRAGONWING DOORBELL ===")
    print("B=toggle CPU/NPU  K=known  U=unknown  R=enroll  ESC=standby  Q=quit")
    print("===========================")

    while True:
        t = time.time() - t0
        now = time.time()
        eng = active_engine()
        backend = active["backend"]

        # Feed latest frame to active engine (downscaled for speed)
        raw_frame = camera.get_frame()
        if raw_frame is not None:
            inf_frame = cv2.resize(raw_frame, (INF_W, INF_H))
            eng.submit(inf_frame)
            inf_frame_cache = inf_frame

        faces = eng.get_faces()

        # QR detection — runs on every frame, triggers registration automatically
        qr_name = None
        if raw_frame is not None:
            qr_name = qr_detector.detect(raw_frame)
            if qr_name:
                print(f"[QR] Detected LinkedIn profile: {qr_name}")

        with app.lock:
            state = app.state
            # QR triggers authorized animation from any active state
            if qr_name and state not in (State.LOADING, State.REGISTRATION, State.QR_AUTHORIZED):
                app.state = State.QR_AUTHORIZED
                app.reg_name = qr_name
                app.qr_authorized_at = now
                app.status_msg = f"QR authorized: {qr_name}"
                state = State.QR_AUTHORIZED

            # Auto state transitions
            if state == State.QR_AUTHORIZED:
                elapsed_qr = now - app.qr_authorized_at
                if elapsed_qr >= 10.0:
                    app.state = State.REGISTRATION
                    app.reg_frames = 0
                    app.reg_start_at = now
                    app.status_msg = f"Registering {app.reg_name} — look at camera"
                    state = State.REGISTRATION
                    reg_thread = RegistrationDriver(app, eng, finish_registration)
                    reg_thread.start()
            elif state not in (State.LOADING, State.REGISTRATION):
                if faces:
                    app.last_face_time = now
                    if state == State.STANDBY:
                        app.state = State.DETECTION
                        app.status_msg = "Face detected — recognizing..."
                        state = State.DETECTION
                    if state in (State.DETECTION, State.RECOGNIZED, State.UNKNOWN):
                        emb = faces[0].embedding
                        if emb is not None:
                            match = recognize_face(emb, app.face_db)
                            if match:
                                app.state = State.RECOGNIZED
                                app.last_match = match
                                app.status_msg = f"Welcome, {match}!"
                            else:
                                app.state = State.UNKNOWN
                                app.last_match = None
                                app.status_msg = "Unknown person — access denied"
                else:
                    if state != State.STANDBY and now - app.last_face_time > NO_FACE_TIMEOUT:
                        app.state = State.STANDBY
                        app.last_match = None
                        app.status_msg = "SYSTEM READY"

            state    = app.state
            match    = app.last_match
            status   = app.status_msg
            reg_n    = app.reg_frames
            reg_tot  = app.reg_target
            db_count = len(app.face_db)

        app.prune_expired()

        # Render
        if state == State.LOADING:
            draw_loading(canvas, t)
        elif state == State.STANDBY or raw_frame is None:
            draw_standby(canvas, t, db_count)
        elif state == State.QR_AUTHORIZED:
            with app.lock:
                elapsed_qr = now - app.qr_authorized_at
                qr_name_cur = app.reg_name
            if elapsed_qr < 5.0:
                draw_qr_authorized(canvas, elapsed_qr, qr_name_cur)
            else:
                countdown = max(0, 5 - int(elapsed_qr - 5.0))
                draw_qr_countdown(canvas, elapsed_qr - 5.0, qr_name_cur, countdown)
        else:
            if state == State.DETECTION:
                draw_camera_state(canvas, raw_frame, faces, INF_W, INF_H,
                                  DETECT, "RECOGNIZING", DETECT)
            elif state == State.RECOGNIZED:
                draw_camera_state(canvas, raw_frame, faces, INF_W, INF_H,
                                  UNLOCK, f"UNLOCKED   {match or '?'}", UNLOCK)
            elif state == State.UNKNOWN:
                draw_camera_state(canvas, raw_frame, faces, INF_W, INF_H,
                                  LOCK, "ACCESS DENIED", LOCK)
            elif state == State.REGISTRATION:
                draw_camera_state(canvas, raw_frame, faces, INF_W, INF_H,
                                  VIOLET, f"ENROLLING   {app.reg_name}", VIOLET,
                                  reg_progress=(reg_n, reg_tot))

            # The loud verdict (DESIGN.md signature): a full-width flat banner
            # over the live feed, display-scale, pulsing on appearance.
            if state in (State.RECOGNIZED, State.UNKNOWN):
                if verdict_since["state"] != state:
                    verdict_since["state"] = state
                    verdict_since["at"] = now
                vt = now - verdict_since["at"]
                if state == State.RECOGNIZED:
                    name = (match or "").upper()
                    btext = f"UNLOCKED  {name}".strip()
                    draw_access_banner(canvas, btext, UNLOCK, vt)
                else:
                    draw_access_banner(canvas, "ACCESS DENIED", LOCK, vt)
            else:
                verdict_since["state"] = None

        npu_tag = "" if _ready["npu"] else " (loading...)"
        draw_hud(canvas, state, status, db_count,
                 fps=eng.fps_counter.fps(),
                 backend=backend + (npu_tag if backend == "npu" else ""))

        cpu_cores, cpu_total, ram_pct, ram_used, ram_total = stats.snapshot()
        draw_stats_panel(canvas,
                         cpu_cores, cpu_total, ram_pct, ram_used, ram_total,
                         cpu_fps=cpu_engine.fps_counter.fps(),
                         npu_fps=npu_engine.fps_counter.fps(),
                         active_backend=active["backend"],
                         npu_ready=_ready["npu"])
        cv2.imshow("Dragonwing Doorbell", canvas)

        key = cv2.waitKey(33) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == 27:  # ESC
            with app.lock:
                app.state = State.STANDBY
                app.last_match = None
                app.status_msg = "SYSTEM READY"
            print("[STATE] → STANDBY (manual)")
        elif key == ord('k') or key == ord('K'):
            with app.lock:
                if app.state in (State.DETECTION, State.UNKNOWN, State.RECOGNIZED):
                    name = random.choice(["Alice", "Bob"])
                    app.last_match = name
                    app.state = State.RECOGNIZED
                    app.status_msg = f"Welcome, {name}! (manual)"
            print("[SIM] K → RECOGNIZED")
        elif key == ord('u') or key == ord('U'):
            with app.lock:
                if app.state in (State.DETECTION, State.RECOGNIZED):
                    app.last_match = None
                    app.state = State.UNKNOWN
                    app.status_msg = "Unknown person (manual)"
            print("[SIM] U → UNKNOWN")
        elif key == ord('b') or key == ord('B'):
            new_be = "npu" if active["backend"] == "cpu" else "cpu"
            if new_be == "npu" and not _ready["npu"]:
                print("[B] NPU still loading, please wait...")
            else:
                active["backend"] = new_be
                print(f"[B] Switched to {new_be.upper()}")
        elif key == ord('r') or key == ord('R'):
            with app.lock:
                can_reg = app.state in (State.DETECTION, State.UNKNOWN, State.RECOGNIZED, State.STANDBY)
                if can_reg:
                    app.state = State.QR_AUTHORIZED
                    app.reg_name = f"Guest_{int(time.time()) % 10000}"
                    app.qr_authorized_at = time.time()
                    app.status_msg = f"QR authorized: {app.reg_name}"
                    name_copy = app.reg_name
            if can_reg:
                print(f"[SIM] R → QR_AUTHORIZED ({name_copy})")

    cpu_engine.stop()
    npu_engine.stop()
    stats.stop()
    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
