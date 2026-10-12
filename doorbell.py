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
    BATCH_TIMEOUT = 0.10    # seconds — collect frames for up to this long, then run.
                            # Kept short on purpose: this wait is the dominant source of
                            # overlay lag (the box/dots trailing the moving face). Smaller
                            # batches cost a little throughput (each ~280ms qnn-net-run spawn
                            # processes fewer frames) but the overlay tracks far more tightly.

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
        try:
            data, pts, _ = self._detector.detectAndDecode(frame)
        except cv2.error:
            # OpenCV 5.0's QR decoder can throw on degenerate candidate
            # contours (zero-area src points) for certain camera frames.
            # It's a transient per-frame failure — never let it kill the
            # unattended demo; just skip QR detection for this frame.
            return None
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
        self.cap = self._open_camera()
        if self.cap is None or not self.cap.isOpened():
            print("[Camera] No camera — using synthetic frames")

    @staticmethod
    def _open_camera():
        """
        Open the first V4L2 index that both opens and yields a real frame.
        A USB webcam can re-enumerate across reboots/replugs (e.g. video0 ->
        video1), so we probe a few indices instead of hardcoding 0 — otherwise
        a moved camera leaves the app on synthetic frames or a stale handle.
        """
        for idx in range(6):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"[Camera] Opened /dev/video{idx}")
                    return cap
            cap.release()
        return None

    def run(self):
        t = 0.0
        while self.running:
            if self.cap is not None and self.cap.isOpened():
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
        if self.cap is not None and self.cap.isOpened():
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

# Native panel resolution. The demo HDMI panel is 1920x1080, so we author and
# render the canvas at that size — drawing every Hershey glyph and vector stroke
# directly into the panel's pixel grid instead of drawing at 720p and letting the
# fullscreen window interpolate it up 1.5x (which is what made text/logo soft).
W, H = 1920, 1080
S = H / 720.0            # UI scale vs. the original 720p authoring (= 1.5)


def _s(v) -> int:
    """Scale a 720p-authored pixel size/offset to the native canvas."""
    return int(round(v * S))


def _th(v) -> int:
    """Scale a stroke thickness, never below 1px."""
    return max(1, int(round(v * S)))

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

# ── Official Dragonwing mark (embedded) ──────────────────────────────────────
# The real circular dragon-head mark, cropped from assets/Qualcomm-Dragonwing-
# logo.jpeg and recolored light-on-transparent so it reads on the indigo bar.
# Embedded as base64 so the single-file scp-to-board deploy needs no assets dir.
_DRAGON_MARK_PNG_B64 = (
    "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"
    "RU5ErkJggg=="
)
_DRAGON_MARK: Optional[np.ndarray] = None


def _load_dragon_mark() -> Optional[np.ndarray]:
    """Decode the embedded mark to a BGRA array once; None on any failure."""
    global _DRAGON_MARK
    if _DRAGON_MARK is None:
        try:
            import base64
            raw = base64.b64decode(_DRAGON_MARK_PNG_B64)
            arr = np.frombuffer(raw, np.uint8)
            _DRAGON_MARK = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        except Exception:
            _DRAGON_MARK = None
    return _DRAGON_MARK


def draw_logo_mark(canvas: np.ndarray, cx: int, cy: int, size: int,
                   fallback_color: tuple = VIOLET):
    """
    Blit the official Dragonwing mark centered at (cx, cy), scaled to `size`.
    Alpha-composited over whatever is already on the canvas. Falls back to the
    drawn vector glyph if the embedded asset can't be decoded.
    """
    mark = _load_dragon_mark()
    if mark is None or mark.shape[2] != 4:
        draw_dragon_mark(canvas, cx, cy, size // 2, fallback_color, bg=INDIGO_DEEP)
        return
    m = cv2.resize(mark, (size, size), interpolation=cv2.INTER_AREA)
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sub = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    roi = roi * (1.0 - a) + sub[:, :, :3].astype(np.float32) * a
    canvas[cy0:cy1, cx0:cx1] = roi.astype(np.uint8)


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
    step = _s(40)
    for i in range(n_lines):
        spread = 0.6 + i * 0.12                     # each line fans out further
        phase = t * 0.25 + i * 0.14
        pts = []
        for x in range(0, W + step, step):
            # curve that bows toward the focus and drifts with phase
            dx = (x - cx) / W
            y = cy + int(np.sin(dx * 3.1 + phase) * (120 + 60 * spread) * spread * 0.5 * S)
            pts.append((x, y))
        shade = 0.20 + 0.5 * (i / n_lines)
        col = tuple(int(INDIGO[c] + (VIOLET_LO[c] - INDIGO[c]) * shade * intensity)
                    for c in range(3))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, col, _th(1), cv2.LINE_AA)


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
    """Centered text. `scale`/`thickness` are in logical 720p units (×S here);
    `cy` is a native-canvas coordinate."""
    scale *= S
    thickness = _th(thickness)
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, cy), FONT, scale, color, thickness, cv2.LINE_AA)


def put_tracked(img, text, cx, cy, scale, color, thickness, spacing):
    """Centered text with manual letter-spacing — used for the DRAGONWING wordmark.
    `scale`/`thickness` are logical 720p units (×S here); `cx`/`cy`/`spacing`
    are native-canvas pixels."""
    scale *= S
    thickness = _th(thickness)
    sizes = [cv2.getTextSize(ch, FONT, scale, thickness)[0][0] for ch in text]
    total = sum(sizes) + spacing * (len(text) - 1)
    x = cx - total // 2
    for ch, w in zip(text, sizes):
        cv2.putText(img, ch, (x, cy), FONT, scale, color, thickness, cv2.LINE_AA)
        x += w + spacing


def draw_wordmark(canvas: np.ndarray, cx: int, cy: int, scale: float = 1.0):
    """DRAGONWING tracked wordmark + DOORBELL subtitle lockup."""
    put_tracked(canvas, "DRAGONWING", cx, cy, 0.95 * scale, INK, 2, _s(10 * scale))
    put_tracked(canvas, "D O O R B E L L", cx, cy + _s(34 * scale),
                0.5 * scale, VIOLET_HI, 1, _s(2 * scale))


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
    # Low band, clear of the face: sits in the lower third, above the status bar.
    band_h = _s(104)
    y1 = H - _s(96)
    y0 = y1 - band_h

    # Flat Deep-Indigo block, full width, square corners.
    canvas[y0:y1, 0:W] = INDIGO_DEEP
    # Thin signal-color rules top and bottom seat the block without softening it.
    cv2.line(canvas, (0, y0), (W, y0), color, _th(3), cv2.LINE_AA)
    cv2.line(canvas, (0, y1), (W, y1), color, _th(3), cv2.LINE_AA)

    # Display-scale verdict, subtly pulsing on appearance.
    pulse = 1.0 - 0.5 * np.exp(-t * 5.0) * abs(np.sin(t * 9.0))
    scale = 1.7 * pulse * S
    thickness = _th(4)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    tx = (W - tw) // 2
    ty = y0 + (band_h + th) // 2
    cv2.putText(canvas, text, (tx, ty), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_face_overlays(canvas: np.ndarray, faces: list, color: tuple,
                        sx: float, sy: float, ox: int, oy: int):
    """
    Draw bounding box + 5 key-points. faces are in inference-frame coords;
    sx/sy/ox/oy map them onto the letterboxed display rectangle. The dense
    106-landmark cloud is intentionally skipped — it costs 100+ draws per
    face per frame and adds visual noise without aiding the demo read.
    """
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(canvas,
                      (int(x1 * sx + ox), int(y1 * sy + oy)),
                      (int(x2 * sx + ox), int(y2 * sy + oy)),
                      color, _th(2))
        if face.kps is not None:
            for x, y in face.kps:
                cv2.circle(canvas, (int(x * sx + ox), int(y * sy + oy)), _s(4), color, -1)
        # confidence score
        score = getattr(face, 'det_score', None)
        if score is not None:
            cv2.putText(canvas, f"{score:.2f}",
                        (int(x1 * sx + ox), int(y1 * sy + oy) - _s(6)),
                        FONT, 0.45 * S, color, _th(1), cv2.LINE_AA)


def draw_loading(canvas: np.ndarray, t: float):
    brand_gradient(canvas)
    draw_wave_field(canvas, t, intensity=0.7)
    cx, cy = W // 2, H // 2 - _s(40)
    pulse = 0.5 + 0.5 * np.sin(t * 2.2)
    draw_dragon_mark(canvas, cx, cy, _s(86), VIOLET, glow=True)

    # Orbiting progress arc around the mark
    ang = (t * 2.0) % (2 * np.pi)
    deg = int(np.degrees(ang))
    cv2.ellipse(canvas, (cx, cy), (_s(120), _s(120)), 0, deg, deg + 90,
                VIOLET_HI, _th(3), cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), _s(120), VIOLET_LO, _th(1), cv2.LINE_AA)

    dots = "." * (int(t * 2) % 4)
    glow_c = tuple(int(MIST[c] + (VIOLET_HI[c] - MIST[c]) * pulse) for c in range(3))
    legibility_halo(canvas, cx, cy + _s(215), _s(440), _s(170), strength=0.6)
    draw_wordmark(canvas, cx, cy + _s(180), 1.1)
    put_centered(canvas, f"Initializing neural engines{dots}", cy + _s(250), 0.6, glow_c, 1)
    put_centered(canvas, "InsightFace buffalo_sc    HTP v75", cy + _s(282), 0.46, VIOLET_LO, 1)


def draw_standby(canvas: np.ndarray, t: float, db_count: int):
    brand_gradient(canvas)
    draw_wave_field(canvas, t, intensity=1.0)

    # Large dragon-mark watermark anchored off-centre, echoing the reference art.
    draw_dragon_mark(canvas, int(W * 0.5), int(H * 0.44), _s(150), INDIGO_LIFT)

    cx, cy = W // 2, H // 2
    pulse = 0.5 + 0.5 * np.sin(t * 2.4)

    # Pulsing violet ring enclosing a lock glyph
    r_outer = _s(104 + 14 * pulse)
    ring_c = tuple(int(VIOLET_LO[c] + (VIOLET[c] - VIOLET_LO[c]) * pulse) for c in range(3))
    cv2.circle(canvas, (cx, cy), r_outer, ring_c, _th(2), cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), r_outer - _s(16), VIOLET_LO, _th(1), cv2.LINE_AA)

    # Padlock glyph, violet body on indigo
    lw, lh = _s(46), _s(36)
    lx, ly = cx - lw // 2, cy - lh // 2 + _s(8)
    cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), VIOLET, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), VIOLET_HI, _th(1), cv2.LINE_AA)
    shackle_c = tuple(int(VIOLET[c] + (VIOLET_HI[c] - VIOLET[c]) * pulse) for c in range(3))
    cv2.ellipse(canvas, (cx, ly), (lw // 3, lh // 2), 0, 180, 360, shackle_c, _th(4), cv2.LINE_AA)
    cv2.circle(canvas, (cx, ly + lh // 2 - _s(2)), _s(5), INDIGO_DEEP, -1, cv2.LINE_AA)

    legibility_halo(canvas, cx, cy + _s(142), _s(430), _s(110), strength=0.62)
    put_centered(canvas, "DOORBELL READY", cy + _s(128), 0.95, INK, 2)
    put_centered(canvas, "Awaiting visitor", cy + _s(162), 0.56, MIST, 1)

    # DB badge, top-right — sits below the HUD bar to avoid the FPS badge
    badge = f"{db_count} enrolled" if db_count else "no faces enrolled"
    (tw, _), _ = cv2.getTextSize(badge, FONT, 0.5 * S, 1)
    cv2.circle(canvas, (W - tw - _s(30), _s(50)), _s(4), UNLOCK if db_count else VIOLET_LO, -1, cv2.LINE_AA)
    cv2.putText(canvas, badge, (W - tw - _s(18), _s(55)), FONT, 0.5 * S, MIST, _th(1), cv2.LINE_AA)


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
    # Map inference-frame coords (inf_w×inf_h) onto the *letterboxed* display
    # rectangle, not the full canvas: through the camera frame, then the same
    # scale+offset the frame was composited with. Keeps the box on the face.
    dw, dh = fw * scale, fh * scale
    lm_sx = dw / inf_w
    lm_sy = dh / inf_h
    draw_face_overlays(canvas, faces, border, lm_sx, lm_sy, ox, oy)

    # Corner brackets in the state color (violet when neutral, green/red/amber per state)
    fx2, fy2 = ox + int(fw * scale), oy + int(fh * scale)
    bl = _s(54)
    bt = _th(3)
    for bx, by, sx2, sy2 in [(ox, oy, 1, 1), (fx2, oy, -1, 1),
                             (ox, fy2, 1, -1), (fx2, fy2, -1, -1)]:
        cv2.line(canvas, (bx, by), (bx + sx2 * bl, by), border, bt, cv2.LINE_AA)
        cv2.line(canvas, (bx, by), (bx, by + sy2 * bl), border, bt, cv2.LINE_AA)

    # Bottom status bar — indigo wash + logo lockup + state label. Seated above
    # the bottom hint strip so the logo and verdict never sit under the controls.
    bar_h = _s(84)
    panel_y = H - _s(22) - bar_h          # clear of the hint bar (H - 22 tall)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, panel_y), (W, H - _s(22)), INDIGO_DEEP, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.line(canvas, (0, panel_y), (W, panel_y), VIOLET_LO, _th(1), cv2.LINE_AA)
    draw_logo_mark(canvas, _s(54), panel_y + bar_h // 2, _s(56))

    # Status dot + label, left-aligned after the mark — the sole verdict now.
    cv2.circle(canvas, (_s(108), panel_y + bar_h // 2), _s(7), label_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label.strip(), (_s(126), panel_y + bar_h // 2 + _s(9)),
                FONT, 1.0 * S, label_color, _th(2), cv2.LINE_AA)

    if reg_progress:
        n, total = reg_progress
        count_txt = f"{n}/{total}"
        (ctw, _), _ = cv2.getTextSize(count_txt, FONT, 0.5 * S, _th(1))
        right_pad = _s(30) + ctw          # room for the count after the bar
        bar_x = _s(620)
        bar_w = W - bar_x - right_pad
        bar_y = panel_y + bar_h // 2 - _s(6)
        bh = _s(12)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bh),
                      INDIGO_LIFT, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bh),
                      VIOLET_LO, _th(1), cv2.LINE_AA)
        fill = int(bar_w * n / total) if total else 0
        if fill > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + bh),
                          VIOLET, -1, cv2.LINE_AA)
        cv2.putText(canvas, count_txt, (bar_x + bar_w + _s(10), bar_y + bh),
                    FONT, 0.5 * S, MIST, _th(1), cv2.LINE_AA)


def draw_qr_authorized(canvas: np.ndarray, elapsed: float, name: str):
    """Phase 1 (0-5s): QR accepted — violet burst + dragon mark + AUTHORIZED."""
    brand_gradient(canvas)
    draw_wave_field(canvas, elapsed, intensity=0.6)
    cx, cy = W // 2, H // 2 - _s(20)

    # Expanding violet ring burst
    for i in range(4):
        phase = elapsed * 2.2 - i * 0.45
        frac = phase % 1.0
        r = int((_s(70) + _s(150) * frac) * (1.0 - frac * 0.4))
        alpha = max(0.0, 1.0 - frac)
        color = tuple(int(INDIGO[c] + (VIOLET_HI[c] - INDIGO[c]) * alpha) for c in range(3))
        if r > 0:
            cv2.circle(canvas, (cx, cy), r, color, max(1, _th(4 * alpha)), cv2.LINE_AA)

    # Dragon mark bursts to full presence
    pulse = 0.6 + 0.4 * abs(np.sin(elapsed * 5))
    draw_dragon_mark(canvas, cx, cy, int(_s(72) * pulse), VIOLET, glow=True)

    # Green confirmation check, ring'd — the one semantic (unlock) accent
    scale_in = min(1.0, elapsed * 2.5)
    cv2.circle(canvas, (cx, cy), int(_s(72) * pulse) + _s(14), UNLOCK, _th(2), cv2.LINE_AA)

    legibility_halo(canvas, cx, cy + _s(180), _s(460), _s(120), strength=0.55)
    put_centered(canvas, "AUTHORIZED", cy + _s(150), 1.6 * scale_in, INK, 3)
    put_centered(canvas, name, cy + _s(196), 0.72 * scale_in, VIOLET_HI, 1)


def draw_qr_countdown(canvas: np.ndarray, elapsed: float, name: str, countdown: int):
    """Phase 2 (5-10s): get-ready countdown before capture begins."""
    brand_gradient(canvas)
    draw_wave_field(canvas, elapsed + 3.0, intensity=0.6)
    cx, cy = W // 2, H // 2 - _s(20)

    # Depleting violet sweep ring
    frac = max(0.0, min(1.0, 1.0 - elapsed / 5.0))
    cv2.ellipse(canvas, (cx, cy), (_s(118), _s(118)), -90, 0, 360, INDIGO_LIFT, _th(8), cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (_s(118), _s(118)), -90, 0, int(360 * frac), VIOLET, _th(8), cv2.LINE_AA)

    # Big countdown digit
    digit = str(max(0, countdown))
    txt_scale = 4.6 + 0.6 * abs(np.sin(elapsed * 3.14))
    legibility_halo(canvas, cx, cy + _s(20), _s(240), _s(240), strength=0.5)
    put_centered(canvas, digit, cy + _s(42), txt_scale, INK, 6)

    legibility_halo(canvas, cx, cy + _s(185), _s(420), _s(110), strength=0.55)
    put_centered(canvas, "GET READY", cy + _s(168), 0.9, VIOLET_HI, 2)
    put_centered(canvas, f"Enrolling  {name}", cy + _s(206), 0.56, MIST, 1)


def draw_hud(canvas: np.ndarray, state: State, status: str, db_count: int,
             fps: float = 0.0, backend: str = "cpu"):
    hint = "B toggle CPU/NPU   K known   U unknown   R enroll   ESC standby   Q quit"

    # Top bar — state name only; backend/FPS live in the telemetry panel,
    # and the per-state verdict lives in the status strip, so the top stays clean.
    cv2.rectangle(canvas, (0, 0), (W, _s(24)), INDIGO_DEEP, -1)
    cv2.line(canvas, (0, _s(24)), (W, _s(24)), VIOLET_LO, _th(1), cv2.LINE_AA)
    cv2.putText(canvas, f"{state.name:<13}", (_s(10), _s(17)),
                FONT_MONO, 1.1 * S, VIOLET_HI, _th(1), cv2.LINE_AA)

    # Bottom hint bar
    cv2.rectangle(canvas, (0, H - _s(22)), (W, H), INDIGO_DEEP, -1)
    cv2.putText(canvas, hint, (_s(10), H - _s(7)), FONT_MONO, 0.95 * S, VIOLET_LO, _th(1), cv2.LINE_AA)


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
    bar = np.empty((_s(8), w, 3), dtype=np.uint8)
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
    PW, PH = _s(262), _s(212)
    px = W - PW - _s(12)
    py = H - PH - _s(34)           # above the hint bar
    pad = _s(8)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (px - pad, py - pad), (px + PW + pad, py + PH + pad),
                  INDIGO_DEEP, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (px - pad, py - pad), (px + PW + pad, py + PH + pad),
                  VIOLET_LO, _th(1), cv2.LINE_AA)

    y = py + _s(16)
    F = cv2.FONT_HERSHEY_PLAIN
    FS = 1.1 * S

    # ── Backend FPS rows ─────────────────────────────────────────────────────
    for label, fps, is_active, ready in [
        ("CPU", cpu_fps, active_backend == "cpu", True),
        ("NPU", npu_fps, active_backend == "npu", npu_ready),
    ]:
        if is_active:
            cv2.rectangle(canvas, (px - _s(4), y - _s(12)), (px + PW + _s(4), y + _s(6)),
                          INDIGO_LIFT, -1)
            cv2.rectangle(canvas, (px - _s(4), y - _s(12)), (px - _s(2), y + _s(6)), VIOLET, -1)

        if not ready:
            fps_str, col = "loading", VIOLET_LO
        else:
            fps_str = f"{fps:5.1f} FPS"
            col = VIOLET_HI if is_active else MIST

        if is_active:
            tri = np.array([(px, y - _s(8)), (px, y), (px + _s(7), y - _s(4))], np.int32)
            cv2.fillConvexPoly(canvas, tri, VIOLET, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + _s(14), y), F, FS, col, _th(1), cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(fps_str, F, FS, _th(1))
        cv2.putText(canvas, fps_str, (px + PW - sw, y), F, FS, col, _th(1), cv2.LINE_AA)
        y += _s(20)

    y += _s(6)
    cv2.line(canvas, (px, y), (px + PW, y), VIOLET_LO, _th(1), cv2.LINE_AA)
    y += _s(12)

    # ── Per-core CPU bars ─────────────────────────────────────────────────────
    cv2.putText(canvas, f"CPU CORES   avg {cpu_total:.0f}%",
                (px, y), F, 0.95 * S, MIST, _th(1), cv2.LINE_AA)
    y += _s(12)

    cols_n = 4
    gap = _s(4)
    bar_w = (PW - (cols_n - 1) * gap) // cols_n
    for i, pct in enumerate(cpu_cores):
        bx = px + (i % cols_n) * (bar_w + gap)
        by = y + (i // cols_n) * _s(18)
        heat = min(255, int(pct * 2.55))
        bar_col = (0, 255 - heat, heat)       # green→red as load rises
        canvas[by:by + _s(8), bx:bx + bar_w] = _bar(bar_w, pct, bar_col)
        cv2.putText(canvas, f"C{i}", (bx, by + _s(18)), F, 0.75 * S, VIOLET_LO, _th(1), cv2.LINE_AA)

    row_count = (len(cpu_cores) + cols_n - 1) // cols_n
    y += row_count * _s(18) + _s(10)

    cv2.line(canvas, (px, y), (px + PW, y), VIOLET_LO, _th(1), cv2.LINE_AA)
    y += _s(12)

    # ── RAM bar ───────────────────────────────────────────────────────────────
    cv2.putText(canvas, f"RAM   {ram_used_mb:.0f} / {ram_total_mb:.0f} MB   {ram_pct:.0f}%",
                (px, y), F, 0.95 * S, MIST, _th(1), cv2.LINE_AA)
    y += _s(10)
    ram_col = (0, int(255 * (1 - ram_pct / 100)), int(255 * ram_pct / 100))
    canvas[y:y + _s(8), px:px + PW] = _bar(PW, ram_pct, ram_col)


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
    # Fill the HDMI panel: size the window to the native canvas, then force
    # fullscreen so the compositor never leaves a maximized-but-windowed gap.
    cv2.resizeWindow("Dragonwing Doorbell", W, H)
    cv2.moveWindow("Dragonwing Doorbell", 0, 0)
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
