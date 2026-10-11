"""
profile_engines.py — headless per-step profiler for CPU vs NPU engines.
Run on the IQ8:  python3 profile_engines.py
"""
import numpy as np
import time
import threading
import os
import sys

import psutil

ONNX_DIR = os.path.expanduser("~/.insightface/models/buffalo_sc")
CTX_DIR  = os.path.expanduser("~/qnn_context")
N_WARMUP = 5
N_RUNS   = 30

# ── CPU sampling helper ───────────────────────────────────────────────────────

class CpuSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self._samples = []
        self._lock = threading.Lock()
        self._running = False
        self.interval = interval

    def start_sampling(self):
        self._samples = []
        self._running = True
        self.start()

    def stop_sampling(self):
        self._running = False
        self.join(timeout=2)
        with self._lock:
            return list(self._samples)

    def run(self):
        psutil.cpu_percent(percpu=True)  # prime
        while self._running:
            pcts = psutil.cpu_percent(percpu=True)
            with self._lock:
                self._samples.append(pcts)
            time.sleep(self.interval)


def avg_cpu(samples):
    if not samples:
        return [], 0.0
    cores = [sum(s[i] for s in samples) / len(samples) for i in range(len(samples[0]))]
    total = sum(cores) / len(cores)
    return cores, total


# ── Timing helpers ────────────────────────────────────────────────────────────

def timeit(fn, n=N_RUNS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return result, times


def stats(times):
    a = np.array(times)
    return a.min(), np.median(a), a.mean(), a.max()


# ── Dummy frames ─────────────────────────────────────────────────────────────

FRAME_BGR = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
DET_IN    = np.random.rand(1, 3, 320, 320).astype(np.float32)
REC_IN    = np.random.rand(1, 3, 112, 112).astype(np.float32)


# ── Step definitions ─────────────────────────────────────────────────────────

def load_ort_sessions():
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_pkg

    htp_lib = qnn_pkg.get_qnn_htp_path()
    det_ctx = os.path.join(CTX_DIR, "det_500m_qcs8275.bin")
    rec_ctx = os.path.join(CTX_DIR, "w600k_mbf_qcs8275.bin")
    has_ctx = os.path.exists(det_ctx) and os.path.exists(rec_ctx)

    cpu_det = ort.InferenceSession(
        os.path.join(ONNX_DIR, "det_500m.onnx"),
        providers=["CPUExecutionProvider"])
    cpu_rec = ort.InferenceSession(
        os.path.join(ONNX_DIR, "w600k_mbf.onnx"),
        providers=["CPUExecutionProvider"])

    def qnn_opts(ctx=None):
        o = {"backend_path": htp_lib, "htp_performance_mode": "burst",
             "enable_htp_fp16_precision": "0"}
        if ctx:
            o["qnn_context_cache_enable"] = "1"
            o["qnn_context_cache_path"] = ctx
        return o

    npu_det = ort.InferenceSession(
        os.path.join(ONNX_DIR, "det_500m.onnx"),
        providers=[("QNNExecutionProvider", qnn_opts(det_ctx if has_ctx else None)),
                   "CPUExecutionProvider"])
    npu_rec = ort.InferenceSession(
        os.path.join(ONNX_DIR, "w600k_mbf.onnx"),
        providers=[("QNNExecutionProvider", qnn_opts(rec_ctx if has_ctx else None)),
                   "CPUExecutionProvider"])

    print(f"  ctx_bin: {'YES' if has_ctx else 'NO'}")
    print(f"  cpu_det providers: {cpu_det.get_providers()}")
    print(f"  npu_det providers: {npu_det.get_providers()}")
    return cpu_det, cpu_rec, npu_det, npu_rec


# ── Pre/post processing ───────────────────────────────────────────────────────

import cv2

_DET_STRIDES    = [8, 16, 32]
_DET_FEAT_MAPS  = [40, 20, 10]
_DET_NUM_ANCHORS = 2


def det_preprocess(bgr):
    img = cv2.resize(bgr, (320, 320))
    img = img[:, :, ::-1].astype(np.float32)
    img = (img - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]


def det_postprocess(outputs, orig_h=480, orig_w=640, det_thresh=0.5, nms_thresh=0.4):
    scores_list = outputs[0:3]
    boxes_list  = outputs[3:6]
    kps_list    = outputs[6:9]
    sx, sy = orig_w / 320.0, orig_h / 320.0
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
        sc  = scores[mask]; bx = boxes[mask]; kp = kps[mask]
        acx = cx[mask];     acy = cy[mask]
        x1 = (acx - bx[:, 0] * stride) * sx
        y1 = (acy - bx[:, 1] * stride) * sy
        x2 = (acx + bx[:, 2] * stride) * sx
        y2 = (acy + bx[:, 3] * stride) * sy
        kp_dec = np.zeros_like(kp)
        kp_dec[:, :, 0] = (acx[:, None] + kp[:, :, 0] * stride) * sx
        kp_dec[:, :, 1] = (acy[:, None] + kp[:, :, 1] * stride) * sy
        boxes_out.append(np.stack([x1, y1, x2, y2], axis=1))
        scores_out.append(sc); kps_out.append(kp_dec)
    if not boxes_out:
        return []
    boxes_all  = np.concatenate(boxes_out)
    scores_all = np.concatenate(scores_out)
    keep = cv2.dnn.NMSBoxes(boxes_all.tolist(), scores_all.tolist(), det_thresh, nms_thresh)
    return [] if len(keep) == 0 else [boxes_all[k] for k in keep.flatten()]


def rec_preprocess(bgr):
    img = cv2.resize(bgr, (112, 112))
    img = img[:, :, ::-1].astype(np.float32)
    img = (img - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]


# ── Main profiler ─────────────────────────────────────────────────────────────

def bench_step(label, fn, sampler=None):
    # warmup
    for _ in range(N_WARMUP):
        fn()
    if sampler:
        sampler._samples = []
    _, times = timeit(fn, N_RUNS)
    samples = sampler._samples[:] if sampler else []
    mn, med, avg, mx = stats(times)
    cpu_cores, cpu_tot = avg_cpu(samples)
    core_str = " ".join(f"{c:.0f}" for c in cpu_cores) if cpu_cores else "n/a"
    print(f"  {label:<30}  min={mn:6.1f}  med={med:6.1f}  avg={avg:6.1f}  max={mx:6.1f} ms"
          f"  cpu={cpu_tot:4.0f}%  [{core_str}]")
    return avg


def main():
    print("=" * 80)
    print("  InsightFace per-step profiler  (CPU vs NPU)")
    print("=" * 80)

    print("\nLoading ORT sessions...")
    cpu_det, cpu_rec, npu_det, npu_rec = load_ort_sessions()

    sampler = CpuSampler(interval=0.02)
    sampler.start_sampling()

    print(f"\n{'─'*80}")
    print("  Pre/post-processing steps (pure Python/NumPy — same for both backends)")
    print(f"{'─'*80}")
    bench_step("det_preprocess(frame)",      lambda: det_preprocess(FRAME_BGR), sampler)
    bench_step("rec_preprocess(crop 100x100)",
               lambda: rec_preprocess(FRAME_BGR[:100, :100]), sampler)
    dummy_det_out = [np.zeros((3200,1),np.float32), np.zeros((800,1),np.float32),
                     np.zeros((200,1),np.float32),  np.zeros((3200,4),np.float32),
                     np.zeros((800,4),np.float32),  np.zeros((200,4),np.float32),
                     np.zeros((3200,10),np.float32), np.zeros((800,10),np.float32),
                     np.zeros((200,10),np.float32)]
    bench_step("det_postprocess(9 tensors)",
               lambda: det_postprocess(dummy_det_out), sampler)

    print(f"\n{'─'*80}")
    print("  Inference — CPU (CPUExecutionProvider / InsightFace ORT)")
    print(f"{'─'*80}")
    cpu_det_ms = bench_step("cpu det_sess.run(320x320)",
                             lambda: cpu_det.run(None, {"input.1": DET_IN}), sampler)
    cpu_rec_ms = bench_step("cpu rec_sess.run(112x112)",
                             lambda: cpu_rec.run(None, {"input.1": REC_IN}), sampler)

    print(f"\n{'─'*80}")
    print("  Inference — NPU (QNNExecutionProvider / HTP v75)")
    print(f"{'─'*80}")
    npu_det_ms = bench_step("npu det_sess.run(320x320)",
                             lambda: npu_det.run(None, {"input.1": DET_IN}), sampler)
    npu_rec_ms = bench_step("npu rec_sess.run(112x112)",
                             lambda: npu_rec.run(None, {"input.1": REC_IN}), sampler)

    print(f"\n{'─'*80}")
    print("  Full pipeline (preproc + infer + postproc) — 1 face assumed")
    print(f"{'─'*80}")

    def cpu_pipeline():
        inp = det_preprocess(FRAME_BGR)
        outs = cpu_det.run(None, {"input.1": inp})
        det_postprocess(outs)
        crop_in = rec_preprocess(FRAME_BGR[:100, :100])
        cpu_rec.run(None, {"input.1": crop_in})

    def npu_pipeline():
        inp = det_preprocess(FRAME_BGR)
        outs = npu_det.run(None, {"input.1": inp})
        det_postprocess(outs)
        crop_in = rec_preprocess(FRAME_BGR[:100, :100])
        npu_rec.run(None, {"input.1": crop_in})

    cpu_total = bench_step("CPU full pipeline", cpu_pipeline, sampler)
    npu_total = bench_step("NPU full pipeline", npu_pipeline, sampler)

    sampler.stop_sampling()

    print(f"\n{'═'*80}")
    print("  SUMMARY")
    print(f"{'═'*80}")
    print(f"  {'Step':<32} {'CPU ms':>8} {'NPU ms':>8} {'Speedup':>9}")
    print(f"  {'-'*60}")
    print(f"  {'det inference':<32} {cpu_det_ms:>8.1f} {npu_det_ms:>8.1f} {cpu_det_ms/npu_det_ms:>8.1f}x")
    print(f"  {'rec inference':<32} {cpu_rec_ms:>8.1f} {npu_rec_ms:>8.1f} {cpu_rec_ms/npu_rec_ms:>8.1f}x")
    print(f"  {'full pipeline (1 face)':<32} {cpu_total:>8.1f} {npu_total:>8.1f} {cpu_total/npu_total:>8.1f}x")
    print(f"\n  CPU FPS: {1000/cpu_total:.1f}   NPU FPS: {1000/npu_total:.1f}")
    print()
    print("  NOTE: if NPU inference is fast but NPU full pipeline is slow,")
    print("  the bottleneck is pre/post-processing (Python/NumPy on CPU cores).")
    print("  If cpu= is high even during NPU runs, the QNNExecutionProvider")
    print("  is doing dequant/copy work on the CPU before/after HTP dispatch.")


if __name__ == "__main__":
    main()
