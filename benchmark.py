"""
benchmark.py — CPU vs NPU (HTP) latency comparison for InsightFace buffalo_sc.
Run on the IQ8 board:
    python3 benchmark.py

Reports per-inference latency (ms) and FPS for:
  - Det model  (buffalo_sc det_500m, 320x320)
  - Rec model  (buffalo_sc w600k_mbf, 112x112)
  - Combined   (both in sequence, as the app uses them)

CPU path:  onnxruntime CPUExecutionProvider
NPU path:  qnn-net-run with HTP context binary, --num_inferences 50
           uses --perf_profile burst for peak HTP performance
"""

import subprocess
import tempfile
import numpy as np
import time
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────

ONNX_DIR   = os.path.expanduser("~/.insightface/models/buffalo_sc")
CTX_DIR    = os.path.expanduser("~/qnn_context")
QNN_LIBS   = "/usr/lib/aarch64-linux-gnu"  # libQnnHtp.so location

DET_ONNX   = f"{ONNX_DIR}/det_500m.onnx"
REC_ONNX   = f"{ONNX_DIR}/w600k_mbf.onnx"
DET_CTX    = f"{CTX_DIR}/det_500m_qcs8275.bin"
REC_CTX    = f"{CTX_DIR}/w600k_mbf_qcs8275.bin"

N_WARMUP = 5
N_RUNS   = 50

# ── CPU benchmark ─────────────────────────────────────────────────────────────

def bench_cpu():
    import onnxruntime as ort
    print("\n[CPU] Loading ONNX models with CPUExecutionProvider...")

    det_sess = ort.InferenceSession(DET_ONNX, providers=['CPUExecutionProvider'])
    rec_sess = ort.InferenceSession(REC_ONNX, providers=['CPUExecutionProvider'])

    det_input = det_sess.get_inputs()[0].name
    rec_input = rec_sess.get_inputs()[0].name

    det_dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)
    rec_dummy = np.random.rand(1, 3, 112, 112).astype(np.float32)

    # warmup
    for _ in range(N_WARMUP):
        det_sess.run(None, {det_input: det_dummy})
        rec_sess.run(None, {rec_input: rec_dummy})

    # det
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        det_sess.run(None, {det_input: det_dummy})
    det_ms = (time.perf_counter() - t0) / N_RUNS * 1000

    # rec
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        rec_sess.run(None, {rec_input: rec_dummy})
    rec_ms = (time.perf_counter() - t0) / N_RUNS * 1000

    combined_ms = det_ms + rec_ms
    print(f"  Det  : {det_ms:6.1f} ms  ({1000/det_ms:5.1f} FPS)")
    print(f"  Rec  : {rec_ms:6.1f} ms  ({1000/rec_ms:5.1f} FPS)")
    print(f"  Both : {combined_ms:6.1f} ms  ({1000/combined_ms:5.1f} FPS)  ← app-level")
    return det_ms, rec_ms, combined_ms


# ── NPU benchmark via qnn-net-run ─────────────────────────────────────────────

def _write_input_list(raw_path: str, tmpdir: str) -> str:
    """Write an input_list.txt pointing to raw_path."""
    list_path = os.path.join(tmpdir, "input_list.txt")
    with open(list_path, "w") as f:
        f.write(raw_path + "\n")
    return list_path


def _save_raw(arr: np.ndarray, path: str):
    arr.astype(np.float32).tofile(path)


def _run_qnn_net_run(ctx_bin: str, input_list: str, out_dir: str, n: int) -> float:
    """
    Run qnn-net-run N times with the HTP backend.
    Returns average inference time in ms (from qnn-net-run's own timer).
    """
    lib_htp = f"{QNN_LIBS}/libQnnHtp.so"
    if not os.path.exists(lib_htp):
        # try QAIRT path
        lib_htp = subprocess.check_output(
            "find /opt /usr -name 'libQnnHtp.so' 2>/dev/null | head -1",
            shell=True).decode().strip()
    if not lib_htp:
        raise RuntimeError("libQnnHtp.so not found")

    cmd = [
        "qnn-net-run",
        f"--backend={lib_htp}",
        f"--retrieve_context={ctx_bin}",
        f"--input_list={input_list}",
        f"--output_dir={out_dir}",
        f"--num_inferences={n}",
        "--perf_profile=burst",
        "--profiling_level=basic",
        "--log_level=error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr

    # qnn-net-run prints: "Avg. Inference Time: X.XX ms" or similar
    import re
    # Look for inference time in output (try multiple patterns)
    patterns = [
        r'[Aa]vg\.?\s+[Ii]nference\s+[Tt]ime[:\s]+([0-9.]+)',
        r'[Ii]nference\s+[Tt]ime[:\s]+([0-9.]+)',
        r'Execute\s+[Tt]ime[:\s]+([0-9.]+)',
        r'Total\s+[Tt]ime[:\s]+([0-9.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, output)
        if m:
            return float(m.group(1))

    # Fallback: time the whole subprocess call and divide by n
    # (includes FastRPC overhead but gives a lower bound)
    print(f"  [WARN] Could not parse timing from qnn-net-run output, using wall-clock fallback")
    print(f"  [DBG] qnn-net-run output (last 300 chars): {output[-300:]}")
    return None


def bench_npu_subprocess():
    """
    NPU benchmark using qnn-net-run subprocess per model.
    This measures steady-state HTP latency when the model stays loaded
    for N consecutive inferences (--num_inferences=50).
    """
    print(f"\n[NPU] Running qnn-net-run with --num_inferences={N_RUNS} (burst mode)...")

    for path in [DET_CTX, REC_CTX]:
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found — run qnn-context-binary-generator first")
            return None, None, None

    with tempfile.TemporaryDirectory() as tmpdir:
        # det
        det_raw = os.path.join(tmpdir, "det_input.raw")
        _save_raw(np.random.rand(1, 3, 320, 320).astype(np.float32), det_raw)
        det_list = _write_input_list(det_raw, tmpdir)
        det_out  = os.path.join(tmpdir, "det_out")
        os.makedirs(det_out)

        t0 = time.perf_counter()
        det_ms_qnn = _run_qnn_net_run(DET_CTX, det_list, det_out, N_RUNS)
        det_wall = (time.perf_counter() - t0) / N_RUNS * 1000

        # rec
        rec_raw = os.path.join(tmpdir, "rec_input.raw")
        _save_raw(np.random.rand(1, 3, 112, 112).astype(np.float32), rec_raw)
        rec_list = _write_input_list(rec_raw, tmpdir)
        rec_out  = os.path.join(tmpdir, "rec_out")
        os.makedirs(rec_out)

        t0 = time.perf_counter()
        rec_ms_qnn = _run_qnn_net_run(REC_CTX, rec_list, rec_out, N_RUNS)
        rec_wall = (time.perf_counter() - t0) / N_RUNS * 1000

    # Use internal timer if parsed, else fall back to wall/N
    det_ms = det_ms_qnn if det_ms_qnn else det_wall
    rec_ms = rec_ms_qnn if rec_ms_qnn else rec_wall
    note   = "(qnn internal timer)" if det_ms_qnn else "(wall-clock / N — includes spawn overhead)"

    combined_ms = det_ms + rec_ms
    print(f"  Det  : {det_ms:6.1f} ms  ({1000/det_ms:5.1f} FPS)  {note}")
    print(f"  Rec  : {rec_ms:6.1f} ms  ({1000/rec_ms:5.1f} FPS)")
    print(f"  Both : {combined_ms:6.1f} ms  ({1000/combined_ms:5.1f} FPS)  ← app-level")
    return det_ms, rec_ms, combined_ms


# ── Summary ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  InsightFace buffalo_sc — CPU vs NPU benchmark")
    print(f"  Warmup: {N_WARMUP}  Runs: {N_RUNS}")
    print("=" * 60)

    cpu_det, cpu_rec, cpu_comb = bench_cpu()
    npu_det, npu_rec, npu_comb = bench_npu_subprocess()

    if npu_comb:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"  {'Model':<12} {'CPU (ms)':>10} {'NPU (ms)':>10} {'Speedup':>10}")
        print(f"  {'-'*44}")
        print(f"  {'Det':<12} {cpu_det:>10.1f} {npu_det:>10.1f} {cpu_det/npu_det:>9.1f}x")
        print(f"  {'Rec':<12} {cpu_rec:>10.1f} {npu_rec:>10.1f} {cpu_rec/npu_rec:>9.1f}x")
        print(f"  {'Combined':<12} {cpu_comb:>10.1f} {npu_comb:>10.1f} {cpu_comb/npu_comb:>9.1f}x")
        print(f"  {'(app FPS)':<12} {1000/cpu_comb:>10.1f} {1000/npu_comb:>10.1f}")
        print()
        print("  NOTE: NPU numbers above reflect qnn-net-run keeping the model")
        print("  loaded for N inferences in one call. A persistent C++ daemon")
        print("  (no per-frame spawn) would match these numbers in the live app.")


if __name__ == "__main__":
    main()
