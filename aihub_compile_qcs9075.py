"""
Submit InsightFace buffalo_sc models to Qualcomm AI Hub for
W8A16 quantization + HTP compilation targeting Dragonwing IQ-9075 EVK (QCS9075 / HTP v73).

Produces:
  qnn_context/det_500m_qcs9075.bin
  qnn_context/w600k_mbf_qcs9075.bin

These are the HTP v73 context binaries for qai-appbuilder on the IQ-9075.
They are the v73 equivalent of det_500m_qcs8275.bin / w600k_mbf_qcs8275.bin (v75).

Run: QAI_HUB_TOKEN=... python3 aihub_compile_qcs9075.py
"""
import os
import zipfile
import tempfile
import numpy as np
import qai_hub as hub

TOKEN  = os.environ.get("QAI_HUB_TOKEN")
DEVICE = "Dragonwing IQ-9075 EVK"
MODELS = "models"
OUT    = "qnn_context"

if TOKEN:
    hub.set_session_token(TOKEN)
elif not os.path.exists(os.path.expanduser("~/.qai_hub/client.ini")):
    raise SystemExit(
        "QAI_HUB_TOKEN is not set and no ~/.qai_hub/client.ini was found.\n"
        "Set your Qualcomm AI Hub token first — see .env.example:\n"
        "  export QAI_HUB_TOKEN=your_token_here\n"
        "Get a token at https://aihub.qualcomm.com (Account -> Settings)."
    )
device = hub.Device(DEVICE)

# ── Load calibration data ─────────────────────────────────────────────────────
cal_det = np.load(f"{MODELS}/cal_det.npy")   # (60, 3, 320, 320)
cal_rec = np.load(f"{MODELS}/cal_rec.npy")   # (60, 3, 112, 112)
print(f"Calibration: det={cal_det.shape}  rec={cal_rec.shape}")

# ── Prepare locally shape-frozen det ONNX ────────────────────────────────────
# The hub's --target_runtime onnx shape-fix corrupts the multi-scale anchor head
# shapes in det_500m (Reshape_160/185/210 dimension mismatch). Fix locally with
# onnx.shape_inference instead, which correctly propagates all intermediate dims.
import onnx
from onnx import shape_inference as _si
_det_static_path = f"{MODELS}/det_500m_static.onnx"
_m = onnx.load(f"{MODELS}/det_500m.onnx")
for _inp in _m.graph.input:
    if _inp.name == "input.1":
        for _i, _v in enumerate([1, 3, 320, 320]):
            _d = _inp.type.tensor_type.shape.dim[_i]
            _d.dim_value = _v
            _d.ClearField("dim_param")
_m = _si.infer_shapes(_m, check_type=True, strict_mode=False)
onnx.save(_m, _det_static_path)
print(f"det static ONNX written: {_det_static_path}")

# ── Upload models ─────────────────────────────────────────────────────────────
print("\nUploading models...")
det_model = hub.upload_model(_det_static_path)   # static, no hub shape-fix needed
rec_model = hub.get_model("mn7lle2oj")           # w600k_mbf.onnx already uploaded
print(f"  det: {det_model}")
print(f"  rec: {rec_model}")

# ── Step 1: Shape-fix rec (already done; reuse result) ────────────────────────
# Rec shape-fix succeeded in job jp11j192p → model mng11wzo6.
print("\nReusing rec shape-fixed model from previous run (job jp11j192p)...")
rec_static = hub.get_model("mng11wzo6")
print(f"  rec static: {rec_static}")

# ── Step 2: Quantize W8A16 ────────────────────────────────────────────────────
# det: quantize directly from original ONNX with input_specs (no shape-freeze)
# rec: reuse already-successful quantize result mm6vve79v (job jgjy7yjxg)
print("\nSubmitting det quantize job (W8A16, direct from original ONNX, no shape-fix)...")
det_quant_job = hub.submit_quantize_job(
    model=det_model,
    calibration_data={"input.1": [cal_det[i:i+1] for i in range(len(cal_det))]},
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT16,
    name="insightface-det-w8a16-direct-qcs9075",
)
print(f"  det quantize job: {det_quant_job}")
print("Reusing rec quantize result from previous run (job jgjy7yjxg)...")
rec_quantized = hub.get_model("mm6vve79v")
print(f"  rec quantized: {rec_quantized}")
print("\nWaiting for det quantize job...")
det_quant_job.wait()
print(f"  det quantize: {det_quant_job.get_status().code}")
if det_quant_job.get_status().code != "SUCCESS":
    raise RuntimeError(f"det quantize failed: {det_quant_job.get_status().message}")
det_quantized = det_quant_job.get_target_model()

# ── Step 3: Compile for HTP v73 — precompiled_qnn_onnx bundles model.bin ─────
print("\nSubmitting compile jobs (HTP v73)...")
det_compile_job = hub.submit_compile_job(
    model=det_quantized, device=device,
    input_specs={"input.1": ((1, 3, 320, 320), "float32")},
    options="--target_runtime precompiled_qnn_onnx",
    name="insightface-det-qcs9075",
)
rec_compile_job = hub.submit_compile_job(
    model=rec_quantized, device=device,
    input_specs={"input.1": ((1, 3, 112, 112), "float32")},
    options="--target_runtime precompiled_qnn_onnx",
    name="insightface-rec-qcs9075",
)
print(f"  det compile job: {det_compile_job}")
print(f"  rec compile job: {rec_compile_job}")
print("\nWaiting for compile jobs...")
det_compile_job.wait(); rec_compile_job.wait()
print(f"  det compile: {det_compile_job.get_status().code}")
print(f"  rec compile: {rec_compile_job.get_status().code}")

# ── Step 4: Download zip, extract model.bin ───────────────────────────────────
# precompiled_qnn_onnx produces a zip containing model.onnx (thin wrapper) +
# model.bin (the raw QNN context binary qai-appbuilder loads directly).
print("\nDownloading and extracting .bin context binaries...")
os.makedirs(OUT, exist_ok=True)

def extract_bin(compile_job, out_path: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    # download() appends the model-format extension (e.g. ".onnx.zip") and
    # returns the actual on-disk path — use that, not tmp_path.
    actual_path = compile_job.get_target_model().download(tmp_path)
    if actual_path is None:
        actual_path = tmp_path  # older SDK: no return value, suffix already correct
    try:
        with zipfile.ZipFile(actual_path) as zf:
            bin_names = [n for n in zf.namelist() if n.endswith(".bin")]
            if not bin_names:
                raise RuntimeError(f"No .bin found in zip contents: {zf.namelist()}")
            with zf.open(bin_names[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    finally:
        for p in {tmp_path, actual_path}:
            if os.path.exists(p):
                os.remove(p)
    print(f"  -> {out_path}  ({os.path.getsize(out_path):,} bytes)")

extract_bin(det_compile_job, f"{OUT}/det_500m_qcs9075.bin")
extract_bin(rec_compile_job, f"{OUT}/w600k_mbf_qcs9075.bin")

print(f"\nDone.")
print(f"  {OUT}/det_500m_qcs9075.bin")
print(f"  {OUT}/w600k_mbf_qcs9075.bin")
print(f"\nNext: commit both .bin files and push — doorbell.py auto-detects the SoC.")
