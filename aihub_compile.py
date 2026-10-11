"""
Submit InsightFace buffalo_sc models to Qualcomm AI Hub for
W8A16 quantization + HTP compilation targeting Arduino VENTUNO Q (QCS8275 / HTP v75).

Run: QAI_HUB_TOKEN=... python3 aihub_compile.py
Output: models/det_500m_qcs8275.dlc  models/w600k_mbf_qcs8275.dlc

The AI Hub token is read from the QAI_HUB_TOKEN environment variable so it is
never committed. Get a token from https://aihub.qualcomm.com (Account → Settings),
then export it (see .env.example) before running. If you have already run
`qai-hub configure --api_token ...`, you can leave the fallback below to pick up
the token the SDK stored in ~/.qai_hub/client.ini.
"""
import os
import numpy as np
import qai_hub as hub

TOKEN   = os.environ.get("QAI_HUB_TOKEN")
DEVICE  = "Arduino VENTUNO Q"
MODELS  = "models"

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

# ── Upload models ─────────────────────────────────────────────────────────────
print("\nUploading models...")
det_model = hub.upload_model(f"{MODELS}/det_500m.onnx")
rec_model = hub.upload_model(f"{MODELS}/w600k_mbf.onnx")
print(f"  det: {det_model}")
print(f"  rec: {rec_model}")

# ── Step 1: Freeze dynamic shapes via ONNX compile job ───────────────────────
print("\nFreezing dynamic shapes...")

det_shape_job = hub.submit_compile_job(
    model=det_model,
    device=device,
    input_specs={"input.1": ((1, 3, 320, 320), "float32")},
    options="--target_runtime onnx",
    name="insightface-det-shape-fix",
)
rec_shape_job = hub.submit_compile_job(
    model=rec_model,
    device=device,
    input_specs={"input.1": ((1, 3, 112, 112), "float32")},
    options="--target_runtime onnx",
    name="insightface-rec-shape-fix",
)
print(f"  det shape job: {det_shape_job}")
print(f"  rec shape job: {rec_shape_job}")
det_shape_job.wait()
rec_shape_job.wait()
print(f"  det shape: {det_shape_job.get_status().code}")
print(f"  rec shape: {rec_shape_job.get_status().code}")
det_static = det_shape_job.get_target_model()
rec_static = rec_shape_job.get_target_model()

# ── Submit quantize jobs (W8A16) ──────────────────────────────────────────────
print("\nSubmitting quantize jobs (W8A16)...")

det_quant_job = hub.submit_quantize_job(
    model=det_static,
    calibration_data={"input.1": [cal_det[i:i+1] for i in range(len(cal_det))]},
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT16,
    name="insightface-det-w8a16",
)

rec_quant_job = hub.submit_quantize_job(
    model=rec_static,
    calibration_data={"input.1": [cal_rec[i:i+1] for i in range(len(cal_rec))]},
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT16,
    name="insightface-rec-w8a16",
)

print(f"  det quantize job: {det_quant_job}")
print(f"  rec quantize job: {rec_quant_job}")

# ── Wait for quantize jobs ────────────────────────────────────────────────────
print("\nWaiting for quantize jobs...")
det_quant_job.wait()
rec_quant_job.wait()

det_quant_status = det_quant_job.get_status()
rec_quant_status = rec_quant_job.get_status()
print(f"  det quantize: {det_quant_status}")
print(f"  rec quantize: {rec_quant_status}")

det_quantized = det_quant_job.get_target_model()
rec_quantized = rec_quant_job.get_target_model()

# ── Submit compile jobs ───────────────────────────────────────────────────────
print("\nSubmitting compile jobs (HTP v75)...")

det_compile_job = hub.submit_compile_job(
    model=det_quantized,
    device=device,
    input_specs={"input.1": ((1, 3, 320, 320), "float32")},
    name="insightface-det-qcs8275",
)

rec_compile_job = hub.submit_compile_job(
    model=rec_quantized,
    device=device,
    input_specs={"input.1": ((1, 3, 112, 112), "float32")},
    name="insightface-rec-qcs8275",
)

print(f"  det compile job: {det_compile_job}")
print(f"  rec compile job: {rec_compile_job}")

# ── Wait for compile jobs ─────────────────────────────────────────────────────
print("\nWaiting for compile jobs...")
det_compile_job.wait()
rec_compile_job.wait()

print(f"  det compile: {det_compile_job.get_status()}")
print(f"  rec compile: {rec_compile_job.get_status()}")

# ── Download artifacts ────────────────────────────────────────────────────────
print("\nDownloading compiled DLC artifacts...")
det_artifact = det_compile_job.get_target_model()
rec_artifact = rec_compile_job.get_target_model()

det_path = f"{MODELS}/det_500m_qcs8275.dlc"
rec_path = f"{MODELS}/w600k_mbf_qcs8275.dlc"
det_artifact.download(det_path)
rec_artifact.download(rec_path)

print(f"\nDone.")
print(f"  {det_path}")
print(f"  {rec_path}")
print(f"\nNext: copy to board and run qnn-context-binary-generator")
