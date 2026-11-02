# Pre-built NPU (HTP) context binaries — committed for one-command reproduce

These `.bin` files are the **HTP (Hexagon Tensor Processor) context binaries** for
the doorbell's NPU (Neural Processing Unit) backend.  Two SoCs are covered so the
same repo works on both supported boards — `doorbell.py` auto-selects the right pair
by reading `/sys/firmware/devicetree/base/model` at startup.

| File | Bytes | SoC | HTP | InsightFace model | Input shape |
|------|-------|-----|-----|-------------------|-------------|
| `det_500m_qcs8275.bin`  | 1,103,832 | QCS8275 (IQ-8275) | v75 | buffalo_sc detector (`det_500m`)   | 1×3×320×320 |
| `w600k_mbf_qcs8275.bin` | 3,936,520 | QCS8275 (IQ-8275) | v75 | buffalo_sc recognizer (`w600k_mbf`) | 1×3×112×112 |
| `det_500m_qcs9075.bin`  | 1,107,928 | QCS9075 (IQ-9075) | v73 | buffalo_sc detector (`det_500m`)   | 1×3×320×320 |
| `w600k_mbf_qcs9075.bin` | 3,927,648 | QCS9075 (IQ-9075) | v73 | buffalo_sc recognizer (`w600k_mbf`) | 1×3×112×112 |

These are **generated artifacts tied to their specific SoC and QAIRT (Qualcomm AI Runtime SDK) version** —
they are *not* portable across HTP versions, and the on-device DSP firmware's QAIRT
must match the `qai-appbuilder` version the image pins.  They are committed only
because they are small (~10 MB total), slow to regenerate (AI Hub round-trip), and
SoC-pinned anyway.

## How they were generated

Via Qualcomm AI Hub (`qai_hub`) from the buffalo_sc ONNX models, using
`aihub_compile_qcs9075.py` (IQ-9075) / `aihub_compile.py` (IQ-8275):

1. **Freeze static shapes locally** — `onnx.shape_inference.infer_shapes()` with the
   input dim set to `(1, 3, 320×320)` or `(1, 3, 112×112)`.  The hub's
   `--target_runtime onnx` shape-fix step corrupts the detector's multi-scale anchor
   head shapes (Reshape_160/185/210 mismatch), so we do this step locally and upload
   the pre-frozen ONNX.
2. `submit_quantize_job(..., weights_dtype=INT8, activations_dtype=INT16)` → W8A16
   (8-bit Weights 16-bit Activations).
3. `submit_compile_job(..., options="--target_runtime precompiled_qnn_onnx")` targeting
   the matching device (`"Arduino VENTUNO Q"` for v75, `"Dragonwing IQ-9075 EVK"` for v73).
   This returns a zip containing `model.onnx` (thin wrapper) **and `model.bin`** — the
   raw QNN (Qualcomm Neural Network) context binary that `qai-appbuilder`'s `QNNContext`
   loads directly.  Extract `model.bin` → rename to the names above.

The generator scripts are committed at the repo root (`aihub_compile*.py`).  They
require an AI Hub token (a secret — set via `QAI_HUB_TOKEN` env var, never committed).
The CPU (Central Processing Unit) ONNX models are **not** committed: InsightFace
auto-downloads `buffalo_sc` into `~/.insightface` on first use.

## SoC auto-selection in doorbell.py

`doorbell.py` calls `_detect_soc()` at import time, which reads
`/sys/firmware/devicetree/base/model` and maps the device-tree string to the SoC tag
(`qcs8275` or `qcs9075`).  Falls back to `qcs8275` on any read error so existing
single-SoC setups are unaffected.

## Regenerating for a new HTP version

Adapt `aihub_compile_qcs9075.py`: change `DEVICE` to the target device name from
`hub.get_devices()`, rename output files with a new SoC tag, and add a new branch in
`_detect_soc()`.  Verify the DSP (Digital Signal Processor) firmware's QAIRT version
(`strings .../dsp/cdsp/libQnnHtp*.so | grep v2.`) matches the pinned `qai-appbuilder`
— a mismatch makes the HTP session silently fall back to CPU
(`openSessionForPriority ... status=0x72`).
