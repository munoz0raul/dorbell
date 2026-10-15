# Pre-built NPU (HTP v75) context binaries — committed for one-command reproduce

These two `.bin` files are the **HTP (Hexagon Tensor Processor) v75 context
binaries** for the doorbell's NPU (Neural Processing Unit) backend on the
**Qualcomm Dragonwing IQ-8275 / QCS8275** (HTP v75). They are committed to the
repo (only ~5 MB total) so a fresh `git clone` reproduces the NPU demo with
**zero model staging** — the container wrappers mount this directory into the
container automatically when `~/qnn_context` is not present.

| File | Bytes | InsightFace model | Input shape |
|------|-------|-------------------|-------------|
| `det_500m_qcs8275.bin`  | 1,103,832 | buffalo_sc detector (`det_500m`) | 1×3×320×320 |
| `w600k_mbf_qcs8275.bin` | 3,936,520 | buffalo_sc recognizer (`w600k_mbf`) | 1×3×112×112 |

These are **generated artifacts tied to this specific SoC and QAIRT version** —
they are *not* portable to a different HTP version, and the on-device DSP
firmware's QAIRT must match the `qai-appbuilder` version the image pins
(see `../container/requirements.txt`). They are committed only because they are
small, slow to regenerate (AI Hub round-trip), and SoC-pinned anyway.

## How they were generated

Via Qualcomm AI Hub (`qai_hub`) from the **static-shape** buffalo_sc ONNX models:

1. Freeze each ONNX to a static input shape (`det_500m_320.onnx` at 320×320,
   `w600k_mbf_112.onnx` at 112×112). Static shapes are required — the
   dynamic-shape sources fail quantize with a `ShapeInferenceError` because the
   detector's reshape constants are baked for 640×640.
2. `submit_quantize_job(..., weights_dtype=int8, activations_dtype=int16)` → W8A16.
3. `submit_compile_job(..., options="--target_runtime precompiled_qnn_onnx")`.
   This returns a zip containing `model.onnx` (a thin wrapper) **and `model.bin`**
   — the raw QNN (Qualcomm Neural Network) context binary that `qai-appbuilder`'s
   `QNNContext` loads directly. Extract `model.bin` → rename to the names above.

The generator script lives outside this repo (it needs an AI Hub token, which is
a secret). The CPU (Central Processing Unit) ONNX models are **not** committed:
InsightFace auto-downloads `buffalo_sc` into `~/.insightface` on first use.

## Regenerating for a different HTP version

If you target a different SoC/HTP (e.g. v73 on an IQ-9075), regenerate the
binaries for that device and verify the DSP firmware's QAIRT version
(`strings .../dsp/cdsp/libQnnHtpV75.so | grep v2.`) matches the pinned
`qai-appbuilder` — a mismatch makes the HTP session silently fall back to CPU
(`openSessionForPriority ... status=0x72`). See `../container/qli/README.md`.
