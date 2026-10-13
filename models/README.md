# Models

This directory is intentionally **code-only** in the repository — the model
artifacts are large and are regenerated from the steps below. Everything here
except this README is gitignored.

The demo uses **InsightFace `buffalo_sc`**, a lightweight pack with two models:

| Role        | Model          | Input          | Output                         |
|-------------|----------------|----------------|--------------------------------|
| Detection   | `det_500m`     | `1×3×320×320`  | 9 tensors (scores/boxes/kps)   |
| Recognition | `w600k_mbf`    | `1×3×112×112`  | `1×512` embedding              |

The CPU backend runs the ONNX files directly through onnxruntime. The NPU
backend runs HTP **context binaries** compiled for QCS8275 / HTP v75.

```
models/
  det_500m.onnx                 ONNX detection model     (source, CPU path)
  w600k_mbf.onnx                ONNX recognition model   (source, CPU path)
  cal_det.npy                   detection calibration    (60, 3, 320, 320)
  cal_rec.npy                   recognition calibration  (60, 3, 112, 112)
  det_500m_qcs8275.dlc          AI Hub compiled DLC      (detection, NPU)
  w600k_mbf_qcs8275.dlc         AI Hub compiled DLC      (recognition, NPU)
```

On the board the NPU context binaries live at:

```
~/qnn_context/det_500m_qcs8275.bin
~/qnn_context/w600k_mbf_qcs8275.bin
```

---

## Step 1 — Obtain the ONNX source models (host)

InsightFace downloads `buffalo_sc` on first use. Grab the two ONNX files it
caches, or export them yourself:

```bash
pip install insightface onnxruntime
python3 - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_sc')          # downloads to ~/.insightface
app.prepare(ctx_id=-1)                          # CPU
PY
cp ~/.insightface/models/buffalo_sc/det_500m.onnx  models/
cp ~/.insightface/models/buffalo_sc/w600k_mbf.onnx models/
```

The CPU backend needs nothing more — copy these two files to the board at
`~/.insightface/models/buffalo_sc/` and `python3 doorbell.py` works.

## Step 2 — Build calibration data (host)

Quantization needs representative inputs. Create 60 preprocessed face crops at
each model's input size and save them as `cal_det.npy` / `cal_rec.npy`
(`float32`, NCHW, scaled `(x-127.5)/128`). Any set of real faces works; more
diverse data gives a better INT8/INT16 fit.

## Step 3 — Quantize + compile on Qualcomm AI Hub (host)

`aihub_compile.py` uploads the ONNX models, freezes their dynamic shapes,
quantizes to **W8A16** (INT8 weights, INT16 activations) using the calibration
data, compiles for **HTP v75**, and downloads the `.dlc` artifacts.

```bash
export QAI_HUB_TOKEN=your_token_here      # see ../.env.example
python3 aihub_compile.py
# -> models/det_500m_qcs8275.dlc
# -> models/w600k_mbf_qcs8275.dlc
```

Get a token at https://aihub.qualcomm.com (Account → Settings). The target
device is `Arduino VENTUNO Q` (QCS8275).

## Step 4 — Generate HTP context binaries (on the board)

The `.dlc` files are converted to self-contained HTP **context binaries** once,
on the board, with the QAIRT tool. A context binary bakes in the graph prepare
so `qnn-net-run` skips recompilation on every launch.

```bash
mkdir -p ~/qnn_context
qnn-context-binary-generator \
  --model     <qnn_model_lib_or_dlc> \
  --backend   /usr/lib/libQnnHtp.so \
  --binary_file det_500m_qcs8275 \
  --output_dir ~/qnn_context
# repeat for w600k_mbf_qcs8275
```

Confirm both binaries exist:

```bash
ls -la ~/qnn_context/*.bin
# det_500m_qcs8275.bin   w600k_mbf_qcs8275.bin
```

The NPU engine in `doorbell.py` loads these context binaries **in-process** via
`qai-appbuilder` (`QNNContext`) and runs them with direct `.Inference()` calls on
the HTP v75 — no `qnn-net-run` subprocess on the runtime path:

```python
from qai_appbuilder import QNNContext, QNNConfig, Runtime, PerfProfile, LogLevel, ProfilingLevel

QNNConfig.Config(Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)   # bundled QNN libs + HTP skels
det = QNNContext("det", "~/qnn_context/det_500m_qcs8275.bin")       # loaded once
PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
outputs = det.Inference([preprocessed_nchw_float32])                # ~2 ms on the HTP
```

Install it with `pip3 install --break-system-packages qai-appbuilder` (it bundles the
matching QNN runtime libraries and HTP skels, so no `QNN_SDK_ROOT` setup is needed).

> The `qnn-net-run` CLI is still used to *build* the context binaries in Step 4 below and is
> handy for standalone verification, e.g.:
>
> ```bash
> qnn-net-run --backend=/usr/lib/libQnnHtp.so \
>             --retrieve_context=~/qnn_context/<model>.bin \
>             --input_list=<batch>.txt --output_dir=<out> \
>             --perf_profile=burst --log_level=error
> ```
>
> The in-process path produces **bit-for-bit identical** outputs to this CLI (verified across all
> detector output tensors).

---

## Measured performance (QCS8275, from `benchmark.py`)

| Model      | CPU (onnxruntime) | NPU (HTP v75, in-process) |
|------------|-------------------|---------------------------|
| Detection  | ~47 ms            | ~2 ms                     |
| Recognition| ~91 ms            | ~1.3 ms                   |

The context binaries load once at startup (~0.2–0.3 s total) and stay resident on the HTP, so these
are steady-state per-frame costs. The engine runs latest-frame-wins (no batching): with inference
in-process there's no per-call spawn to amortize, so the overlay tracks the live face in real time
at full frame rate.
