<div align="center">

# Dragonwing Doorbell

**An autonomous face-recognition doorbell running on the Qualcomm® Dragonwing™ IQ-8 (QCS8275).**

Real-time face detection and recognition at the edge — on the CPU *or* the HTP v75 NPU —
with a QR-triggered self-enrollment flow and a live on-screen telemetry overlay.

</div>

---

## What it does

Point a USB webcam at your door. The app sits in **Standby** until someone appears, then
runs InsightFace detection + recognition on every frame:

- **Known face** → green **UNLOCKED** banner.
- **Unknown face** → red **ACCESS DENIED** banner.
- **Show a LinkedIn QR code** → the person is authorized and the app captures ~10 frames to
  **enroll** their face into an in-memory database with a **2-minute TTL** (Time To Live).

Press **B** at any time to switch the inference backend between **CPU** (onnxruntime) and
**NPU** (HTP v75 via `qnn-net-run`) live, and watch the FPS (Frames Per Second) and per-core
CPU load change in the telemetry panel. That side-by-side is the whole point of the demo:
the same recognition pipeline, same models, offloaded to the NPU.

| State | Screen |
|-------|--------|
| Standby | Dragonwing idle screen, awaiting a visitor |
| Detection | Live feed, amber brackets, "RECOGNIZING" |
| Recognized | Green brackets, "UNLOCKED — <name>" |
| Unknown | Red brackets, "ACCESS DENIED" |
| QR authorized | Violet burst + green confirm, then a get-ready countdown |
| Registration | Live feed, violet brackets, capture progress bar |

---

## The three demonstration stages

This repository is the reference for a three-stage demonstration that walks from the maker
ecosystem to a production-style embedded deployment. **Stage 2 is fully documented below**;
Stages 1 and 3 are summarized and will be expanded.

| Stage | Hardware / OS | Camera | Startup | Represents |
|-------|---------------|--------|---------|------------|
| **1** | Arduino VENTUNO Q + **AppLab** | — | AppLab | Maker ecosystem; the start of the journey |
| **2** | **IQ-8275 + Ubuntu 24.04** | **USB webcam** | Manual (`python3 doorbell.py`) | **Beginning of prototyping — this guide** |
| **3** | IQ-8275 + **QLI 2.1** | **MIPI CSI** | Automatic on boot | Advanced, production-shaped prototype |

### Stage 1 — Ventuno + AppLab *(summary)*

The entry point for makers: the Arduino VENTUNO Q running the model through AppLab, Qualcomm's
guided application environment. No Linux setup, no manual model conversion — the goal is to show
the recognition working with the least friction. *Full walkthrough to follow.*

### Stage 3 — IQ-8 + MIPI CSI + QLI 2.1 + autostart *(summary)*

The production-shaped prototype: the same application on **QLI 2.1** (Qualcomm Linux Image),
capturing from a **MIPI CSI** camera instead of USB, and launched automatically on boot via a
systemd service so the board behaves like a finished appliance. *Full walkthrough to follow.*

---

## Stage 2 — full reproduction guide (IQ-8275 + Ubuntu + USB webcam + HDMI)

### 2.0 — What you need

**Hardware**
- Qualcomm Dragonwing IQ-8275 board (QCS8275, HTP v75 NPU)
- USB webcam (UVC class — any standard `/dev/video0` camera)
- HDMI display, keyboard
- Host machine (for flashing + model conversion)

**Accounts / SDKs**
- A [Qualcomm AI Hub](https://aihub.qualcomm.com) account + API token (for model conversion)
- QAIRT 2.46.0 runtime present on the board (ships `qnn-net-run`, `qnn-context-binary-generator`,
  `libQnnHtp.so`)

### 2.1 — Flash Ubuntu 24.04 to the board

Flash the Dragonwing Ubuntu 24.04 desktop image to the board (`.img.xz` plus the provisioning
blobs — `dtb.bin`, `rawprogram0.xml`, the provision zip). Follow the board's flashing procedure
(EDL / `qdl`). After flashing, boot to the Ubuntu desktop on the HDMI display and log in.

> The flash images and provisioning blobs are **not** in this repo (large, board-specific). Use
> the image bundle provided for your board.

### 2.2 — Install system + Python dependencies (on the board)

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv libgl1 v4l-utils

pip3 install --upgrade pip
pip3 install insightface onnxruntime numpy psutil opencv-python
```

| Package | Why |
|---------|-----|
| `opencv-python` / `python3-opencv` | camera capture, QR decode, full-screen canvas UI |
| `insightface` | `buffalo_sc` detection + recognition (CPU backend) |
| `onnxruntime` | runs the ONNX models on CPU |
| `numpy` | preprocessing, embedding math |
| `psutil` | per-core CPU + RAM telemetry panel |
| `v4l-utils` | verify the USB camera (`v4l2-ctl --list-devices`) |

The **NPU** backend additionally needs the QAIRT runtime on the `PATH` (`qnn-net-run`), which the
Dragonwing image provides. The host-side model conversion needs `qai-hub` (`pip3 install qai-hub`).

### 2.3 — Prepare the models

The CPU backend needs only the two ONNX files; the NPU backend needs the compiled HTP context
binaries. The full pipeline — ONNX export → AI Hub W8A16 quantize/compile → `qnn-context-binary-generator`
— is documented in **[models/README.md](models/README.md)**. In short:

```bash
# Host: produce the DLC artifacts via Qualcomm AI Hub
export QAI_HUB_TOKEN=your_token_here          # see .env.example
python3 aihub_compile.py                       # -> models/*_qcs8275.dlc

# Board: ONNX for the CPU path
mkdir -p ~/.insightface/models/buffalo_sc
cp det_500m.onnx w600k_mbf.onnx ~/.insightface/models/buffalo_sc/

# Board: HTP context binaries for the NPU path
mkdir -p ~/qnn_context
qnn-context-binary-generator --backend /usr/lib/libQnnHtp.so \
  --binary_file det_500m_qcs8275  --output_dir ~/qnn_context  --model <det.dlc>
qnn-context-binary-generator --backend /usr/lib/libQnnHtp.so \
  --binary_file w600k_mbf_qcs8275 --output_dir ~/qnn_context  --model <rec.dlc>
```

### 2.4 — Run

```bash
# From the board, with the display attached:
python3 doorbell.py                 # CPU backend
python3 doorbell.py --backend npu   # NPU (HTP v75) backend
```

The window opens full-screen on the HDMI display. **Click the window first** so it receives
keyboard input, then use the controls below. Over SSH the app auto-discovers the GNOME Xwayland
session (`DISPLAY=:0`), so you can launch it from a terminal and it appears on the attached display.

### 2.5 — Controls

| Key | Action |
|-----|--------|
| **B** | Toggle CPU ↔ NPU backend live |
| **K** | Force a "known face" result (demo) |
| **U** | Force an "unknown face" result (demo) |
| **R** | Start enrollment manually (fallback for the QR trigger) |
| **ESC** | Return to Standby |
| **Q** | Quit |
| *QR code* | Showing a LinkedIn QR code triggers authorization + enrollment automatically |

---

## Architecture

Single-file application (`doorbell.py`) built around cooperating daemon threads, so capture,
inference, telemetry, and the UI never block each other:

```
CameraThread     V4L2 capture (synthetic frames if no camera), always-latest frame
FaceEngine       CPU backend — InsightFace / onnxruntime, latest-frame-wins
NpuFaceEngine    NPU backend — batches frames, one qnn-net-run call per batch
StatsMonitor     psutil per-core CPU % + RAM, polled every 0.5s
RegistrationDriver  captures N embeddings during enrollment
main loop        state machine + Dragonwing rendering + key handling
```

**State machine:** `LOADING → STANDBY → DETECTION → {RECOGNIZED | UNKNOWN}`, with
`QR_AUTHORIZED → REGISTRATION` branching off on a QR trigger, and a 5-second no-face timeout
returning to Standby.

### Why the NPU engine batches

`qnn-net-run` is a CLI invoked per call, and each call carries a fixed ~270–305 ms cost
(FastRPC setup + model load). Calling it per frame would cap throughput near 3 FPS. Instead
`NpuFaceEngine` accumulates up to **16 frames** (or 0.35 s) and runs the batch in **one**
`qnn-net-run` call, amortizing the fixed cost across the batch and reaching ~15 FPS live.

### CPU vs NPU (measured on QCS8275, `benchmark.py`)

| Model | CPU (onnxruntime) | NPU (HTP v75) | Speedup |
|-------|-------------------|---------------|---------|
| Detection (`det_500m`, 320×320) | ~47 ms | ~11 ms | ~4× |
| Recognition (`w600k_mbf`, 112×112) | ~91 ms | ~5 ms | ~18× |

With the NPU active, overall CPU load drops sharply (the process spends most of its time blocked
waiting on the HTP), versus the CPU backend saturating all cores. The telemetry panel shows this
live.

---

## Repository layout

```
doorbell.py            The application (single file — scp to the board and run)
aihub_compile.py       Host: ONNX -> AI Hub W8A16 quantize + HTP v75 compile -> DLC
benchmark.py           CPU vs NPU latency benchmark
tools/
  npu_fps_test.py      Headless NPU FPS harness
  profile_engines.py   Per-step engine profiler
  render_states.py     Headless render of every UI screen to PNG (visual QA)
models/README.md       Full model-conversion pipeline (dir is otherwise gitignored)
assets/                Dragonwing brand references + sample LinkedIn QR
.env.example           QAI_HUB_TOKEN template
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `libQnnHtp.so: cannot open shared object file` | Use the correct path: `/usr/lib/libQnnHtp.so` (not `/usr/lib/aarch64-linux-gnu/`). Find it with `find /usr -name libQnnHtp.so`. |
| NPU "context binaries missing" at startup | `~/qnn_context/det_500m_qcs8275.bin` / `w600k_mbf_qcs8275.bin` not present — run `qnn-context-binary-generator` (§2.3). |
| Window never appears over SSH | The app sets `DISPLAY=:0` and discovers the GNOME Xwayland auth from `/run/user/1000/.mutter-Xwaylandauth.*`. Make sure you're logged into the desktop session on the board. |
| No camera / black feed | Check `v4l2-ctl --list-devices`; the app falls back to synthetic frames when `/dev/video0` can't be opened. |
| NPU backend seems to use the CPU | The Python QNNExecutionProvider silently falls back to CPU on this board — that's why the NPU path uses `qnn-net-run`, not ORT. If FPS and CPU load look identical to CPU mode, confirm you launched with `--backend npu` and the context binaries exist. |
| `QAI_HUB_TOKEN is not set` | Export your AI Hub token before `aihub_compile.py` (see `.env.example`), or run `qai-hub configure --api_token ...` once. |
| Keyboard keys do nothing | Click the full-screen window first so OpenCV's highgui receives focus. |

---

## Security

- The AI Hub token is read from the **`QAI_HUB_TOKEN`** environment variable — never hardcoded.
  Copy `.env.example` to `.env` (gitignored) or export it in your shell.
- No board credentials are stored in this repository.
- The face database is **in-memory only** with a 2-minute TTL; nothing is persisted to disk.

---

<div align="center">
<sub>Qualcomm, Dragonwing, and Snapdragon are trademarks of Qualcomm Incorporated. This is a
technical demonstration, not an affiliated product.</sub>
</div>
