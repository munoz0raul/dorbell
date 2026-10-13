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
  **enroll** their face under their LinkedIn username, into an in-memory database with a TTL
  (Time To Live) that defaults to 2 minutes. *Any* QR code in frame is read, but only a LinkedIn
  profile QR authorizes; the TTL and the idle timeout are both adjustable live from the keyboard.

Press **B** at any time to switch the inference backend between **CPU** (onnxruntime) and
**NPU** (HTP v75, in-process via `qai-appbuilder`) live, and watch the FPS (Frames Per Second) and
per-core CPU load change in the telemetry panel. That side-by-side is the whole point of the demo:
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
  `libQnnHtp.so`) — used to *build* the HTP context binaries (§2.3)
- `qai-appbuilder` (pip) for the NPU runtime — loads those binaries in-process (§2.2.1)

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

# NPU backend only — in-process QNN runtime for the HTP v75 (see 2.2.1):
pip3 install --break-system-packages qai-appbuilder
```

| Package | Why |
|---------|-----|
| `opencv-python` / `python3-opencv` | camera capture, QR decode, full-screen canvas UI |
| `insightface` | `buffalo_sc` detection + recognition (CPU backend) |
| `onnxruntime` | runs the ONNX models on CPU |
| `numpy` | preprocessing, embedding math |
| `psutil` | per-core CPU + RAM telemetry panel |
| `v4l-utils` | verify the USB camera (`v4l2-ctl --list-devices`) |
| `qai-appbuilder` | **NPU backend** — loads the HTP context binaries *in-process* and runs inference without spawning `qnn-net-run` per frame |

#### 2.2.1 — The NPU backend runs in-process (`qai-appbuilder`)

The NPU path loads the two compiled HTP (Hexagon Tensor Processor) context binaries **once** via
[`qai-appbuilder`](https://pypi.org/project/qai-appbuilder/) (`QNNContext`) and then calls
`.Inference()` directly on the HTP v75 — **no per-frame subprocess, no temp-file round-trip.** This is
the engine that makes the on-screen box track the live face in real time (~5 ms/frame instead of a
~0.5 s batch stall — see [Why the NPU engine runs in-process](#why-the-npu-engine-runs-in-process)).

- `qai-appbuilder` ships its **own** matching QNN (Qualcomm Neural Network) runtime libraries and HTP
  skels under its package `libs/`, and points `ADSP_LIBRARY_PATH` at them automatically — you do **not**
  need to set `QNN_SDK_ROOT` or copy skels by hand.
- Ubuntu 24.04 is a PEP 668 "externally-managed" environment, so `--break-system-packages` is required
  to install into the same site-packages the app uses (matching how `insightface`/`onnxruntime` were
  installed above).
- It consumes the **exact same `~/qnn_context/*.bin`** files the old `qnn-net-run` path used — **no model
  re-conversion.** The output is bit-for-bit identical to the CLI path (verified: all detector output
  tensors match exactly), so boxes, the 5 landmarks, and embeddings are unchanged.

The QAIRT (Qualcomm AI Runtime SDK) CLI tools (`qnn-net-run`, `qnn-context-binary-generator`) are still
required on the board **for building** the context binaries in §2.3; they're just no longer on the
runtime hot path. The host-side model conversion needs `qai-hub` (`pip3 install qai-hub`).

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
| **G** | Authorize a **guest** — runs the same flow as a QR scan with a random name, no QR needed |
| **C** | Clear the enrolled-face database |
| **[** / **]** | Decrease / increase the **idle timeout** (seconds of no face before returning to Standby) — 5–60 s in 5 s steps |
| **-** / **=** | Decrease / increase how long **enrolled data is stored** (the face-DB TTL) — 60–600 s in 60 s steps |
| **ESC** | Return to Standby |
| **Q** | Quit |
| *QR code* | **Any** QR code in frame is read, but only a **LinkedIn** profile QR authorizes — enrollment then starts automatically under the LinkedIn username |

The current idle-timeout and data-store values are shown live in the bottom hint bar, so you can
adjust them with `[`/`]` and `-`/`=` and see the effect without leaving the demo.

### 2.6 — Keep the display awake (demo boards)

Ubuntu's GNOME desktop blanks the HDMI output after a few minutes of inactivity, which
interrupts an unattended demo. On this board the Xwayland server has **no DPMS extension**
(`xset -dpms` reports *"Server does not have the DPMS Extension"*), so the screensaver /
`xset` route alone won't keep it on — GNOME's own **idle manager** is what blanks it. Disable
it with `gsettings` (the settings persist in dconf across reboots):

```bash
# Run inside the board's desktop session (or over SSH with the session bus exported):
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus

gsettings set org.gnome.desktop.session idle-delay 0                              # never go idle
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false
gsettings set org.gnome.desktop.screensaver lock-enabled false                    # no lock screen
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power idle-dim false              # no dimming
```

`run_doorbell.sh` re-asserts all of the above on every launch, so a freshly flashed or
reset board is always demo-ready. To re-enable normal blanking later, set `idle-delay` back
to a non-zero value (default is `300`, i.e. 5 minutes).

### 2.7 — Desktop launcher (one-click NPU start)

For an appliance-like demo you can start the doorbell by double-clicking a desktop icon
instead of opening a terminal. Two small files do it — both are in the repo:

- **`run_doorbell.sh`** — the launcher wrapper (sets `DISPLAY`/`XAUTHORITY`, keeps the
  display awake per §2.6, stops any prior instance, then runs `doorbell.py`).
- **`dragonwing-doorbell-npu.desktop`** — the GNOME desktop entry that calls it with
  `--backend npu` and shows the Dragonwing dragon-head mark as its icon.

Install them on the board:

```bash
# Launcher + the icon (the Dragonwing mark, assets/2.png)
cp run_doorbell.sh                 ~/run_doorbell.sh
chmod +x ~/run_doorbell.sh
mkdir -p ~/.local/share/icons
cp assets/2.png                    ~/.local/share/icons/dragonwing-doorbell.png

# Desktop entry
cp dragonwing-doorbell-npu.desktop ~/Desktop/
chmod +x ~/Desktop/dragonwing-doorbell-npu.desktop

# Mark it trusted so GNOME launches it without an "Allow Launching" prompt
gio set ~/Desktop/dragonwing-doorbell-npu.desktop metadata::trusted true
```

Double-click **Dragonwing Doorbell (NPU)** on the desktop to launch on the HTP v75 NPU.
Because the launcher stops any running instance first, clicking it again safely restarts
the demo rather than fighting over the camera. The wrapper runs the CPU path when invoked
as `run_doorbell.sh cpu`; duplicate the `.desktop` file with that argument if you want a
second CPU icon.

---

## Architecture

Single-file application (`doorbell.py`) built around cooperating daemon threads, so capture,
inference, telemetry, and the UI never block each other:

```
CameraThread     V4L2 capture (synthetic frames if no camera), always-latest frame
FaceEngine       CPU backend — InsightFace / onnxruntime, latest-frame-wins
NpuFaceEngine    NPU backend — HTP v75 via qai-appbuilder, in-process, latest-frame-wins
StatsMonitor     psutil per-core CPU % + RAM, polled every 0.5s
RegistrationDriver  captures N embeddings during enrollment
main loop        state machine + Dragonwing rendering + key handling
```

**State machine:** `LOADING → STANDBY → DETECTION → {RECOGNIZED | UNKNOWN}`, with
`QR_AUTHORIZED → REGISTRATION` branching off on a QR trigger, and a no-face timeout (default
5 s, adjustable live with `[`/`]`) returning to Standby.

### Why the NPU engine runs in-process

The HTP (Hexagon Tensor Processor) itself is extremely fast (det ~2 ms, rec ~1.3 ms per frame).
The challenge was *how you invoke it*. The original engine shelled out to the `qnn-net-run` CLI,
and each spawn carries a fixed **~270–305 ms** cost (FastRPC setup + model reload, *every call*).
To stay usable, that engine had to batch 16 frames behind a 0.10 s window and amortize one spawn
across the batch — but the batch window was itself the dominant source of overlay lag, making the
box and landmarks visibly trail a moving face.

`NpuFaceEngine` now loads both context binaries **once** at startup through `qai-appbuilder`
(`QNNContext`) and calls `.Inference()` directly on the HTP — no subprocess, no file round-trip.
With the spawn cost gone, batching has no purpose, so the engine runs **latest-frame-wins**, exactly
like the CPU path. Per-frame latency drops from a ~0.5 s batch stall to **~5 ms**, and the overlay
tracks the live face in real time. The output is **bit-for-bit identical** to the old CLI path
(verified: every detector output tensor matches exactly), so boxes, the 5 landmarks, and recognition
embeddings are unchanged — only the latency is different.

### CPU vs NPU (measured on QCS8275, `benchmark.py`)

| Model | CPU (onnxruntime) | NPU (HTP v75, in-process) | Speedup |
|-------|-------------------|---------------------------|---------|
| Detection (`det_500m`, 320×320) | ~47 ms | ~2 ms | ~20× |
| Recognition (`w600k_mbf`, 112×112) | ~91 ms | ~1.3 ms | ~70× |

The NPU context binaries load once at startup (~0.2–0.3 s total) and then stay resident on the HTP, so
these are the steady-state per-frame costs. Because inference no longer blocks on a subprocess, the
main loop runs at full frame rate on the NPU path; the telemetry panel shows the FPS jump and the
per-core load live when you toggle with **B**.

---

## Repository layout

```
doorbell.py            The application (single file — scp to the board and run)
run_doorbell.sh        Board launcher — sets display env, keeps screen awake, single-instance
dragonwing-doorbell-npu.desktop  GNOME desktop icon that one-click-launches the NPU backend
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
| `No module named 'qai_appbuilder'` at NPU startup | The in-process NPU runtime isn't installed. `pip3 install --break-system-packages qai-appbuilder` (§2.2.1). The CPU backend still runs without it. |
| NPU "context binaries missing" at startup | `~/qnn_context/det_500m_qcs8275.bin` / `w600k_mbf_qcs8275.bin` not present — run `qnn-context-binary-generator` (§2.3). |
| Window never appears over SSH | The app sets `DISPLAY=:0` and discovers the GNOME Xwayland auth from `/run/user/1000/.mutter-Xwaylandauth.*`. Make sure you're logged into the desktop session on the board. |
| No camera / black feed | Check `v4l2-ctl --list-devices`; the app probes `/dev/video0`–`video5` and uses the first that yields a frame, falling back to synthetic frames if none work. (A USB camera that re-enumerates to a new index after replug is handled automatically.) |
| NPU backend seems to use the CPU | The Python QNNExecutionProvider silently falls back to CPU on this board — that's why the NPU path drives the HTP in-process via `qai-appbuilder`, not ORT. If FPS and CPU load look identical to CPU mode, confirm you launched with `--backend npu`, that `qai-appbuilder` imports, and the context binaries exist. |
| NPU mode uses *more* CPU than before | Expected after the in-process switch: the main loop now runs at full frame rate (no 0.5 s batch stall), so per-frame work — QR decode especially — runs ~10× more often. The HTP itself is still doing the inference. |
| `QAI_HUB_TOKEN is not set` | Export your AI Hub token before `aihub_compile.py` (see `.env.example`), or run `qai-hub configure --api_token ...` once. |
| Keyboard keys do nothing | Click the full-screen window first so OpenCV's highgui receives focus. |
| Display blanks / screen turns off mid-demo | GNOME's idle manager blanks the HDMI output (this Xwayland server has no DPMS extension, so `xset -dpms` alone won't help). Disable idle blanking — see **§2.6**. `run_doorbell.sh` already re-asserts these on every launch. |

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
