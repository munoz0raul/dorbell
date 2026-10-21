# Demo walkthrough — the Dragonwing Doorbell, explained

*A script and a technical reference for presenting the three-board demo side by side:
**VENTUNO Q + Ubuntu**, **IQ-8275 + Ubuntu**, and **IQ-8275 + QLI 2.1**.*

This document answers three questions a viewer will ask, in order:

1. **What am I looking at?** (the high-level story — [§1](#1-the-high-level-story))
2. **What is actually being demonstrated, technically?** (the per-board setup — [§2](#2-what-each-board-demonstrates))
3. **What is the AI — how was the model trained, converted, and deployed?** (the full pipeline — [§3](#3-the-full-ai-pipeline))

For the narrative of *why* the journey is staged this way, see [`blog.md`](blog.md). For
reproduction, see [`README.md`](README.md), [`INSTALL.md`](INSTALL.md), and
[`INSTALL-ubuntu.md`](INSTALL-ubuntu.md).

---

## 1. The high-level story

> **The elevator version.** "This is one face-recognition doorbell application, running on three
> different platforms at once. On the left it's a maker board you can buy and program in minutes. In
> the middle it's the same app on the production chip under plain Ubuntu, where we develop it. On the
> right it's that *exact same container* on Qualcomm Linux — the operating system a shipped product
> actually runs. Same AI, same code, three rungs of a ladder from prototype to product. And every
> face you see recognized is being computed on the **NPU**, the dedicated AI accelerator, not the
> CPU — I can prove that live by toggling between them."

Point a USB webcam at each screen. Each one shows the same thing:

- It sits in **Standby** until a face appears.
- When someone steps in, it draws a box around **every** face: **green** with a name for a known
  person, **red** labelled `unknown` for a stranger. A headline banner reads **UNLOCKED** or
  **ACCESS DENIED** based on the closest face.
- Show a **LinkedIn QR code** to the camera and the app enrolls you: a 10-second authorize
  animation, then it captures ~30 frames of your face (turn your head slowly) inside an on-screen
  **"O" ring**, and from then on you show up green with your name. Enrollments live in memory with a
  2-minute TTL (Time To Live) — nothing is written to disk.

The three screens are **identical in behaviour** because they run the **same application code**. The
only differences are underneath, and that's the point of the demo.

### The one-line pitch for each board

| | Board + OS | The pitch |
|---|---|---|
| **Left** | VENTUNO Q + Ubuntu (via App Lab) | "The **maker on-ramp** — prove the idea in minutes, no Linux setup." |
| **Middle** | IQ-8275 + Ubuntu 24.04 | "The **development bench** — production silicon, full control, fast iteration." |
| **Right** | IQ-8275 + QLI 2.1 | "The **product** — the container runs unchanged on the OS you'd actually ship." |

### The two moments that land the demo

1. **Press `B` on the middle or right board.** The whole pipeline swaps between the **CPU** and the
   **NPU** live. Watch the telemetry panel: on CPU, detection takes ~47 ms and the per-core load
   spikes; on the NPU it drops to ~2 ms and the box snaps to the face in real time. *Same models, same
   result, computed on the accelerator.* This is the single most convincing thing to show.
2. **Enroll yourself with a QR on the left board, then do it again on the right.** The flow is
   pixel-for-pixel identical — because `doorbell_core.py` on the Arduino side is a byte-for-byte copy
   of `doorbell.py` on the production side. The audience sees prototype and product behaving as one
   app.

---

## 2. What each board demonstrates

All three run the same `doorbell.py` logic (the Arduino side imports it verbatim as
`doorbell_core.py`). What differs is the **two I/O (Input/Output) ends** — how inference is invoked,
and how the screen is drawn — and the **host plumbing** that reaches the NPU. This is the technical
core of the demo: *the application is constant; the platform integration is the variable.*

### Side-by-side

| Aspect | **Left: VENTUNO Q + Ubuntu** | **Middle: IQ-8275 + Ubuntu** | **Right: IQ-8275 + QLI 2.1** |
|---|---|---|---|
| **Role in the story** | Stage 1 — maker prototype | Stage 2 — dev on production silicon | Stage 3 — shippable product |
| **SoC / NPU** | QCS8300, HTP v75 | QCS8275, HTP v75 | QCS8275, HTP v75 |
| **How it starts** | Arduino **App Lab** "Run" button | `python3 doorbell.py` or a container | container on the managed distro |
| **How the app is packaged** | an App Lab **app** = two bricks: a custom `arduino:face_recognition` + stock `arduino:web_ui` | bare Python, then a Docker image | the **same Docker image**, byte-for-byte |
| **Inference end** | brick → a companion **runner container** doing inference via **`onnxruntime-qnn`** (reads `.onnx`, QNN plugin-EP drives the HTP) | in-process **`qai-appbuilder`** loading pre-compiled **`.bin`** context binaries, `.Inference()` on the HTP | identical to middle (`qai-appbuilder` + `.bin`) |
| **Display end** | canvas **JPEG-streamed into a webview** (`:7000`); keys return over Socket.IO | `cv2.imshow()` full-screen on HDMI/Xwayland | same `cv2.imshow()`, but on **Weston**'s Xwayland |
| **NPU runtime delivery** | wheel bundles QNN 2.51 + v75 skels | wheel bundles QNN; pinned to firmware | wheel pinned to the QLI DSP firmware's QAIRT |
| **How the NPU is reached** | inside the brick's runner container | host libs + CDI (Container Device Interface) on Ubuntu paths | CDI on **flat `/usr/lib`** paths, device nodes passed to preserve host gids |
| **Camera** | USB webcam (App Lab bind-mounts host `/dev`) | USB webcam at `/dev/video0` | USB webcam enumerates **high** (e.g. `/dev/video28`) behind on-SoC ISP nodes |

### Why the two inference ends differ (and why that's honest, not inconsistent)

The left board and the middle/right boards reach the *same* HTP v75 two different ways — and that
difference is itself worth explaining to a technical audience:

- **VENTUNO (App Lab)** uses **`onnxruntime-qnn`**: the runner loads the plain `.onnx` models and
  onnxruntime's QNN plugin-EP (Execution Provider) JIT-compiles the graph for the HTP at startup.
  It's the most *portable* path — hand the runner an ONNX file and it runs — which fits App Lab's
  "drop in an app" model. The trade-off: the compile happens on the device at load time.
- **IQ-8275 (Ubuntu + QLI)** uses **`qai-appbuilder`** with **pre-compiled `.bin` context
  binaries**: the graph is quantized and compiled *ahead of time* on Qualcomm AI Hub into a
  self-contained HTP context binary, which `qai-appbuilder` loads once and runs with direct
  `.Inference()` calls. It's the most *production-shaped* path — the heavy compile is done on a
  server, the device just loads and runs — which is exactly what you want for a shipped appliance.

Same InsightFace models, same HTP, same results. One path optimizes for drop-in portability
(prototype); the other for ahead-of-time, reproducible, fast-loading deployment (product). That
mirrors the prototype→product arc of the whole demo.

### What to say while pointing at each screen

- **Left (VENTUNO):** "This is where a maker starts. The whole app is an Arduino App Lab project — I
  hit Run and it comes up. Under the hood it's our custom face-recognition *brick* talking to an NPU
  runner container. The UI you see is streamed into a browser."
- **Middle (IQ-8275 + Ubuntu):** "Same app, now on the production chip under normal Ubuntu — this is
  our development bench. Here I can press `B` and flip CPU↔NPU live. Watch the latency and CPU load
  change. This is also where we containerize it."
- **Right (IQ-8275 + QLI):** "And here's that *same container* on Qualcomm Linux — the Yocto-based OS
  a real product ships on. Nothing about the app changed; everything about the OS underneath it did.
  This is the rung where you stop using someone else's operating system and start managing your own."

---

## 3. The full AI pipeline

This is the "what did you actually do with the model" section. The short answer: **we did not train
a model from scratch.** We took a strong pretrained face-recognition pack, verified it on our data,
then did the real edge-AI engineering — **quantization, hardware compilation, and in-process
deployment** — to make it run in real time on the HTP v75. That's deliberate and it's the right call
for this problem; the interesting work on an NPU is rarely the training, it's getting a proven model
to run correctly and fast on fixed-point silicon.

### 3.1 The model: InsightFace `buffalo_sc`

The demo uses **InsightFace**'s `buffalo_sc` pack — a lightweight, well-validated pair of models:

| Role | Model | Architecture | Input | Output |
|---|---|---|---|---|
| **Detection** | `det_500m` | SCRFD-family detector | `1×3×320×320` | 9 tensors — face scores, bounding boxes, 5-point landmarks |
| **Recognition** | `w600k_mbf` | MobileFaceNet backbone, **ArcFace** training | `1×3×112×112` | a **512-dimension embedding** |

**How these were trained (by InsightFace, not by us).** `w600k_mbf` is a MobileFaceNet trained with
the **ArcFace** (Additive Angular Margin) loss on the **WebFace600K** dataset (~600K identities).
ArcFace is what makes the 512-d embedding *discriminative*: it pushes same-identity embeddings
together and different-identity embeddings apart on a hypersphere, so a simple **cosine similarity**
between two embeddings is a reliable "same person?" score. The detector `det_500m` is an
SCRFD-style single-shot detector trained to emit boxes plus the 5 facial landmarks. We use these
**pretrained weights as published** — they are the product of large-scale training we'd have no
reason to reproduce.

**Why no custom training / fine-tuning?** Face *recognition* here is not a classification problem we
train for — there are no fixed "classes." Enrollment is **few-shot at runtime**: we capture ~30
frames of a new face, compute their 512-d embeddings, average them into one reference vector, and
store it. Recognizing someone later is a nearest-neighbour cosine-similarity lookup against those
stored vectors. Adding a person is adding a vector, not retraining a network — which is exactly why
the demo can enroll you live in 15 seconds. The model's job is only to produce good embeddings, and
a WebFace600K-trained ArcFace model already does that better than anything we'd fine-tune on a demo
dataset.

### 3.2 Recognition at runtime — how a face becomes a decision

For each camera frame, on both the CPU and NPU paths:

```
frame ──► det_500m ──► boxes + 5 landmarks per face
                         │
                         ▼
             norm_crop (landmark-aligned 112×112)   ← InsightFace 5-point affine align
                         │
                         ▼
            w600k_mbf ──► 512-d embedding (L2-normalized)
                         │
                         ▼
   cosine similarity vs each enrolled reference vector
                         │
         ┌───────────────┴───────────────┐
     max sim ≥ threshold              max sim < threshold
         │                                │
       GREEN  name + score              RED  "unknown"
```

Two details that matter for quality:

- **Landmark-aligned crops (`norm_crop`).** Before recognition, each face is warped to a canonical
  112×112 using its 5 landmarks (eyes, nose, mouth corners). This alignment is applied on **both**
  the CPU and NPU paths, so the embeddings stay comparable and a non-enrolled face scores low and
  stays red instead of drifting toward a false match.
- **Per-face, independent decisions.** Every face in the frame is recognized on its own each frame;
  the **largest (closest)** face decides the headline UNLOCKED/ACCESS-DENIED banner, so a bystander
  in the background can't flip it.

### 3.3 Conversion & quantization — ONNX → W8A16 → HTP context binary

This is the edge-AI engineering. The HTP is a fixed-point accelerator; to run on it, the floating-
point ONNX models must be **quantized** and **compiled** for the specific HTP version. We do this on
**Qualcomm AI Hub** ([`aihub_compile.py`](aihub_compile.py); full steps in
[`models/README.md`](models/README.md)):

1. **Export the ONNX source.** InsightFace caches `det_500m.onnx` and `w600k_mbf.onnx` on first use;
   these are the CPU-path models and the quantization source.
2. **Freeze dynamic shapes.** Both models ship with dynamic input dims; we pin them to static shapes
   (`320×320` for detection, `112×112` for recognition) via an AI Hub ONNX compile job. *Static
   shapes are required* — the dynamic-shape detector fails quantization with a `ShapeInferenceError`
   because its reshape constants are baked for a different resolution.
3. **Build calibration data.** Quantization needs representative inputs to choose fixed-point ranges.
   We preprocess 60 real face crops at each model's input size (`cal_det.npy` `(60,3,320,320)`,
   `cal_rec.npy` `(60,3,112,112)`, `float32` NCHW, scaled `(x-127.5)/128`) and feed them as
   calibration data. More diverse faces give a better INT8/INT16 fit.
4. **Quantize to W8A16.** `submit_quantize_job(..., weights=INT8, activations=INT16)`. We chose
   **W8A16** (8-bit weights, **16-bit activations**) rather than full INT8 deliberately: the wider
   16-bit activations preserve the numerical range the ArcFace embedding needs to stay discriminative,
   while 8-bit weights keep the model small and fast. Full W8A8 measurably degrades embedding quality
   for recognition; W8A16 is the accuracy/throughput sweet spot on this HTP.
5. **Compile for HTP v75.** `submit_compile_job(..., --target_runtime precompiled_qnn_onnx)` returns
   a `model.bin` — the self-contained **QNN context binary** for the QCS8275's HTP v75. We extract
   and rename it to `det_500m_qcs8275.bin` / `w600k_mbf_qcs8275.bin`. These are committed to the repo
   (~5 MB total) so a clone reproduces the NPU demo with zero staging; see
   [`qnn_context/README.md`](qnn_context/README.md).

These `.bin` files are **pinned to this SoC and QAIRT version** — not portable to a different HTP,
and the on-device DSP firmware's QAIRT must match the runtime version the image ships (a mismatch
makes the HTP session silently fall back to CPU). On the VENTUNO/App Lab path we skip the AI Hub
round-trip entirely: `onnxruntime-qnn` compiles the ONNX for the HTP on-device at load time.

### 3.4 Deployment — running it in real time (the in-process trick)

Having a compiled context binary isn't enough; *how you invoke it* decides whether the demo feels
alive. The naïve approach shells out to the `qnn-net-run` CLI per inference, and each spawn costs
**~270–305 ms** of FastRPC setup + model reload. That forced a batching scheme whose latency made the
box visibly trail a moving face.

Instead, `NpuFaceEngine` loads both context binaries **once** at startup through **`qai-appbuilder`**
(`QNNContext`) and calls `.Inference()` directly on the HTP — no subprocess, no file round-trip:

```python
from qai_appbuilder import QNNContext, QNNConfig, Runtime, PerfProfile, ProfilingLevel, LogLevel

QNNConfig.Config(Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)   # bundled QNN libs + HTP v75 skels
det = QNNContext("det", "~/qnn_context/det_500m_qcs8275.bin")       # loaded once, resident on HTP
PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
outputs = det.Inference([preprocessed_nchw_float32])               # ~2 ms on the HTP
```

With the spawn cost gone, batching has no purpose, so the engine runs **latest-frame-wins** exactly
like the CPU path — and the overlay tracks the live face in real time. The in-process output is
**bit-for-bit identical** to the `qnn-net-run` CLI path (verified across all detector output
tensors), so boxes, landmarks, and embeddings are unchanged; only the latency differs. `qai-appbuilder`
bundles its own version-matched QNN runtime and HTP skels, so no `QNN_SDK_ROOT` setup is needed — this
is also why the container stays driver-free and the CDI only forwards kernel-coupled plumbing.

### 3.5 Measured results — the number that sells the demo

Steady-state per-frame cost on the QCS8275 (from [`benchmark.py`](benchmark.py)):

| Model | CPU (onnxruntime) | NPU (HTP v75, in-process) | Speedup |
|---|---|---|---|
| Detection (`det_500m`, 320×320) | ~47 ms | ~2 ms | **~20×** |
| Recognition (`w600k_mbf`, 112×112) | ~91 ms | ~1.3 ms | **~70×** |

The context binaries load once (~0.2–0.3 s total) and stay resident on the HTP. Because inference no
longer blocks on a subprocess, the main loop runs at full frame rate on the NPU path — which is
exactly what the audience sees when you press **`B`** and the box snaps to the face.

---

## 4. Running the three-board demo

| Board | Start it with | Reference |
|---|---|---|
| **VENTUNO Q + Ubuntu** | App Lab **Run** (after `install-brick.sh`) | [dorbell-app-bricks → install-bricks.md](https://github.com/munoz0raul/dorbell-app-bricks/blob/main/install-bricks.md) |
| **IQ-8275 + Ubuntu** | `python3 doorbell.py --backend npu`, or `bash container/run_doorbell_container.sh npu` | [README §2](README.md), [INSTALL-ubuntu.md](INSTALL-ubuntu.md) |
| **IQ-8275 + QLI 2.1** | `bash container/qli/run_doorbell_container_qli.sh npu` | [INSTALL.md](INSTALL.md), [container/qli/README.md](container/qli/README.md) |

**Pre-flight for a live demo:**

- Plug a **USB webcam** into each board before launching (otherwise the app runs on synthetic
  frames — fine for a dry run, no real recognition).
- Have a **LinkedIn QR code** ready on your phone for the enrollment moment.
- On the IQ-8275 boards, confirm the NPU is genuinely in use (CPU fallback is silent): the runner/app
  log should show the HTP providers, and pressing **`B`** should visibly change the telemetry. See
  the NPU-verification notes in each board's install doc.
- Keep the displays awake — `run_doorbell.sh` re-asserts the GNOME idle-disable settings on every
  launch (see [README §2.6](README.md)).

**Controls (every board):** `B` toggle CPU↔NPU (IQ-8275 paths) · `G` authorize a guest · `C` clear
the database · `[` `]` idle timeout · `-` `=` enrolled-face TTL · `Esc` back to Standby · `Q` quit ·
*any LinkedIn QR* authorizes + enrolls.

---

<div align="center">
<sub>Qualcomm, Dragonwing, and Snapdragon are trademarks of Qualcomm Incorporated. InsightFace and
the <code>buffalo_sc</code> models are the work of their respective authors. This is a technical
demonstration, not an affiliated product.</sub>
</div>
