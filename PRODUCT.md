# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

<!-- ACTUAL PLATFORM: embedded Linux desktop application, NOT web. The UI is drawn
by OpenCV highgui on a full-screen 1280×720 NumPy canvas (cv2 primitives + Hershey
fonts — ASCII only, no HTML/CSS/DOM). Runs on a Qualcomm Dragonwing IQ-8275 board
(QCS8275 / HTP v75 NPU) under Ubuntu 24.04 on an HDMI display. "web" is recorded
only because the Impeccable schema requires one of web/ios/android/adaptive; none
fit. All design work targets the OpenCV canvas renderer, not a browser. -->

## Users

Two audiences, weighted equally:

1. **Qualcomm field application engineers and sales** running the demo live at trade
   shows, booths, and customer visits. The board sits on a table with an HDMI display
   and a USB webcam; a presenter walks a visitor up, shows a face, toggles CPU↔NPU, and
   talks through what changes on screen. The demo must look polished, read from a few
   feet away, and run unattended without blanking or crashing.
2. **Makers and developers evaluating the IQ-8 board**, who clone the public repo
   (github.com/munoz0raul/dorbell) and reproduce the demo themselves from the README to
   learn the on-device NPU workflow (model conversion → quantize → compile → run).

## Product Purpose

An autonomous face-recognition doorbell that runs the entire pipeline — detection,
recognition, QR-triggered self-enrollment, and live telemetry — on the edge, on a
Qualcomm Dragonwing IQ-8275, with the same InsightFace models runnable on either the
CPU (onnxruntime) or the HTP v75 NPU (via `qnn-net-run`), switchable live.

Success is a viewer leaving with one understanding: **there is an easy, continuous path
from the maker ecosystem to a production-shaped embedded product.** The project is the
reference for a three-stage journey:

- **Stage 1** — Arduino VENTUNO Q + AppLab (maker entry; least friction).
- **Stage 2** — IQ-8275 + Ubuntu 24.04 + USB webcam + HDMI (prototyping start; the fully
  documented stage).
- **Stage 3** — IQ-8275 + MIPI CSI + QLI 2.1 + systemd autostart (production-shaped).

The CPU↔NPU toggle is the demo's live proof point (measured ~4× detection, ~18×
recognition speedup, with CPU load dropping as work offloads to the HTP), but it serves
the larger narrative of the maker→production ramp rather than standing alone.

## Positioning

The same recognition pipeline and the same models run on both CPU and NPU and switch
live on one keypress — so the NPU's benefit is shown as a measured, side-by-side
before/after on identical work, not asserted. On this board the Python
QNNExecutionProvider silently falls back to CPU, so the real NPU path is the
`qnn-net-run` CLI driven by a batching engine (up to 16 frames per invocation to
amortize the fixed ~270–305 ms spawn/model-load cost) — a non-obvious implementation
detail that is core to the demo actually reaching ~15 FPS on the NPU.

## Operating Context

- **Hardware:** Dragonwing IQ-8275 (QCS8275, HTP v75), USB UVC webcam at `/dev/video0`,
  HDMI display, keyboard. QAIRT 2.46.0 runtime on the board (`qnn-net-run`,
  `qnn-context-binary-generator`, `libQnnHtp.so` at `/usr/lib/`).
- **Launch:** full-screen window on the HDMI output. On the demo board a GNOME desktop
  icon (`dragonwing-doorbell-npu.desktop`) and `run_doorbell.sh` wrapper start it one
  click on the NPU backend; the wrapper also disables GNOME idle blanking so the display
  stays awake, and enforces a single instance. Over SSH the app discovers the GNOME
  Xwayland session (`DISPLAY=:0`, `/run/user/1000/.mutter-Xwaylandauth.*`).
- **Interaction:** keyboard controls (B toggle backend, K/U force demo results, R manual
  enroll, ESC standby, Q quit) plus the QR-code trigger — showing a LinkedIn QR
  authorizes a visitor and starts enrollment.
- **State machine:** LOADING → STANDBY → DETECTION → {RECOGNIZED | UNKNOWN}, with
  QR_AUTHORIZED → REGISTRATION branching off, and a no-face timeout returning to Standby.

## Capabilities and Constraints

- **Rendering:** OpenCV highgui on a 1280×720 NumPy canvas. All text is Hershey fonts —
  **ASCII only**; no Unicode glyphs (·, ▸, arrows render as `?`). Decorative marks must
  be drawn with cv2 primitives (polylines, fillConvexPoly, ellipse), not font glyphs.
- **Deployment (current):** `doorbell.py` is a single file copied to the board and run;
  no packaging/build step for the app. (A convenience, not a user-pinned commitment.)
- **Privacy (current behavior):** the face database is in-memory only with a 2-minute
  TTL; nothing is persisted to disk. (True today; not pinned as non-negotiable.)
- **Threading:** cooperating daemon threads — CameraThread (V4L2, synthetic-frame
  fallback when no camera), FaceEngine (CPU), NpuFaceEngine (NPU batch), StatsMonitor
  (psutil per-core CPU% + RAM every 0.5 s), RegistrationDriver. Visual work must not
  change this proven threading/engine behavior.
- **Models:** InsightFace `buffalo_sc` — `det_500m` (detection, 1×3×320×320) and
  `w600k_mbf` (recognition, 1×3×112×112 → 512-d embedding). NPU path uses W8A16 HTP
  context binaries at `~/qnn_context/*.bin`.
- **Secrets:** the AI Hub token is read from the `QAI_HUB_TOKEN` env var, never
  committed. The repo is code-only; models, calibration, and flash images are gitignored.

## Brand Commitments

**Binding (user-pinned):**

- **Dragonwing brand identity.** Deep indigo `#31017D` (BGR `(125,1,49)`) ground and
  electric violet `#5B0CD9` (BGR `(217,12,91)`) accent; the dragon-head mark and the
  DRAGONWING / DOORBELL wordmark lockup.

  **Reference images** (`assets/`, 1920×1080, the binding source of the visual language —
  future work matches these pixels, not a prose paraphrase):

  - **`assets/1.png` — the contour-wave field.** Dozens of fine, evenly-spaced parallel
    curved strokes in bright violet (`#5B0CD9`), fanning and twisting across a deep-indigo
    (`#31017D`) ground like a ribbon of wind, massed in the lower-center and thinning
    toward the edges. This is the atmosphere motif, not a logo. **Binds:** every
    full-screen brand state (loading, standby, QR flow) carries this as its ground,
    recreated as an animated `cv2.polylines` wave field over an indigo vertical gradient —
    thin bright-violet lines, never heavy; motion is a slow drift, never a flash. It sits
    *behind* content and must never compete with the access-result color or the mark.

  - **`assets/2.png` — the dragon-head mark, lockup crop.** Two interlocking swooshes form
    a stylized dragon/phoenix head: a large rounded violet body with a hooked upper beak/crest
    sweeping right and a lower jaw curling back, with a **negative-space crescent eye** cut
    from the violet by the indigo ground showing through. The head is offset to the **right**,
    leaving open indigo field on the left — composed for a wordmark to sit beside it.
    **Binds:** this is the canonical lockup placement and the **desktop icon**
    (`~/.local/share/icons/dragonwing-doorbell.png`). The mark is reproduced as a *drawn
    vector glyph* (filled `cv2` ellipses/convex polygons plus the negative-space eye), never
    a pasted bitmap, so it scales cleanly on the canvas. The crescent eye is load-bearing —
    it must always read.

  - **`assets/3.png` — the dragon-head mark, hero crop.** The same mark, centered and zoomed
    larger, standing alone with indigo field all around. **Binds:** the standalone/hero
    placement — centered on the loading and standby screens, pulsing-glow treatment allowed.
    Same drawn-glyph rule as 2.png; 2.png and 3.png are the same symbol at two framings
    (beside-wordmark vs. centered-solo), not two different marks.

  The two swoosh forms, the crescent-eye negative space, and the indigo/violet-only palette
  of the mark are the invariants; crop and scale are the degrees of freedom.

  **Official Qualcomm Dragonwing brand assets** (`assets/`, added as the authoritative
  source — these are the real Qualcomm brand marks, and they supersede 1/2/3.png as the
  reference for mark geometry, wordmark, and type. The demo remains an unaffiliated
  technical demonstration, so it uses its own "DRAGONWING DOORBELL" canvas lockup rather
  than impersonating the official product lockup; these assets govern *style fidelity*, not
  product identity claims):

  - **`assets/Qualcomm-Dragonwing-logo.jpeg` — primary horizontal lockup.** The official
    logo on a **white** ground: the dragon-head mark in deep indigo (`#31017D`) at left,
    with the wordmark to its right — "Qualcomm" in a lighter/regular weight stacked over
    "**Dragonwing**" in bold, both indigo. The authoritative mark is more detailed than the
    1/2/3.png abstraction: a dragon head with a hooked crest and the negative-space crescent
    eye, enclosed in a rounded body that curls into a **flared tail sweeping down-right**,
    the whole mark reading as a near-circular emblem. **Binds:** the canonical mark geometry
    (crest + crescent eye + body + flared tail curl) and the real wordmark spelling and
    weight relationship (light "Qualcomm" over bold "Dragonwing"). Confirms the mark works
    as a positive indigo form on light grounds, not only reversed on indigo.

  - **`assets/qualcomm-unveils-dragonwing-brand-for-industrial-and-enterpr_yqj8.jpg` —
    square brand badge.** A rounded-corner **indigo tile**: "Qualcomm Dragonwing" wordmark
    in near-white at the top, the dragon-head mark in light lavender/near-white below. The
    reversed (light-on-indigo) treatment of the lockup. **Binds:** the app's own indigo
    full-screen states are the same reversed treatment — light mark and type on an indigo
    ground — so this is the closest official analog to the on-canvas look and the model for
    the stacked wordmark-over-mark vertical arrangement.

  - **`assets/1x1_1200x1200_highres-qualcomm-dragonwing.webp` — composite hero tile.** A
    square social/hero image: indigo gradient ground with the fine **contour-wave lines**
    (the same motif as 1.png) sweeping through the corners, and a white circular dragon-mark
    badge beside the "Qualcomm Dragonwing" wordmark, composed left-of-center. **Binds:** this
    is the official proof that the three elements combine — wave field + circular mark badge +
    wordmark on indigo — which is exactly the composition the loading/standby screens target.
    It is the single best full-reference for the intended on-screen brand feel.

  Relationship: **1/2/3.png are stylized demo-canvas renditions** of the mark and wave motif
  (built for cheap `cv2` redraw); **these three are the official Qualcomm brand assets** and
  are authoritative for geometry, wordmark, type character (a rounded humanist sans with soft
  terminals), and the reversed-on-indigo treatment. Where they disagree, the official assets
  win on fidelity, within the ASCII-font and drawn-glyph constraints of the OpenCV canvas.
- **Universal access-control color semantics.** Green = unlocked / known, red = denied /
  unknown. These must stay universally legible and must **never** be recolored to fit the
  brand palette. Brand chrome (frames, HUD, panels, standby, loading) is indigo/violet;
  the semantic green/red is reserved for the access result and the per-core load meter.

Name: **Dragonwing Doorbell**. Qualcomm, Dragonwing, and Snapdragon are Qualcomm
trademarks; this is a technical demonstration, not an affiliated product.

## Evidence on Hand

- Working application: `doorbell.py` (deployed and running on the board on the NPU
  backend as of this session).
- Measured performance (`benchmark.py`, QCS8275): detection CPU ~47 ms / NPU ~11 ms
  (~4×); recognition CPU ~91 ms / NPU ~5 ms (~18×); ~15 FPS end-to-end on the NPU.
- Public repo with comprehensive README documenting Stage 2 end-to-end; `models/README.md`
  documents the full conversion pipeline. Headless QA harness (`tools/render_states.py`)
  renders every UI state to PNG.
- Stages 1 and 3 are currently **README stubs** — future work must not fabricate their
  details beyond what is confirmed here.

## Product Principles

1. **The maker→production ramp is the story.** Every surface should make the journey from
   AppLab to a production appliance feel continuous and attainable, not just show one
   board working.
2. **Prove the NPU, don't claim it.** Benefits are shown as live, measured, side-by-side
   results on identical work (the B toggle), never as marketing numbers alone.
3. **Brand chrome and safety semantics never trade places.** Indigo/violet owns the
   identity; green/red owns the access result. Neither borrows the other's color.
4. **Demo-floor robustness.** It must read from a few feet away, run unattended without
   the screen blanking, survive a missing camera (synthetic frames), and restart cleanly
   from one click.
5. **Reproducible and honest.** Code-only public repo, secrets via env, no persisted
   biometric data; a developer can follow the README from flash to `--backend npu`.

## Accessibility & Inclusion

Legibility at demo-booth distance on an HDMI display drives type size and contrast. The
green/unlocked vs red/denied result is reinforced by text labels ("UNLOCKED", "ACCESS
DENIED") and bracket color together, so the state does not rely on color alone.
