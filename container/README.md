# Run the Dragonwing Doorbell in a container (IQ-8275 + Ubuntu)

This directory packages the Stage-2 demo as a Docker image that runs on the
IQ-8275 board exactly like the bare-metal `python3 doorbell.py` — full-screen on
the HDMI display, reading the USB webcam, running recognition on the **HTP v75
NPU** — just containerized. `doorbell.py` is unchanged; it already auto-discovers
the display and camera and loads models from `$HOME`.

## How the NPU is reached from the container

The official Dragonwing way to use the NPU inside Docker on Ubuntu is a
**Container Device Interface (CDI)** file. It forwards the kernel-coupled plumbing
— the FastRPC device node, the DMA heaps, and the host's FastRPC userspace stubs —
into the container, and you select it with a single `docker run` flag:
`--device qualcomm.com/device=cdi-hw-acc`.

The key simplification for this app: the NPU path uses **`qai-appbuilder`** (a pip
wheel), which **bundles its own QNN runtime, including `libQnnHtpV75Skel.so` and
the HTP v75 stub**, and sets `ADSP_LIBRARY_PATH` itself. So the image does **not**
install any QNN SDK or host QNN libraries — `pip install qai-appbuilder` in the
Dockerfile supplies them. The CDI only forwards what *cannot* be pip-installed
because it is coupled to the host kernel driver:

| What | Why |
|------|-----|
| `/dev/fastrpc-cdsp` | FastRPC channel to the compute DSP |
| `/dev/dma_heap/system`, `/dev/dma_heap/qcom,system` | DMA buffers shared with the HTP |
| `/usr/lib/aarch64-linux-gnu/lib{cdsp,adsp}rpc.so*` | host userspace stubs — must match the host kernel driver, so bind-mounted, not reinstalled |
| `/usr/lib/aarch64-linux-gnu/libdmabufheap.so.0*` | DMA-heap allocator — **both `libcdsprpc.so` and qai-appbuilder's `libQnnHtpV75Stub.so` link it**, and it is absent from a minimal `ubuntu:24.04`. Without it FastRPC fails with `loadRemoteSymbols failed err 4000` and the NPU silently falls back to CPU |
| `/usr/lib/dsp/cdsp` | DSP shell + skels (not strictly required for this app — qai-appbuilder's bundled v75 skel is self-sufficient — but harmless and matches upstream) |

Two further things this CDI deliberately gets right (they cost real debugging to find):

- **No `ADSP_LIBRARY_PATH` in the CDI `env`.** CDI env entries are injected *literally* (no shell expansion), so the upstream value `"/usr/lib/dsp/cdsp;${ADSP_LIBRARY_PATH}"` would arrive with literal quotes and an unexpanded `${...}`, and would *block* qai-appbuilder from setting its own correct path to the bundled HTP v75 skel. We omit it and let qai-appbuilder set it.
- **`libatomic1` is apt-installed in the image** (see the Dockerfile) — qai-appbuilder's native libs link `libatomic.so.1`, also absent from minimal `ubuntu:24.04`; without it `import qai_appbuilder` throws and the NPU path falls back to CPU.

The upstream example ships CDIs for HTP **v73** (RB3 6490, IQ-9075 9100) and they
carry ~200 GStreamer binaries. **`cdi/cdi-hw-acc-8275.json` here is authored for
the IQ-8275 (HTP v75) and trimmed to just the doorbell's NPU needs** — no
GStreamer, no host QNN libs.

## One-time setup on the board

### 1. Install Docker

```bash
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker "$USER"
# log out and back in (or: newgrp docker) so the group takes effect
```

### 2. Install the CDI

Copy this `container/` directory and `doorbell.py` to the board, then:

```bash
sudo python3 cdi/install_cdi.py --file cdi/cdi-hw-acc-8275.json
```

The installer writes `/etc/cdi/cdi-hw-acc.json`, automatically dropping any mount
whose host path is absent. It should report **no missing mounts** on a stock
IQ-8275 Ubuntu 24.04 image. (If it says the file already exists, remove the old
`/etc/cdi/cdi-hw-acc.json` first.)

### 3. Build the image

Build from the **repo root** so `doorbell.py` is in the build context:

```bash
docker build -t dragonwing-doorbell -f container/Dockerfile .
```

The aarch64 wheels (insightface, onnxruntime, opencv, qai-appbuilder) take a few
minutes to install on the first build.

### 4. Models — committed for NPU, auto-downloaded for CPU

The NPU HTP context binaries are **committed to the repo** under
[`qnn_context/`](../qnn_context/) (only ~5 MB, SoC-pinned), so after `git clone`
they are already present — the wrapper mounts them automatically. **Nothing to
stage for the NPU path.** The CPU ONNX models are **not** committed: InsightFace
auto-downloads `buffalo_sc` into `~/.insightface` on first use.

```
qnn_context/det_500m_qcs8275.bin   qnn_context/w600k_mbf_qcs8275.bin   # NPU — committed in-repo
~/.insightface/models/buffalo_sc/*.onnx                                # CPU — auto-downloaded
```

To use **freshly regenerated** NPU binaries instead of the committed ones, stage
them at `~/qnn_context/` — the wrapper prefers that over the repo copy. See
[`qnn_context/README.md`](../qnn_context/README.md) for how the binaries are
produced and the main [README](../README.md) §2.3.

## Run

```bash
bash container/run_doorbell_container.sh npu     # NPU (default)
bash container/run_doorbell_container.sh cpu     # CPU fallback (B toggles live too)
```

The wrapper is the container analogue of `run_doorbell.sh`: it re-asserts the
GNOME/Xwayland display keep-awake settings on the host, removes any prior
container (single-instance), resolves the `fastrpc`/`dmaheap`/`video` group IDs
and the present `/dev/video*` nodes at runtime, and starts the container with:

- `--device qualcomm.com/device=cdi-hw-acc` — the NPU, via the CDI
- the USB camera nodes + `--group-add` for their restricted groups
- the host Xwayland socket (`/tmp/.X11-unix` + `/run/user/1000`) and auth cookie, with `DISPLAY=:0`
- the model directories mounted read-only under `/root` (the committed
  `qnn_context/`, or `~/qnn_context` if you staged your own; plus `~/.insightface`
  if present)

The controls (`O` enroll, `B` backend toggle, `[`/`]` timeout, `q`/ESC quit, …)
are identical to bare metal — see main README §2.5.

## Troubleshooting (deltas vs bare metal)

- **No window appears / `could not connect to display :0`.** The host desktop
  session must be up and own `:0`. The X11 socket (`/tmp/.X11-unix`) *and* the
  Xwayland auth cookie must both be mounted — the wrapper does both. Check
  `ls /tmp/.X11-unix/X0` and `ls /run/user/1000/.mutter-Xwaylandauth.*` exist on
  the host; the wrapper warns if the cookie is missing.
- **HTP falls back to CPU (`loadRemoteSymbols failed err 4000`, `Failed to create
  transport`).** This is almost always a missing userspace library, *not* a
  permissions problem — `--privileged` / `--pid=host` do **not** fix it. The usual
  cause is `libdmabufheap.so.0` not reaching the container (the CDI mounts it); run
  `docker run --rm --device qualcomm.com/device=cdi-hw-acc dragonwing-doorbell` with
  `--entrypoint bash` and `ldd /usr/lib/aarch64-linux-gnu/libcdsprpc.so` — any
  `=> not found` is the culprit. Note that `QNNContext()` prints `[NPU] ... ready`
  even when the HTP never came up, so trust the `err 4000` lines (and a near-zero
  `load Nms`) over the "ready" line. Definitive liveness check: on the host,
  `sudo ls -l /proc/<container-pid>/fd | grep fastrpc` — if the FastRPC fd is held
  open, the DSP transport is really up.
- **Camera not found.** The app probes `video0..5` and falls back to synthetic
  frames. Confirm the USB cam enumerates (`v4l2-ctl --list-devices`) and that its
  node is in the `0..3` range the wrapper passes; add higher nodes to the `for n in`
  loop if it enumerates higher.
- **highgui window creation fails.** This image uses `opencv-python-headless` + the
  host Xwayland socket + the X11 client libs in the Dockerfile. If window creation
  still fails, switch the image to full `opencv-python`.
