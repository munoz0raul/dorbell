# Run the Dragonwing Doorbell in a container on **Qualcomm Linux (QLI 2.1)**

This is the QLI (Yocto) counterpart of [`../README.md`](../README.md), which
targets an **Ubuntu** host. The container **image is identical** (same
`../Dockerfile`, `ubuntu:24.04` base, portable) — only the **host plumbing**
differs, because QLI is not Ubuntu. Everything here was built and verified
end-to-end on an **IQ-8275 EVK running QLI 2.1** (HTP v75), full-screen on the
Weston/HDMI display with a USB webcam.

## What differs from the Ubuntu host, and why

| Area | Ubuntu host | QLI (Yocto) host | How it's handled |
|------|-------------|------------------|------------------|
| **Display auth** | GNOME publishes a `.mutter-Xwaylandauth` cookie | **Weston** — its bundled Xwayland authorizes clients by **matching UID** over the X11 socket; there is **no cookie file** | run the container as **`--user 1000:1000`** (the `weston` uid, auto-detected from the owner of `/tmp/.X11-unix/X0`). No `xhost`, no cookie, no access-control weakening. |
| **Container `$HOME`** | root, `$HOME=/root` | uid 1000 → not `/root` | set `HOME=/models` in the container and mount both model trees under it, so `doorbell.py`'s `expanduser("~/qnn_context")` / `~/.insightface` resolve regardless of uid. Qt/cache writes redirected to `/tmp`. |
| **Library paths** | `/usr/lib/aarch64-linux-gnu/…` (multiarch) | **flat `/usr/lib/…`** | the QLI CDI (`cdi-hw-acc-8275-qli.json`) bind-maps each host lib to the multiarch path the `ubuntu:24.04` container expects. |
| **FastRPC lib deps** | host `libcdsprpc.so` links `libdmabufheap.so.0` | varies by QLI build; often links **`libyaml-0.so.2`** and `libbsd.so.0` instead, and `libdmabufheap` may be absent | `libyaml-0-2` is apt-installed in `../Dockerfile`; the CDI lists every candidate lib and `install_cdi.py` **prunes whatever the host lacks**. Always `ldd` the mounted `libcdsprpc.so` *inside* the container to catch a `=> not found`. |
| **DMA-heap nodes** | `/dev/dma_heap/system` + `qcom,system` | usually only **`/dev/dma_heap/system`** | the QLI CDI lists only `system`. |
| **Camera node** | USB cam usually `/dev/video0` | on-SoC Iris codec + ISP occupy `video0..27`; the **USB webcam enumerates high** (e.g. `/dev/video28`) | the wrapper passes through only the **USB-backed** `/dev/video*` nodes; `doorbell.py` probes the actual present nodes (not a fixed `0..5` range). |

The NPU runtime itself needs no QNN SDK on the host: `qai-appbuilder` (installed
by the image) bundles its own QNN libraries + the HTP **v75** skel. The CDI only
forwards the kernel-coupled FastRPC/DMA plumbing — see [`../README.md`](../README.md)
for the full rationale.

## One-time setup on the board

Copy this repo's `container/` directory and `doorbell.py` to the board (e.g.
under `/root/dorbell`), then:

```bash
# 1. Install the QLI CDI  (writes /etc/cdi/cdi-hw-acc.json, pruning absent libs)
cd /root/dorbell/container/qli
python3 ../cdi/install_cdi.py --file cdi-hw-acc-8275-qli.json

# 2. Build the image  (from the repo root, so doorbell.py is in the build context)
cd /root/dorbell
docker build -t dragonwing-doorbell -f container/Dockerfile .
```

Docker on QLI 2.1 ships with CDI support enabled (`docker info` lists the
`/etc/cdi` spec dir and discovers `qualcomm.com/device=cdi-hw-acc`).

## Models on the host

Same as bare metal — large, SoC-specific, **never** baked into the image or an
OS flash, so they must be staged after each reflash:

```
/root/qnn_context/det_500m_qcs8275.bin      /root/qnn_context/w600k_mbf_qcs8275.bin   # NPU (HTP)
/root/.insightface/models/buffalo_sc/det_500m.onnx   …/w600k_mbf.onnx                 # CPU
```

The `.bin` HTP context binaries are produced by the AI Hub → `qnn-context-binary-generator`
pipeline in [`../../README.md`](../../README.md) §2.3 — they are **generated
artifacts, not shipped in any QLI image**, so a reflash erases them. The CPU
ONNX files are enough to run the CPU backend.

## Run

```bash
cd /root/dorbell/container/qli
bash run_doorbell_container_qli.sh npu     # NPU (needs the *.bin context binaries)
bash run_doorbell_container_qli.sh cpu     # CPU fallback (B toggles live too)
```

The wrapper auto-detects the Weston display uid, resolves the
`fastrpc`/`dmaheap`/`video` GIDs, passes the USB-backed camera nodes, mounts the
models read-only under `/models`, and starts the container. Controls are
identical to bare metal (`O` enroll, `B` backend toggle, `[`/`]` timeout,
`q`/ESC quit) — see [`../../README.md`](../../README.md) §2.5.

Env overrides: `WESTON_UID` (default: owner of `/tmp/.X11-unix/X0`),
`MODELS_HOME` (default: the invoking user's `$HOME`).

## Troubleshooting (QLI-specific deltas)

- **`could not connect to display :0` / `Authorization required`.** The
  container isn't running as the uid that owns the Weston X socket. Confirm
  `stat -c %u /tmp/.X11-unix/X0` (usually `1000`) and that the wrapper passed
  `--user <that uid>`. Weston has no cookie to mount — UID match *is* the auth.
- **`[Camera] No camera — using synthetic frames` with a USB cam plugged in.**
  The cam enumerates high on QLI (behind the ISP nodes). Check
  `v4l2-ctl --list-devices` for its real node and that the wrapper passed it
  (`--device /dev/videoNN`). `doorbell.py` probes all present nodes and skips the
  non-capture ISP/codec ones (they open but return no frame).
- **HTP falls back to CPU (`loadRemoteSymbols failed err 4000`).** A host lib the
  CDI forwards didn't reach the container. `ldd` the mounted
  `/usr/lib/aarch64-linux-gnu/libcdsprpc.so` *inside* the container
  (`docker run --rm --device qualcomm.com/device=cdi-hw-acc --entrypoint bash
  dragonwing-doorbell -c 'ldd …'`); any `=> not found` is the culprit — add that
  lib to the image (apt) or the CDI (bind-mount) depending on whether it's a
  stock Ubuntu package or a host-coupled one. Note `QNNContext()` prints
  `[NPU] … ready` even on skel-load failure; trust a non-trivial `load Nms` and
  the absence of `err 4000` instead.
- **`[NPU] ERROR: context binaries missing`.** The `*.bin` files aren't under
  `$MODELS_HOME/qnn_context` — stage them (see **Models** above). CPU backend
  works without them.
