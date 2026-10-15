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
| **DMA-heap + FastRPC nodes** | passed via CDI | CDI injection stamps the nodes **gid 0**, so `--group-add dmaheap/fastrpc` can't grant access and the uid-1000 process is denied the ION heap (NPU dies) | the nodes are passed as **plain `docker run --device`** (which **preserves the host gid** — `dmaheap`/`fastrpc`), *not* via the CDI; the CDI now carries only the lib mounts. `--group-add` unlocks them. |
| **Camera node** | USB cam usually `/dev/video0` | on-SoC Iris codec + ISP occupy `video0..27`; the **USB webcam enumerates high** (e.g. `/dev/video28`) | the wrapper passes through only the **USB-backed** `/dev/video*` nodes; `doorbell.py` probes the actual present nodes (not a fixed `0..5` range). |
| **QAIRT version match** | n/a | the on-SoC DSP firmware ships a **specific QAIRT version** (e.g. **2.50.0** on this QLI 2.1 build); a mismatched host QNN is rejected on-device (`openSessionForPriority status=0x72`, `Failed to load skel 1002`) → silent CPU fallback | `qai-appbuilder` is pinned in `../requirements.txt` to the matching version (**2.50.40**). Verify firmware: `strings .../dsp/cdsp/libQnnHtpV75.so \| grep v2.5`. |
| **FastRPC DSP shell** | n/a | the FastRPC lib derives the board DSP dir from `/sys/firmware/devicetree/base/model` — **absent in the container** — then only probes hardcoded dirs; `DSP_LIBRARY_PATH` is *not* consulted for the shell | the wrapper auto-detects the board `…/dsp/cdsp` dir (the one holding `fastrpc_shell_unsigned_3` + `libQnnHtpV75Skel.so`) and bind-mounts it onto the hardcoded **`/usr/lib/dsp/cdsp`**. |

The NPU runtime needs no QNN SDK on the host: `qai-appbuilder` (installed by the
image) bundles its own QNN libraries + the HTP **v75** skel — but its bundled
QNN version **must match the host DSP firmware's QAIRT** (see the "QAIRT version
match" row), and the host's **FastRPC DSP shell** must be reachable at a hardcoded
path (the "FastRPC DSP shell" row). The CDI forwards only the lib mounts; the
kernel-coupled device nodes are passed as plain `--device` so their host gids
survive — see [`../README.md`](../README.md) for the full rationale.

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

The NPU HTP context binaries are **committed to the repo** under
[`../../qnn_context/`](../../qnn_context/) (~5 MB, SoC-pinned), so a fresh
`git clone` already has them — the wrapper mounts the repo copy automatically
when `$MODELS_HOME/qnn_context` is absent. **Nothing to stage for the NPU path**
beyond the clone. The CPU ONNX models are not committed; InsightFace
auto-downloads `buffalo_sc` into `~/.insightface` on first use (or stage them for
offline CPU).

```
qnn_context/det_500m_qcs8275.bin   qnn_context/w600k_mbf_qcs8275.bin   # NPU — committed in-repo
$MODELS_HOME/.insightface/models/buffalo_sc/*.onnx                     # CPU — auto-downloaded
```

To run **freshly regenerated** binaries instead, stage them at
`$MODELS_HOME/qnn_context/` and the wrapper prefers that over the repo copy. The
`.bin` HTP context binaries are produced by the AI Hub → `precompiled_qnn_onnx`
pipeline (see [`../../qnn_context/README.md`](../../qnn_context/README.md)); they
are **generated artifacts pinned to this SoC + QAIRT version**, committed only
because they are small and slow to regenerate.

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
- **HTP falls back to CPU with `openSessionForPriority … status=0x72` /
  `Failed to load skel 1002` / `Transport setup 14001`.** The context binaries
  loaded but the DSP refused the session. Three independent causes (all handled
  by the current wrapper + image; this list is for when it breaks on a new board):
  1. **QAIRT version skew** — the bundled `qai-appbuilder` QNN ≠ the host DSP
     firmware's QAIRT. The failing log line shows the host QNN version in the
     skel URI (`…?qnn_2_51_40_skel_handle_invoke`); compare to the firmware
     (`strings …/dsp/cdsp/libQnnHtpV75.so | grep v2.`). Pin the matching
     `qai-appbuilder` in `../requirements.txt` (available: 2.51.40, 2.50.40,
     2.48.40, 2.40.0) and rebuild.
  2. **FastRPC shell not found** — the lib can't derive the board DSP dir
     (`/sys/firmware/devicetree/base/model` is absent in-container) and only
     probes hardcoded dirs. The wrapper bind-mounts the board `…/dsp/cdsp` onto
     `/usr/lib/dsp/cdsp`; if your board dir is elsewhere, fix the auto-detect
     `find /usr/share/qcom …` in the wrapper.
  3. **DMA-heap/fastrpc gid = 0** — if you pass the nodes via the CDI instead of
     plain `--device`, they arrive gid 0 and `--group-add` can't help. Confirm
     `stat -c %g /dev/dma_heap/system` *inside* the container shows the host gid
     (not 0). Isolate the whole DSP path with `fastrpc_test -d 3` (passes
     bare-metal; make it pass in-container too).

  Note `QNNContext()` prints `[NPU] … ready` even on session failure; trust a
  non-trivial `load Nms` (hundreds of ms, not ~4 ms) and the **absence** of the
  `0x72`/`err 4000` lines instead.
- **HTP falls back to CPU (`loadRemoteSymbols failed err 4000`).** A host lib the
  CDI forwards didn't reach the container. `ldd` the mounted
  `/usr/lib/aarch64-linux-gnu/libcdsprpc.so` *inside* the container
  (`docker run --rm --device qualcomm.com/device=cdi-hw-acc --entrypoint bash
  dragonwing-doorbell -c 'ldd …'`); any `=> not found` is the culprit — add that
  lib to the image (apt) or the CDI (bind-mount) depending on whether it's a
  stock Ubuntu package or a host-coupled one.
- **`[NPU] ERROR: context binaries missing`.** The `*.bin` files aren't under
  `$MODELS_HOME/qnn_context` — stage them (see **Models** above). CPU backend
  works without them.
