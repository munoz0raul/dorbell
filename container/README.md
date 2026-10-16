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

### 2. Install the NPU runtime stack (fresh Canonical Ubuntu only)

A **fresh Canonical Ubuntu 24.04 IQ-8275 image does NOT preinstall the
FastRPC/QNN host stack** — no `libcdsprpc.so`, no DMA-heap allocator, no
`/dev/fastrpc-cdsp` group. The CDI (next step) forwards these host libraries and
device nodes into the container, so they must exist on the host first. Install
them from the Qualcomm IoT PPA:

```bash
sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
sudo apt update
sudo apt install -y qcom-fastrpc1 qcom-libdmabufheap qairt-libs qairt-dsp-binaries
```

This brings `/usr/lib/aarch64-linux-gnu/lib{cdsp,adsp}rpc.so*` and
`libdmabufheap.so.0`, creates the `fastrpc` and `dmaheap` groups, and sets
`/dev/fastrpc-cdsp` to `fastrpc:fastrpc 0664`. **Then create the unversioned
FastRPC symlinks** — `qcom-fastrpc1` installs only the versioned `*.so.1`, but the
CDSP runtime `dlopen`s the unversioned name; without these the CDI prunes the
mount and the NPU silently falls back to CPU (`loadRemoteSymbols failed err 4000`,
`load 6ms`):

```bash
sudo ln -sf libcdsprpc.so.1 /usr/lib/aarch64-linux-gnu/libcdsprpc.so
sudo ln -sf libadsprpc.so.1 /usr/lib/aarch64-linux-gnu/libadsprpc.so
```

Add yourself to the groups (the
run wrapper passes them to the container, but you need them to probe on the host):

```bash
sudo usermod -aG fastrpc,dmaheap "$USER"   # log out/in for this to take effect
```

> **QLI (Yocto) hosts skip this step** — the QLI multimedia image ships the
> FastRPC/QNN runtime as part of the OS. This step is **only** for a fresh
> Canonical Ubuntu image. See [`qli/README.md`](qli/README.md) for the QLI path.

### 3. Install the CDI

Copy this `container/` directory and `doorbell.py` to the board, then:

```bash
sudo python3 cdi/install_cdi.py --file cdi/cdi-hw-acc-8275.json
```

The installer writes `/etc/cdi/cdi-hw-acc.json`, automatically dropping any mount
whose host path is absent. After step 2 it should report **no missing mounts**;
if you skipped step 2 it will prune the FastRPC libs and the NPU will silently
fall back to CPU. (If it says the file already exists, remove the old
`/etc/cdi/cdi-hw-acc.json` first.)

### 4. Build the image

Build from the **repo root** so `doorbell.py` is in the build context:

```bash
docker build -t dragonwing-doorbell -f container/Dockerfile .
```

The aarch64 wheels (insightface, onnxruntime, opencv) plus the QNN runtime take a
few minutes to install on the first build. **Use the plain build above on every
board.** The committed context binaries are generated at QAIRT 2.50, so the image
installs a **2.50.x `qai-appbuilder`** — specifically Qualcomm's public GitHub
release `qualcomm/qai-appbuilder` **v2.50.0** (PyPI no longer ships any 2.50.x
wheel). That build is correct on every board whose `.bin` is the committed 2.50
set, **even one whose DSP firmware is older** (e.g. the Canonical Ubuntu image's
2.46 — the older firmware still accepts the 2.50-host HTP session). Do **not**
try to "match" older firmware with an older `qai-appbuilder`: that makes the host
SDK older than the `.bin` and the context load is rejected (`Using newer context
binary on old SDK`), silently dropping to CPU.

Why the image carries a **python3.10 venv** on an ubuntu:24.04 base (see the
Dockerfile header for the full reasoning): the only public 2.50.x `qai-appbuilder`
is a **cp310** wheel, but ubuntu:24.04's system Python is 3.12 — so the app runs
from a deadsnakes python3.10 venv. The base must stay ubuntu:24.04 because the
host `libcdsprpc.so` the CDI forwards links **GLIBC_2.38** (ubuntu:22.04 ships
only 2.35 → `loadRemoteSymbols failed err 4000`). To run `.bin` **regenerated** at
a different SDK, point the build at that version's wheel:

```bash
docker build -t dragonwing-doorbell -f container/Dockerfile \
  --build-arg QAI_APPBUILDER_WHEEL=https://github.com/qualcomm/qai-appbuilder/releases/download/v2.51.40/qai_appbuilder-2.51.40-cp310-cp310-manylinux_2_39_aarch64.whl .
```

### 5. Models — committed for NPU, auto-downloaded for CPU

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
- **Build fails: `No matching distribution found for qai-appbuilder==2.50.40`.**
  PyPI removed every 2.50.x wheel. The current Dockerfile does not use PyPI for
  it — it installs the public GitHub wheel into a python3.10 venv. Make sure you
  are on a current checkout (`git pull`); an older Dockerfile pinned the
  now-absent PyPI version.
- **HTP falls back to CPU with `loadRemoteSymbols failed err 4000` /
  `Transport layer setup failed: 14001` on the FastRPC stub.** The container
  base's GLIBC is older than the host `libcdsprpc.so` requires (it links
  `GLIBC_2.38`). Keep the base at **ubuntu:24.04** (GLIBC 2.39); ubuntu:22.04
  (2.35) triggers exactly this. Confirm inside the container with `ldd` on the
  bundled `libQnnHtpV75Stub.so` — a `GLIBC_2.38 not found (required by
  libcdsprpc.so)` line is the signature.
- **HTP falls back to CPU with `Using newer context binary on old SDK` /
  `Fail to get context blob with err 5000` / `Failed to create context from
  binary with err 0x1388`.** The image's `qai-appbuilder` host SDK is **older**
  than the SDK that generated `qnn_context/*.bin` (2.50). This is a host-SDK vs
  `.bin`-generation check done *before* the DSP, so it is independent of the
  board's DSP firmware version — a 2.46-firmware board loads the 2.50 `.bin`
  fine with the default (2.50.0) image. The fix is to **not** downgrade: build
  plainly (`docker build -t dragonwing-doorbell -f container/Dockerfile .`) with
  no `QAI_APPBUILDER_WHEEL` override. Only override if you *regenerated* the
  `.bin` at a different SDK generation (point it at that version's cp310 wheel).
- **`install_cdi.py` prunes `libcdsprpc.so` / `libdmabufheap.so.0` on a fresh
  Ubuntu image.** You skipped step 2 — the Canonical image ships without the
  FastRPC/QNN runtime. Install it (`sudo apt install qcom-fastrpc1
  qcom-libdmabufheap qairt-libs qairt-dsp-binaries` from the
  `ubuntu-qcom-iot/qcom-ppa` PPA) and re-run the installer (`rm -f
  /etc/cdi/cdi-hw-acc.json` first).
- **HTP falls back to CPU (`loadRemoteSymbols failed err 4000`, `Failed to create
  transport`).** This is almost always a missing userspace library, *not* a
  permissions problem — `--privileged` / `--pid=host` do **not** fix it. Run
  `docker run --rm --device qualcomm.com/device=cdi-hw-acc dragonwing-doorbell` with
  `--entrypoint bash` and `ldd /usr/lib/aarch64-linux-gnu/libcdsprpc.so` — a
  `No such file or directory` means the lib never reached the container, and any
  `=> not found` names a missing dependency. Two common causes on **Canonical
  Ubuntu**: (1) the unversioned `libcdsprpc.so`/`libadsprpc.so` symlinks were never
  created — `qcom-fastrpc1` ships only `*.so.1`, so `install_cdi.py` pruned the
  mount (fix: `sudo ln -sf libcdsprpc.so.1 .../libcdsprpc.so` + same for adsprpc,
  then reinstall the CDI); (2) `libdmabufheap.so.0` not reaching the container.
  Note that `QNNContext()` prints `[NPU] ... ready`
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
