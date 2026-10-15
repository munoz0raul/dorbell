# Installation — IQ-8275 EVK + **Canonical Ubuntu 24.04 Desktop**

This is the exact install for running this demo's **containerized NPU (Neural
Processing Unit)** path on an **IQ-8275 EVK** flashed with the **Canonical Ubuntu
24.04 Desktop** image for Dragonwing (the `x15` preinstalled-desktop build), a USB
(Universal Serial Bus) webcam, and an HDMI (High-Definition Multimedia Interface)
display. It takes you from a bare board to the full-screen NPU doorbell running in
Docker.

> **Looking for the QLI (Qualcomm Linux Image) reference config instead?** See
> [`INSTALL.md`](INSTALL.md). That path is simpler because the QLI multimedia image
> ships the FastRPC/QNN (Qualcomm Neural Network) runtime **in the OS**. The
> Canonical Ubuntu image does **not**, so this page adds one extra step (§4) to
> install it from the Qualcomm IoT (Internet of Things) PPA (Personal Package
> Archive). Everything was verified end-to-end on an IQ-8275 EVK running Ubuntu
> 24.04.5 (DSP firmware QAIRT 2.46.0, HTP v75).

Why this works even though the board firmware is **QAIRT (Qualcomm AI Runtime SDK)
2.46** while the committed NPU context binaries were generated at **2.50**: the
version that matters is the **host SDK bundled in `qai-appbuilder` vs the SDK that
generated the `.bin`**, not the board's DSP (Digital Signal Processor) firmware.
The default image installs a 2.50.x `qai-appbuilder` (Qualcomm's public GitHub
v2.50.0 wheel), which matches the committed
`.bin`, and the older 2.46 DSP firmware accepts that HTP (Hexagon Tensor
Processor) session fine. **Do not downgrade `qai-appbuilder` to "match" 2.46** —
that re-triggers a `Using newer context binary on old SDK` rejection and silently
falls back to CPU (Central Processing Unit). See
[`container/README.md`](container/README.md) for the full rationale.

---

## 1. Download the Ubuntu image

Canonical publishes the Dragonwing Ubuntu images here (pick the latest `x`
revision; `x15` was used for this verification):

```bash
# Resolve the latest revision
curl -s -L 'https://people.canonical.com/~platform/images/qualcomm-iot/ubuntu-24.04/' \
  | grep -oE 'ubuntu-24\.04-x[0-9]+' | sort -t'x' -k2 -n | tail -1
# -> e.g. ubuntu-24.04-x15
```

Each revision ships a per-board YAML with the image URL, its SHA256, and the
matching boot-firmware URL. For the IQ-8275 desktop build the YAML is at:

```
https://people.canonical.com/~platform/images/qualcomm-iot/ubuntu-24.04/ubuntu-24.04-x15/ubuntu-desktop-24.04/ubuntu-24.04-preinstalled-desktop-arm64+dragonwing-x15-<date>-iq8.yaml
```

From it you need two things: the **image SHA256** (to verify your download) and the
**boot-firmware tarball URL** (the `QLI.1.8-Ver.1.1-ubuntu-QCS8300-nhlos-bins.tar.gz`
on the CodeLinaro artifacts server).

## 2. Assemble the flashable folder

Flashing needs the image plus the board firmware, device tree, and partition XMLs
collected into one directory. On your host (these commands are for macOS/Linux;
`<date>` is the revision's date stamp, e.g. `20260925`):

```bash
BASE='https://people.canonical.com/~platform/images/qualcomm-iot/ubuntu-24.04/ubuntu-24.04-x15/ubuntu-desktop-24.04'
OUT_DIR=~/ubuntu-iq8275/flashable-desktop
WORK_DIR=~/ubuntu-iq8275/work-desktop
mkdir -p "$OUT_DIR" "$WORK_DIR"

# Image, device tree, top-level rawprogram
curl -L --fail --retry 3 -o "$WORK_DIR/ubuntu.img.xz" \
  "$BASE/ubuntu-24.04-preinstalled-desktop-arm64+dragonwing-x15-<date>.img.xz"
curl -L --fail --retry 3 -o "$WORK_DIR/dtb.bin"         "$BASE/dtb.bin"
curl -L --fail --retry 3 -o "$WORK_DIR/rawprogram0.xml" "$BASE/rawprogram0.xml"

# Boot firmware (from the URL in the YAML)
curl -L --fail --retry 3 -o "$WORK_DIR/boot_fw.tar.gz" \
  'https://artifacts.codelinaro.org/artifactory/qli-ci/flashable-binaries/ubuntu-fw/QCS8300/QLI.1.8-Ver.1.1/QLI.1.8-Ver.1.1-ubuntu-QCS8300-nhlos-bins.tar.gz'
mkdir -p "$WORK_DIR/boot_fw_extract"
tar -xzf "$WORK_DIR/boot_fw.tar.gz" -C "$WORK_DIR/boot_fw_extract"
TOP=$(find "$WORK_DIR/boot_fw_extract" -mindepth 1 -maxdepth 1 -type d | head -1)
cp -a "$TOP"/. "$OUT_DIR"/

# Device tree + top-level rawprogram into the flashable dir
cp -a "$WORK_DIR/dtb.bin" "$WORK_DIR/rawprogram0.xml" "$OUT_DIR"/

# Partition XMLs + GPT (GUID Partition Table) bins live in partition_ufs/ ->
# copy them to the root of the flashable dir (qdl expects them there)
cp "$OUT_DIR/partition_ufs"/rawprogram{1,2,3,4}.xml \
   "$OUT_DIR/partition_ufs"/patch{1,2,3,4}.xml \
   "$OUT_DIR/partition_ufs"/*.bin \
   "$OUT_DIR"/

# Verify the download, extract, and give the .img the EXACT name rawprogram0.xml references
shasum -a 256 "$WORK_DIR/ubuntu.img.xz"    # compare to the SHA256 in the YAML
unxz -k -f "$WORK_DIR/ubuntu.img.xz"
cp "$WORK_DIR/ubuntu.img" \
   "$OUT_DIR/ubuntu-24.04-preinstalled-desktop-arm64+dragonwing-x15-<date>.img"
```

> **Key differences from QLI flashing:** Ubuntu uses **no `patch0.xml`**; the
> `rawprogram1-4` / `patch1-4` XMLs and the GPT `.bin` files live inside
> `partition_ufs/` and must be copied to the flashable root first; and the `.img`
> filename must **exactly** match what `rawprogram0.xml` references.

## 3. Flash (EDL mode) and boot

Flash with Qualcomm's `qdl` tool over USB in **EDL (Emergency Download) mode**:

```bash
# Enter EDL: set DIP switch S2-8 to ON, power-cycle, then confirm the 9008 device:
system_profiler SPUSBDataType | grep -A5 "9008"      # macOS
# (on Linux: lsusb | grep 9008)

cd "$OUT_DIR"
qdl --storage ufs prog_firehose_ddr.elf \
  rawprogram0.xml rawprogram1.xml rawprogram2.xml rawprogram3.xml rawprogram4.xml \
  patch1.xml patch2.xml patch3.xml patch4.xml
# Expected tail: "52 patches applied / partition 1 is now bootable"
```

Set **S2-8 back to OFF** and power-cycle. The first boot takes 2–3 minutes for the
GNOME first-time setup. You get a GNOME desktop on HDMI and an SSH (Secure Shell)
login as `ubuntu` (use the default Dragonwing Ubuntu image password from
Canonical's EVK guide, and change it before putting the board on an untrusted
network).

Bring up networking (the build and the one-time repo clone need outbound
internet), then confirm you're online:

```bash
nmcli dev wifi list
nmcli dev wifi connect <WiFi-SSID> password <WiFi-password>
ping -c1 github.com
```

## 4. Install the NPU runtime stack (Ubuntu-only)

**This is the one step the QLI path doesn't need.** A fresh Canonical Ubuntu image
does **not** ship the FastRPC/QNN host stack — no `libcdsprpc.so`, no DMA (Direct
Memory Access)-heap allocator, no `fastrpc` group on `/dev/fastrpc-cdsp`. The CDI
(§6) forwards those host libraries and device nodes into the container, so they
must exist on the host first. Install them from the Qualcomm IoT PPA:

```bash
sudo add-apt-repository -y ppa:ubuntu-qcom-iot/qcom-ppa
sudo apt update
sudo apt install -y qcom-fastrpc1 qcom-libdmabufheap qairt-libs qairt-dsp-binaries
sudo usermod -aG fastrpc,dmaheap "$USER"      # log out/in for the groups to take effect
```

This brings `/usr/lib/aarch64-linux-gnu/lib{cdsp,adsp}rpc.so*` and
`libdmabufheap.so.0`, creates the `fastrpc` and `dmaheap` groups, and sets
`/dev/fastrpc-cdsp` to `fastrpc:fastrpc 0664`. Verify:

```bash
ls -l /dev/fastrpc-cdsp                       # -> crw-rw-r-- fastrpc fastrpc
ls /usr/lib/aarch64-linux-gnu/libcdsprpc.so*  # -> present
```

## 5. Install Docker

```bash
sudo apt install -y docker.io
sudo usermod -aG docker "$USER"
# log out and back in (or: newgrp docker) so the group takes effect
```

## 6. Fetch the repo, install the CDI, build, and run

The NPU context binaries ship **inside the repo** (`qnn_context/`, ~5 MB), so a
clone has everything — nothing to stage by hand. Ubuntu has `git`, so clone
directly:

```bash
# --- 6.1  Fetch the repo ---
cd ~
git clone https://github.com/munoz0raul/dorbell.git
cd dorbell

# --- 6.2  Install the Ubuntu CDI (writes /etc/cdi/cdi-hw-acc.json; prunes absent libs) ---
cd ~/dorbell/container
sudo rm -f /etc/cdi/cdi-hw-acc.json                 # the installer refuses to overwrite
sudo python3 cdi/install_cdi.py --file cdi/cdi-hw-acc-8275.json
# After §4 this should report NO missing mounts. If it prunes libcdsprpc.so /
# libdmabufheap.so.0, you skipped §4 -- install the PPA stack and re-run.

# --- 6.3  Build the image (from the repo root, so doorbell.py is in the build context) ---
cd ~/dorbell
docker build -t dragonwing-doorbell -f container/Dockerfile .
# Use the PLAIN build -- NO --build-arg. The default 2.50.x qai-appbuilder
# matches the committed .bin and loads on this 2.46-firmware board.

# --- 6.4  Run the NPU demo (full-screen on HDMI) ---
cd ~/dorbell
bash container/run_doorbell_container.sh npu
```

The build takes a few minutes the first time (it pulls the aarch64 Python wheels —
insightface, onnxruntime, opencv, and the 2.50.x `qai-appbuilder` QNN runtime from
Qualcomm's public GitHub release, into a python3.10 venv — see
[`container/README.md`](container/README.md) for the GLIBC + QAIRT-version
rationale). The run wrapper
auto-detects everything board-specific: the GNOME/Xwayland display cookie, the USB
webcam's `/dev/video*` node, and the `fastrpc`/`dmaheap`/`video` group IDs — and it
runs the container as **root** (unlike the QLI path, which runs as the Weston uid).

## 7. Confirm the NPU is actually running

On a healthy run the HTP engine reports a **real** multi-hundred-millisecond load
(not the ~4 ms CPU-fallback fake), with **no** `Using newer context binary` / `err
5000` / `0x72` lines:

```
NPU context binaries: using repo-committed /home/ubuntu/dorbell/qnn_context
[Camera] Opened /dev/video0
[NPU] in-process HTP engine ready  (load 230ms)
[NPU] det_ctx=/root/qnn_context/det_500m_qcs8275.bin
[NPU] rec_ctx=/root/qnn_context/w600k_mbf_qcs8275.bin
```

Definitive liveness check from a second shell — the container process should hold
the FastRPC node and a DMA buffer open:

```bash
pid=$(sudo docker inspect -f '{{.State.Pid}}' dragonwing-doorbell)
sudo ls -l /proc/$pid/fd | grep -E 'fastrpc|dma_heap'   # -> /dev/fastrpc-cdsp, /dev/dma_heap/system
```

The window is full-screen on HDMI with the live camera feed. Controls (`O` enroll,
`B` CPU/NPU toggle, `[`/`]` idle timeout, `q`/ESC quit) are documented in the main
[README](README.md) §2.5.

To stop the demo:

```bash
docker rm -f dragonwing-doorbell
```

---

## Troubleshooting

The container deltas vs bare metal (display auth, camera node, the HTP enablement
barriers) are in [`container/README.md`](container/README.md). The three most
common Ubuntu first-run snags:

- **`install_cdi.py` prunes `libcdsprpc.so` / `libdmabufheap.so.0`.** You skipped
  §4 — the Canonical image ships without the FastRPC/QNN runtime. Install the PPA
  stack and re-run the installer (`sudo rm -f /etc/cdi/cdi-hw-acc.json` first).
- **`[NPU] ... ready` but `Using newer context binary on old SDK` / `err 5000` /
  `Failed to create context from binary 0x1388` above it.** The image's
  `qai-appbuilder` host SDK is **older** than the SDK that generated the `.bin`
  (2.50). Build **plainly** with no `QAI_APPBUILDER_WHEEL` override — the
  default 2.50.x wheel is correct for the committed binaries. (This is independent of
  the board's 2.46 firmware.) Trust the `load Nms` figure and the **absence** of
  those error lines, not the "ready" string.
- **`[Camera] No camera — using synthetic frames` with a webcam plugged in.**
  Check `v4l2-ctl --list-devices` for the cam's real node; the wrapper passes the
  low-numbered `/dev/video*` nodes and the app probes all present ones.
