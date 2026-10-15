# Installation — IQ-8275 EVK + QLI 2.1 (`qcom-multimedia` image, either variant)

This is the exact install for **this demo's reference configuration**: a Qualcomm
Dragonwing **IQ-8275 EVK** flashed with the **Qualcomm Linux (QLI) 2.1**
`qcom-distro-sota` multimedia image, a USB (Universal Serial Bus) webcam, and an
HDMI (High-Definition Multimedia Interface) display. It takes you from a bare
board to the containerized NPU (Neural Processing Unit) doorbell running
full-screen.

Every command below was verified end-to-end on a freshly reflashed IQ-8275 EVK
running QLI 2.1 (DSP firmware QAIRT 2.50.0, HTP v75) — **on both multimedia image
variants** (see below).

---

## 1. Download the image — either variant works

**You can run this demo on either IQ-8275 multimedia image.** Both were verified
end-to-end with the identical commands in §4; pick either one and download it from
the CodeLinaro QLI CI artifacts:

| Image | Direct download |
|-------|-----------------|
| **Standard** multimedia | https://artifacts.codelinaro.org/artifactory/qli-ci/flashable-binaries/meta-qcom/qcom-distro-sota/iq-8275-evk/qli-2.1.1-qcom-multimedia-image.zip |
| **Proprietary** multimedia | https://artifacts.codelinaro.org/artifactory/qli-ci/flashable-binaries/meta-qcom/qcom-distro-sota/iq-8275-evk/qli-2.1.1-qcom-multimedia-proprietary-image.zip |

Both ship the closed-source Hexagon/FastRPC runtime the NPU needs —
`libcdsprpc.so`, the signed CDSP (Compute DSP) firmware, and
`/usr/share/qcom/.../dsp/cdsp/` with the `fastrpc_shell_*` + `libQnnHtpV75Skel.so`
— so the HTP (Hexagon Tensor Processor) v75 path works out of the box on either.
The two builds differ only in details the run wrapper auto-detects at runtime
(device-node group IDs, the on-SoC DSP directory, the USB camera's `/dev/video*`
node), so **no change to the demo is needed to run on either image.** Unzip
whichever you downloaded:

```bash
unzip qli-2.1.1-qcom-multimedia-image.zip              # standard
# or
unzip qli-2.1.1-qcom-multimedia-proprietary-image.zip  # proprietary
```

## 2. Install (flash) the image

Flash the unzipped image set to the EVK over USB in **EDL (Emergency Download)
mode** using Qualcomm's flashing tool. The exact cabling, EDL-entry button combo,
and flash command are board-specific and documented by Qualcomm — follow the
official IQ-8 EVK flashing guide:

> Qualcomm Linux — *Flash images* (IQ-8 / RB-series EVK):
> https://docs.qualcomm.com/bundle/publicresource/topics/80-70018-254/

In short: put the board in EDL mode, point the flasher (`qdl` / QFIL — Qualcomm
Flash Image Loader) at the `rawprogram*.xml` + `patch*.xml` from the unzipped
image, flash, and reboot. When it comes up you have a Weston desktop on HDMI and
an SSH (Secure Shell) login as `root` (use the default QLI image password from
Qualcomm's EVK guide, and change it before putting the board on an untrusted
network).

> **Note:** reflashing wipes everything on the `otaroot` partition (the rootfs,
> `/root`, Docker images). The DSP runtime is part of the OS image, so it returns
> with the flash — nothing NPU-related needs re-staging by hand. The steps below
> are all you re-run after any reflash.

## 3. Boot and configure Wi-Fi

Log in on the board (serial console or `ssh root@<board-ip>`) and bring up Wi-Fi —
the build and the one-time repo fetch need outbound internet:

```bash
nmcli dev wifi list
nmcli dev wifi connect <WiFi-SSID> password <WiFi-password>
```

Confirm you're online:

```bash
ping -c1 github.com
```

## 4. Fetch the repo, install the CDI, build, and run

The QLI 2.1 image already includes Docker, `curl`, `tar`, and `python3`, but **not
git**, so step one pulls the repo as a tarball. These four steps are the whole
reproduce — the NPU context binaries ship **inside the repo** (`qnn_context/`,
~5 MB), so there is nothing to stage by hand.

```bash
# --- 4.1  Fetch the repo (no git on the board -> tarball from GitHub) ---
cd /root
curl -fsSL https://codeload.github.com/munoz0raul/dorbell/tar.gz/refs/heads/main -o dorbell.tar.gz
tar xzf dorbell.tar.gz
mv dorbell-main dorbell

# --- 4.2  Install the QLI CDI (writes /etc/cdi/cdi-hw-acc.json; prunes absent libs) ---
cd /root/dorbell/container/qli
rm -f /etc/cdi/cdi-hw-acc.json                       # the installer refuses to overwrite
python3 ../cdi/install_cdi.py --file cdi-hw-acc-8275-qli.json

# --- 4.3  Build the image (from the repo root, so doorbell.py is in the build context) ---
cd /root/dorbell
docker build -t dragonwing-doorbell -f container/Dockerfile .

# --- 4.4  Run the NPU demo (full-screen on HDMI) ---
cd /root/dorbell/container/qli
bash run_doorbell_container_qli.sh npu
```

The build takes a few minutes the first time (it pulls the aarch64 Python wheels —
insightface, onnxruntime, opencv, and the 2.50.x `qai-appbuilder` QNN runtime from
Qualcomm's public GitHub release, into a python3.10 venv — see
[`container/README.md`](container/README.md) for the GLIBC + QAIRT-version
rationale). The run wrapper
auto-detects everything board-specific: the Weston display uid, the on-SoC DSP
runtime directory, the USB webcam's `/dev/video*` node, and the
`fastrpc`/`dmaheap`/`video` group IDs.

## 5. Confirm the NPU is actually running

On a healthy run the wrapper prints the committed-binary path and the HTP engine
reports a **real** multi-hundred-millisecond load (not the ~4 ms CPU-fallback
fake), with no `0x72` session errors:

```
NPU context binaries: using repo-committed /root/dorbell/qnn_context
Starting container 'dragonwing-doorbell' backend=npu as uid=1000 (QLI/Weston host) ...
  DSP runtime: /usr/share/qcom/.../dsp/cdsp -> /usr/lib/dsp/cdsp
[Camera] Opened /dev/video28
[NPU] in-process HTP engine ready  (load 253ms)
[NPU] det_ctx=/models/qnn_context/det_500m_qcs8275.bin
[NPU] rec_ctx=/models/qnn_context/w600k_mbf_qcs8275.bin
```

Definitive liveness check from a second shell — the container process should hold
the FastRPC node and DMA (Direct Memory Access) buffers open:

```bash
pid=$(docker inspect -f '{{.State.Pid}}' dragonwing-doorbell)
ls -l /proc/$pid/fd | grep -E 'fastrpc|dma_heap'      # -> /dev/fastrpc-cdsp, /dev/dma_heap/system
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

QLI-specific deltas (Weston display auth, high-numbered USB camera node, the three
NPU/HTP enablement barriers) are documented in
[`container/qli/README.md`](container/qli/README.md). The two most common first-run
snags:

- **`[NPU] in-process HTP engine ready` but inference is slow / falls back to CPU.**
  Trust the `load Nms` figure (hundreds of ms = real HTP) and the **absence** of
  `openSessionForPriority ... status=0x72`. The usual cause on a new board is a
  QAIRT (Qualcomm AI Runtime SDK) version mismatch — verify the firmware is 2.50.x
  (`strings /usr/share/qcom/*/Qualcomm/*/dsp/cdsp/libQnnHtpV75.so | grep v2.5`); the
  image installs a 2.50.x `qai-appbuilder` (Qualcomm's public GitHub v2.50.0 wheel)
  to match.
- **`[Camera] No camera — using synthetic frames` with a webcam plugged in.** On QLI
  the USB cam enumerates high (behind the ISP — Image Signal Processor — nodes);
  check `v4l2-ctl --list-devices`. The app probes all present nodes automatically.
