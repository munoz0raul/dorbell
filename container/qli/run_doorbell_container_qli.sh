#!/bin/bash
# Run the Dragonwing Doorbell as a Docker container on the IQ-8275 under
# **Qualcomm Linux (QLI) 2.1** -- the Yocto reference distro, NOT Ubuntu.
#
# The container image is identical to the Ubuntu one (ubuntu:24.04 base,
# portable); only the HOST plumbing differs. The three QLI-specific mechanics,
# all verified on an IQ-8275 EVK running QLI 2.1 (2026-10-14):
#
#   1. DISPLAY is **Weston** (Wayland), not GNOME/Xwayland. Weston's bundled
#      Xwayland owns the X11 socket /tmp/.X11-unix/X0 as uid 1000 (user
#      `weston`) and authorizes clients by **matching UID over the socket** --
#      there is NO .Xauthority cookie to mount (GNOME has one; Weston does not).
#      So the container must run as **--user 1000:1000** to be allowed to draw.
#      That is why we do not hunt for a cookie and do not touch `xhost`.
#   2. Running as uid 1000 means $HOME is no longer /root, but doorbell.py finds
#      its models via `expanduser("~/qnn_context")` / `~/.insightface`. We set
#      HOME=/models inside the container and mount both model trees under it, so
#      the lookup resolves regardless of the numeric uid. Qt/cache writes are
#      redirected to /tmp (the uid-1000 process cannot write elsewhere).
#   3. Host libs are flat under /usr/lib (not /usr/lib/aarch64-linux-gnu); the
#      QLI CDI (cdi-hw-acc-8275-qli.json) bind-maps them to the multiarch paths
#      the ubuntu:24.04 container expects. The image also needs libyaml-0-2
#      because QLI's libcdsprpc.so links libyaml-0.so.2 (already in
#      container/Dockerfile).
#
# NPU/HTP enablement -- three things beyond the CDI's lib mounts, all verified on
# the IQ-8275 QLI 2.1 board (2026-10-14) by isolating with `fastrpc_test -d 3`:
#
#   A. Device nodes are passed as **plain `docker run --device`**, NOT via the
#      CDI. The CDI's deviceNodes inject /dev/dma_heap/system and /dev/fastrpc-cdsp
#      as **gid 0**, so `--group-add dmaheap`/`fastrpc` can't grant access and the
#      uid-1000 process is denied the DMA heap (FastRPC ION alloc fails -> no DSP
#      session). Docker's plain `--device` **preserves the host gid** (dmaheap=993,
#      fastrpc=991), which `--group-add` then unlocks. So the CDI carries only the
#      lib mounts; the nodes live here.
#   B. The board DSP runtime dir (holding fastrpc_shell_unsigned_3 + the matching
#      v75 QNN skel) is bind-mounted onto **/usr/lib/dsp/cdsp**. The FastRPC lib
#      normally derives the board-specific path from /sys/firmware/devicetree/
#      base/model, which is absent in the container; it then only probes a few
#      HARDCODED dirs (/usr/lib/dsp/cdsp among them) -- DSP_LIBRARY_PATH is not
#      consulted for the shell. Without this the DSP can't spawn the user PD
#      (openSessionForPriority status=0x72, "Failed to load skel 1002").
#   C. qai-appbuilder is pinned to the version whose bundled QNN matches the board
#      DSP firmware's QAIRT (2.50 here) -- see container/requirements.txt. A
#      mismatch is rejected on-device with the same 0x72 error.
#
# Prereqs (one-time, see container/qli/README.md):
#   - CDI installed:  python3 ../cdi/install_cdi.py --file cdi-hw-acc-8275-qli.json
#   - Image built:    docker build -t dragonwing-doorbell -f ../Dockerfile <repo-root>
#   - Models present on the host (any readable location; default /root):
#       $MODELS_HOME/qnn_context/{det_500m,w600k_mbf}_qcs8275.bin   (NPU HTP)
#       $MODELS_HOME/.insightface/models/buffalo_sc/*.onnx          (CPU; auto-
#         downloaded on first `B`->CPU toggle if absent -- the wrapper creates a
#         writable ~/.insightface for the uid-1000 container to download into)
#
# Usage: bash run_doorbell_container_qli.sh [cpu|npu]   (default: npu)
#   Env overrides: WESTON_UID (default auto-detected), MODELS_HOME (default: the
#   invoking user's home, i.e. /root when run as root).

set -euo pipefail
BACKEND=${1:-npu}
IMAGE=dragonwing-doorbell
NAME=dragonwing-doorbell

# --- Who owns the Weston/Xwayland display socket? -----------------------------
# The container must run as that uid to pass Weston's same-UID X11 auth. Detect
# it from the X0 socket owner; fall back to 1000 (the QLI `weston` user).
if [ -S /tmp/.X11-unix/X0 ]; then
    WESTON_UID=${WESTON_UID:-$(stat -c %u /tmp/.X11-unix/X0 2>/dev/null || echo 1000)}
else
    WESTON_UID=${WESTON_UID:-1000}
    echo "WARNING: /tmp/.X11-unix/X0 not present -- is Weston up? (systemctl status weston)" >&2
fi

# --- Models: mounted under /models, which we make the container's HOME --------
# The NPU HTP context binaries (*.bin) are COMMITTED to the repo under
# qnn_context/ (only ~5 MB), so a fresh `git clone` reproduces the NPU demo with
# zero model staging. Prefer a user-staged $MODELS_HOME/qnn_context if present
# (lets you point at freshly regenerated .bin); otherwise fall back to the
# repo-committed copy. The CPU .insightface ONNX models are NOT committed --
# insightface auto-downloads buffalo_sc on first use; override MODELS_HOME to a
# dir holding .insightface/models/buffalo_sc if you need the CPU backend offline.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MODELS_HOME=${MODELS_HOME:-$HOME}
if [ -d "$MODELS_HOME/qnn_context" ]; then
    QNN_CONTEXT_DIR="$MODELS_HOME/qnn_context"
elif [ -d "$REPO_ROOT/qnn_context" ]; then
    QNN_CONTEXT_DIR="$REPO_ROOT/qnn_context"
    echo "NPU context binaries: using repo-committed $QNN_CONTEXT_DIR"
else
    QNN_CONTEXT_DIR="$MODELS_HOME/qnn_context"   # non-existent; NPU will warn below
fi
if [ ! -d "$QNN_CONTEXT_DIR" ] && [ ! -d "$MODELS_HOME/.insightface" ]; then
    echo "WARNING: no models found (qnn_context/.insightface)." >&2
    echo "         NPU needs qnn_context/*.bin; CPU needs .insightface/models/buffalo_sc/*.onnx" >&2
fi

# --- Single instance ----------------------------------------------------------
docker rm -f "$NAME" >/dev/null 2>&1 || true

# --- Group IDs for the device nodes (board-local, resolved at runtime) --------
# These unlock the host gids that plain `--device` passthrough preserves (see
# NPU note A at the top): dmaheap for the ION heap, fastrpc for the DSP node.
GROUP_ARGS=()
for grp in fastrpc dmaheap video; do
    gid=$(getent group "$grp" | cut -d: -f3)
    [ -n "${gid:-}" ] && GROUP_ARGS+=(--group-add "$gid")
done

# --- NPU device nodes: plain --device so the host gid survives (NPU note A) ----
# The CDI would inject these as gid 0, defeating --group-add. Passing them here
# as plain --device preserves dmaheap(993)/fastrpc(991) ownership in-container.
NPU_DEV_ARGS=()
for node in /dev/dma_heap/system /dev/fastrpc-cdsp; do
    [ -e "$node" ] && NPU_DEV_ARGS+=(--device "$node")
done

# --- Board DSP runtime dir -> mounted at the hardcoded /usr/lib/dsp/cdsp -------
# (NPU note B.) Auto-detect any .../dsp/cdsp dir that holds BOTH a fastrpc shell
# and the v75 QNN skel; mount it where the in-container FastRPC lib will look.
DSP_MOUNT_ARGS=()
DSP_CDSP_DIR=""
for d in $(find /usr/share/qcom -type d -path '*/dsp/cdsp' 2>/dev/null); do
    if [ -e "$d/fastrpc_shell_unsigned_3" ] && [ -e "$d/libQnnHtpV75Skel.so" ]; then
        DSP_CDSP_DIR="$d"; break
    fi
done
if [ -n "$DSP_CDSP_DIR" ]; then
    DSP_MOUNT_ARGS+=(-v "$DSP_CDSP_DIR":/usr/lib/dsp/cdsp:ro)
else
    echo "WARNING: no board DSP cdsp dir found under /usr/share/qcom (fastrpc_shell" >&2
    echo "         + libQnnHtpV75Skel.so). NPU/HTP will fall back to CPU." >&2
fi

# --- USB camera device nodes --------------------------------------------------
# On QLI /dev/video0..27 are the on-SoC Iris codec + ISP/CSI pipelines, NOT a
# USB webcam. A UVC webcam, when plugged in, enumerates as a higher /dev/video*
# node backed by a usb device. Pass through every video node that is USB-backed;
# the app probes them and falls back to synthetic frames if none open.
CAM_ARGS=()
for dev in /dev/video*; do
    [ -e "$dev" ] || continue
    n=${dev#/dev/video}
    link=$(readlink -f "/sys/class/video4linux/video$n" 2>/dev/null || true)
    case "$link" in
        *usb*) CAM_ARGS+=(--device "$dev") ;;   # USB UVC cam
    esac
done
# If nothing USB-backed was found, pass the low nodes anyway (harmless; lets a
# differently-wired cam be probed). Comment: synthetic frames are the fallback.
if [ ${#CAM_ARGS[@]} -eq 0 ]; then
    for n in 0 1 2 3; do
        [ -e "/dev/video$n" ] && CAM_ARGS+=(--device "/dev/video$n")
    done
    echo "NOTE: no USB webcam detected; passing on-SoC video nodes. If no cam" >&2
    echo "      opens, the app uses synthetic frames (plug in a UVC webcam)." >&2
fi

echo "Starting container '$NAME' backend=$BACKEND as uid=$WESTON_UID (QLI/Weston host) ..."
[ -n "$DSP_CDSP_DIR" ] && echo "  DSP runtime: $DSP_CDSP_DIR -> /usr/lib/dsp/cdsp"

# --- Model mounts: NPU .bin (repo-committed or user-staged) + CPU ONNX ---------
# The CPU buffalo_sc ONNX models are NOT committed (only the NPU .bin are), so on
# a fresh board they are absent. InsightFace auto-downloads them on first CPU use
# -- but the container runs as uid 1000 with HOME=/models, so its ~/.insightface
# must be a WRITABLE host dir owned-or-world-writable for that download to land
# (a read-only or root-owned mount is why `B`->CPU failed with
# `Permission denied: '/models/.insightface'`). Create it if absent, make it
# writable by the container uid, and mount it read-write. First CPU toggle then
# downloads buffalo_sc (needs internet); subsequent runs reuse it. To preseed
# offline, drop the ONNX files under $MODELS_HOME/.insightface/models/buffalo_sc.
INSIGHTFACE_DIR="$MODELS_HOME/.insightface"
mkdir -p "$INSIGHTFACE_DIR/models"
chmod -R a+rwX "$INSIGHTFACE_DIR" 2>/dev/null || true

MODEL_MOUNT_ARGS=()
[ -d "$QNN_CONTEXT_DIR" ] && MODEL_MOUNT_ARGS+=(-v "$QNN_CONTEXT_DIR":/models/qnn_context:ro)
MODEL_MOUNT_ARGS+=(-v "$INSIGHTFACE_DIR":/models/.insightface)   # rw: CPU self-download

exec docker run --rm --name "$NAME" \
    --user "${WESTON_UID}:${WESTON_UID}" \
    --device qualcomm.com/device=cdi-hw-acc \
    "${NPU_DEV_ARGS[@]}" \
    "${DSP_MOUNT_ARGS[@]}" \
    "${CAM_ARGS[@]}" \
    "${GROUP_ARGS[@]}" \
    -e DISPLAY=:0 \
    -e PYTHONUNBUFFERED=1 \
    -e HOME=/models \
    -e XDG_CONFIG_HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp \
    -e XDG_RUNTIME_DIR=/run/user/"${WESTON_UID}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    "${MODEL_MOUNT_ARGS[@]}" \
    "$IMAGE" --backend "$BACKEND"
