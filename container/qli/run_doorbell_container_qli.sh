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
#      the ubuntu:24.04 container expects, and only one DMA-heap node exists
#      (/dev/dma_heap/system). The image also needs libyaml-0-2 because QLI's
#      libcdsprpc.so links libyaml-0.so.2 (already in container/Dockerfile).
#
# Prereqs (one-time, see container/qli/README.md):
#   - CDI installed:  python3 ../cdi/install_cdi.py --file cdi-hw-acc-8275-qli.json
#   - Image built:    docker build -t dragonwing-doorbell -f ../Dockerfile <repo-root>
#   - Models present on the host (any readable location; default /root):
#       $MODELS_HOME/qnn_context/{det_500m,w600k_mbf}_qcs8275.bin   (NPU HTP)
#       $MODELS_HOME/.insightface/models/buffalo_sc/*.onnx          (CPU)
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
MODELS_HOME=${MODELS_HOME:-$HOME}
if [ ! -d "$MODELS_HOME/qnn_context" ] && [ ! -d "$MODELS_HOME/.insightface" ]; then
    echo "WARNING: no models under $MODELS_HOME (qnn_context/.insightface)." >&2
    echo "         NPU needs qnn_context/*.bin; CPU needs .insightface/models/buffalo_sc/*.onnx" >&2
fi

# --- Single instance ----------------------------------------------------------
docker rm -f "$NAME" >/dev/null 2>&1 || true

# --- Group IDs for the device nodes (board-local, resolved at runtime) --------
GROUP_ARGS=()
for grp in fastrpc dmaheap video; do
    gid=$(getent group "$grp" | cut -d: -f3)
    [ -n "${gid:-}" ] && GROUP_ARGS+=(--group-add "$gid")
done

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
exec docker run --rm --name "$NAME" \
    --user "${WESTON_UID}:${WESTON_UID}" \
    --device qualcomm.com/device=cdi-hw-acc \
    "${CAM_ARGS[@]}" \
    "${GROUP_ARGS[@]}" \
    -e DISPLAY=:0 \
    -e PYTHONUNBUFFERED=1 \
    -e HOME=/models \
    -e XDG_CONFIG_HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp \
    -e XDG_RUNTIME_DIR=/run/user/"${WESTON_UID}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$MODELS_HOME/qnn_context":/models/qnn_context:ro \
    -v "$MODELS_HOME/.insightface":/models/.insightface:ro \
    "$IMAGE" --backend "$BACKEND"
