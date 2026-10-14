#!/bin/bash
# Run the Dragonwing Doorbell as a Docker container on the IQ-8275 board.
#
# This is the container analogue of run_doorbell.sh: same display keep-awake
# setup, same single-instance guard, same cpu|npu arg -- but the app runs inside
# the `dragonwing-doorbell` image with the NPU reached through the CDI.
#
# Prereqs (one-time, see container/README.md):
#   - Docker installed, user in the `docker` group.
#   - CDI installed:  sudo python3 cdi/install_cdi.py --file cdi/cdi-hw-acc-8275.json
#   - Image built:    docker build -t dragonwing-doorbell -f container/Dockerfile .
#   - Models present on the host under $HOME (mounted read-only below):
#       ~/qnn_context/{det_500m,w600k_mbf}_qcs8275.bin     (NPU HTP context)
#       ~/.insightface/models/buffalo_sc/*.onnx            (CPU fallback)
#
# Usage: bash run_doorbell_container.sh [cpu|npu]   (default: npu)

set -euo pipefail
BACKEND=${1:-npu}
IMAGE=dragonwing-doorbell
NAME=dragonwing-doorbell

# --- Resolve the real user's home (survive `sudo` / `sudo su`) -----------------
# The models live under the login user's $HOME (~/qnn_context, ~/.insightface),
# and the Xwayland cookie/socket belong to that user's session. Under `sudo su`
# $HOME flips to /root, where those paths are empty -- the models silently fail
# to mount and insightface tries to re-download into a read-only mount. So pin
# $HOME to the invoking user, not root, regardless of how we were launched.
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
elif [ "$(id -u)" = 0 ] && [ -d /home/ubuntu ]; then
    USER_HOME=/home/ubuntu          # plain `su -` loses SUDO_USER; fall back
else
    USER_HOME=$HOME
fi
if [ ! -d "$USER_HOME/qnn_context" ] && [ ! -d "$USER_HOME/.insightface" ]; then
    echo "WARNING: no models under $USER_HOME (qnn_context/.insightface). Run as the" >&2
    echo "         login user, or put the models there. NPU will fail to load." >&2
fi

# --- Host display: same keep-awake dance as the bare-metal launcher -----------
# Done on the HOST (not in the container): this is the GNOME/Xwayland session
# that owns the HDMI output. The container just draws into its socket.
AUTH=$(ls /run/user/1000/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
if [ -z "$AUTH" ]; then
    echo "WARNING: no Xwayland auth cookie under /run/user/1000 -- is the desktop up?" >&2
fi
export DISPLAY=:0
gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null || true
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false 2>/dev/null || true
gsettings set org.gnome.desktop.screensaver lock-enabled false 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' 2>/dev/null || true
gsettings set org.gnome.settings-daemon.plugins.power idle-dim false 2>/dev/null || true
xset s off 2>/dev/null || true; xset s noblank 2>/dev/null || true

# --- Single instance: remove any prior container before starting --------------
docker rm -f "$NAME" >/dev/null 2>&1 || true

# --- Group IDs for the device nodes -------------------------------------------
# /dev/fastrpc-cdsp and the DMA heaps are group-restricted on the host; the
# container process must be in those groups to open them. GIDs are board-local
# so we resolve them at runtime rather than hard-coding.
GROUP_ARGS=()
for grp in fastrpc dmaheap video; do
    gid=$(getent group "$grp" | cut -d: -f3)
    [ -n "${gid:-}" ] && GROUP_ARGS+=(--group-add "$gid")
done

# --- USB camera device nodes --------------------------------------------------
# The app probes video0..5 and falls back to synthetic frames, so passing the
# low-numbered nodes that exist is enough (video32/33 are the ISP, not the cam).
CAM_ARGS=()
for n in 0 1 2 3; do
    [ -e "/dev/video$n" ] && CAM_ARGS+=(--device "/dev/video$n")
done

echo "Starting container '$NAME' with backend=$BACKEND ..."
exec docker run --rm --name "$NAME" \
    --device qualcomm.com/device=cdi-hw-acc \
    "${CAM_ARGS[@]}" \
    "${GROUP_ARGS[@]}" \
    -e DISPLAY=:0 \
    -e PYTHONUNBUFFERED=1 \
    -e XDG_RUNTIME_DIR=/run/user/1000 \
    -e XAUTHORITY=/tmp/xauth \
    -v "$AUTH":/tmp/xauth:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /run/user/1000:/run/user/1000 \
    -v "$USER_HOME/qnn_context":/root/qnn_context:ro \
    -v "$USER_HOME/.insightface":/root/.insightface:ro \
    "$IMAGE" --backend "$BACKEND"
