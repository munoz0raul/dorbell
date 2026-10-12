#!/bin/bash
# Usage: bash run_doorbell.sh [cpu|npu]   (default: cpu)
# Also launched by the "Dragonwing Doorbell (NPU)" desktop icon.
BACKEND=${1:-cpu}
AUTH=$(ls /run/user/1000/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
export DISPLAY=:0
export XAUTHORITY=$AUTH
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export PATH=$PATH:/home/ubuntu/.local/bin

# --- Single instance: stop any prior doorbell before starting -----------------
# Clicking the desktop icon twice (or launching while one already runs) would
# otherwise have two processes fight over the camera and the display.
pkill -f "doorbell_prototype.py" 2>/dev/null
pkill -f "python3 /home/ubuntu/doorbell.py" 2>/dev/null
sleep 1

# --- Keep the HDMI display awake for the whole demo ----------------------------
# This Xwayland server has no DPMS extension, so GNOME's idle manager is what
# blanks the screen; these gsettings are the ones that actually matter. They
# persist in dconf across reboots, but we re-assert on every launch so the demo
# board is always demo-ready regardless of prior state.
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver idle-activation-enabled false
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power idle-dim false
xset s off 2>/dev/null; xset s noblank 2>/dev/null   # harmless if unsupported

echo "Starting doorbell with backend=$BACKEND ..."
python3 /home/ubuntu/doorbell.py --backend $BACKEND
