#!/bin/bash
# Wrapper script to launch RealESRGAN Upscaler from a system-wide RPM installation

APP_DIR="/opt/realesrgan-upscaler"
VENV_DIR="$APP_DIR/venv"
USER_VENV="$HOME/.venv/realesrgan"

if [ -d "$VENV_DIR" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    exec realesrgan-upscaler "$@"
fi

if [ -d "$USER_VENV" ]; then
    # shellcheck source=/dev/null
    source "$USER_VENV/bin/activate"
    exec realesrgan-upscaler "$@"
fi

export PYTHONPATH="$APP_DIR:${PYTHONPATH:-}"
exec python3 -m upscaler "$@"
