# RealESRGAN Upscaler

PyQt6 desktop app for image and video upscaling with Real-ESRGAN models. The app is optimized for Fedora and supports NVIDIA CUDA, AMD ROCm, and CPU-only installs.

Python 3.10 or newer is required. For AMD ROCm wheels, Python 3.10-3.12 is recommended.

## Current Architecture

- `upscaler/gui/main_window.py` provides the PyQt6 interface.
- `upscaler/core/` contains the GUI-independent job, workspace, ffmpeg, and video pipeline code.
- Video jobs use per-job cache workspaces under `~/.cache/realesrgan-upscaler/jobs/`.
- Downloaded model weights are stored under `~/.local/share/realesrgan-upscaler/weights/` by default.
- Packaged `upscaler/weights/` files are treated as read-only fallback weights.

## Quick Start

From the repository root:

```bash
chmod +x fedora_setup.sh
./fedora_setup.sh
source ~/.venv/realesrgan/bin/activate
realesrgan-upscaler
```

The setup script installs system packages, chooses the appropriate PyTorch build for your GPU mode, installs app dependencies, and creates a desktop launcher.

## Manual Development Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision
pip install -e "upscaler[cpu]"
realesrgan-upscaler
```

For CUDA or ROCm, install the matching PyTorch wheel before `pip install -e upscaler`:

```bash
# NVIDIA CUDA example
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# AMD ROCm example
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

## Video Upscaling Notes

Video support requires `ffmpeg` and `ffprobe`.

The rebuilt video path currently processes one video job at a time. Image batches remain supported separately. This keeps cancellation, per-job cache state, resume behavior, and progress reporting accurate.

Video output options include:

- Container: `mp4` or `mkv`
- Audio: copy original audio or re-encode to AAC
- Optional preservation of upscaled intermediate frames

## Model Weights

Use **Tools > Download Models** in the GUI. Downloads are written to:

```text
~/.local/share/realesrgan-upscaler/weights/
```

You can override this location in the config file with `weights_dir`.

## RPM Packaging

The RPM path is intentionally a thin launcher/app-file package. It does not run networked `pip install` during package installation because PyTorch wheels vary by GPU and should be installed through `fedora_setup.sh` or a managed virtual environment.

Build from the repository root:

```bash
chmod +x build_rpm.sh
./build_rpm.sh
```

## Tests

```bash
python -m pytest tests
```

The core pipeline tests mock ffmpeg and do not require a GPU or real media files.
