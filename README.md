# randotools

A collection of tools and scripts. The main component is the **RealESRGAN Upscaler** — a
PyQt6-based image and video upscaling application with NVIDIA and AMD GPU support.

Also included: **[NACA Inlet](naca_inlet/README.md)** — import a STEP surface and cut a
parametric NACA submerged inlet at a given location, sized by throat area.

## Quick Start

```bash
chmod +x fedora_setup.sh
./fedora_setup.sh          # auto-detect GPU (Fedora 43)
source ~/.venv/realesrgan/bin/activate
realesrgan-upscaler         # launch GUI
```

See [upscaler/README.md](upscaler/README.md) for full documentation.
