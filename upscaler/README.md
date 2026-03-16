# RealESRGAN Upscaler GUI

A modern, feature-rich PyQt6-based image upscaler for Fedora Linux using Real-ESRGAN models.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OS](https://img.shields.io/badge/OS-Fedora%2043-red)

## Features

✨ **Core Features**
- 🖼️ Batch image upscaling with 2x and 4x scale factors
- ⚡ Multi-GPU support with automatic load balancing
- ⏸️ Pause/resume functionality for long-running jobs
- 🎨 Multiple model selection (Standard quality, Anime optimized)
- 📊 Real-time GPU monitoring and system diagnostics
- 🔧 Half-precision (FP16) support for faster processing
- 💾 Configuration persistence

🐧 **Fedora-Specific**
- 🎯 Optimized for Fedora 43 and later
- 📦 RPM package support
- 🔧 One-click setup helpers for:
  - ROCm driver installation
  - GPU permission fixes
  - Dependency installation
- 🏥 System diagnostics for AMD GPU troubleshooting

🎮 **GUI Features**
- 💻 Intuitive tabbed interface with 4 main sections:
  - **Upscaler**: Main processing interface
  - **Settings**: Configuration and setup helpers
  - **System Monitor**: Real-time GPU and system stats
  - **Diagnostics**: System information and troubleshooting

## Quick Start

### Installation

```bash
# Install system dependencies
sudo dnf install -y python3-pip python3-devel gcc-c++ git

# Clone and install
git clone https://github.com/mikelongjr/randotools.git
cd randotools/upscaler
pip install --user -e .

# For AMD GPU support (recommended)
sudo dnf install rocm-hip rocm-opencl
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
sudo usermod -aG render,video $USER
# Then reboot or: newgrp render video
```

### Launch

```bash
realesrgan-upscaler-gui
```

## Detailed Documentation

- 📖 [Installation Guide](INSTALL.md) - Complete setup instructions
- 🚀 [Getting Started](docs/GETTING_STARTED.md) - Usage tutorial
- 🔧 [Configuration Guide](docs/CONFIGURATION.md) - Advanced settings
- 🐛 [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and fixes

## Screenshots

*GUI Interface (Coming soon)*

## System Requirements

### Minimum
- **OS**: Fedora 43+
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB for models

### Recommended
- **GPU**: NVIDIA, AMD (ROCm), or Apple Silicon
- **RAM**: 8GB+
- **Storage**: SSD for faster processing

## Installation Methods

### Method 1: User Install (Recommended)
```bash
git clone https://github.com/mikelongjr/randotools.git
cd randotools/upscaler
pip install --user -e .
```

### Method 2: System-wide via RPM
```bash
sudo dnf install rpmbuild fedora-packager
cd randotools/upscaler
rpmbuild -ba realesrgan-upscaler.spec
sudo dnf install ~/rpmbuild/RPMS/noarch/realesrgan-upscaler-gui-2.0.0-1.fc43.noarch.rpm
```

### Method 3: Development Mode
```bash
git clone https://github.com/mikelongjr/randotools.git
cd randotools/upscaler
python3 -m venv venv
source venv/bin/activate
pip install -e .
python3 -m gui.upscaler_app
```

## Usage

### Basic Workflow
1. Launch the application: `realesrgan-upscaler-gui`
2. Select input directory (images to upscale)
3. Select output directory (where upscaled images will be saved)
4. Choose model type and scale factor
5. Click "Start Processing"

### Advanced Options
- **Pause/Resume**: Stop processing temporarily and continue later
- **GPU Selection**: Choose specific GPU device
- **Worker Threads**: Optimize for your hardware
- **Half-Precision**: Trade accuracy for speed
- **Resume on Startup**: Automatically continue interrupted jobs

### Settings
Access via the "Settings" tab to:
- Configure GPU device and worker count
- Install ROCm drivers
- Fix GPU permissions
- Install missing dependencies
- Configure application preferences

### Monitoring
Monitor system performance in "System Monitor" tab:
- GPU status and name
- Memory usage
- Real-time utilization (during processing)

### Diagnostics
Troubleshoot issues in "Diagnostics" tab:
- System information
- PyTorch and GPU details
- Group permission status
- Device node availability

## Models

### Standard (RealESRGAN_x4plus)
- **Quality**: Highest quality upscaling
- **Speed**: Slower
- **Use Case**: Photography, realistic images
- **Scale Factors**: 2x, 3x, 4x

### Anime (RealESRGAN_x4plus_anime_6B)
- **Quality**: Optimized for anime art
- **Speed**: Faster (6B parameters vs 68M)
- **Use Case**: Anime, manga, anime-style art
- **Scale Factors**: 4x
- **Requirements**: BasicSR library

## GPU Support

### NVIDIA GPUs
```bash
# CUDA support is included in standard PyTorch
pip install --user torch torchvision torchaudio
```

### AMD GPUs (Recommended on Fedora)
```bash
# Install ROCm
sudo dnf install rocm-hip rocm-opencl

# Install PyTorch with ROCm
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Fix permissions
sudo usermod -aG render,video $USER
```

### CPU-only
```bash
# No special installation needed
pip install --user torch torchvision torchaudio
```

## Configuration

Configuration is saved to: `~/.config/realesrgan-upscaler/config.json`

Example:
```json
{
  "input_dir": "/path/to/input",
  "output_dir": "/path/to/output",
  "model_type": "standard",
  "scale_factor": 4,
  "num_workers": 1,
  "use_half_precision": true,
  "resume_on_startup": true,
  "gpu_device": 0
}
```

## Performance Tips

1. **Use Half-Precision**: Provides ~2x speedup (enabled by default)
2. **Batch Processing**: Process multiple images for better GPU utilization
3. **SSD Storage**: Use SSD for input/output directories
4. **Multi-GPU**: Increase workers if you have multiple GPUs
5. **Model Selection**: Use Anime model only for anime images

## Troubleshooting

### GPU Not Detected
```bash
# Fix permissions
sudo usermod -aG render,video $USER

# For AMD GPUs with specific GFX versions
HSA_OVERRIDE_GFX_VERSION=10.3.0 realesrgan-upscaler-gui
```

### PyTorch Import Error
```bash
# Reinstall PyTorch for your GPU
pip install --force-reinstall --user torch torchvision
```

### BasicSR Import Error
```bash
# Install without GPU extensions
CUDA_VISIBLE_DEVICES='' pip install --user basicsr
```

For more troubleshooting, see [INSTALL.md](INSTALL.md#troubleshooting)

## Architecture

### Project Structure
```
upscaler/
├── gui/
│   ├── __init__.py
│   └── upscaler_app.py          # Main PyQt6 application
├── upscale_frames.py             # CLI upscaler (legacy)
├── temperature_watcher.py        # GPU temperature monitoring
├── setup.py                      # Package setup
├── realesrgan-upscaler.spec     # RPM spec file
├── requirements.txt              # Python dependencies
├── INSTALL.md                    # Installation guide
└── README.md                     # This file
```

### Dependencies
- **PyQt6**: Modern GUI framework
- **PyTorch**: Deep learning framework
- **RealESRGAN**: Upscaling model implementation
- **Pillow**: Image processing
- **BasicSR**: Advanced architecture definitions
- **OpenCV**: Additional image utilities

## Development

### Setup Development Environment
```bash
git clone https://github.com/mikelongjr/randotools.git
cd randotools/upscaler
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests
```bash
python3 -m pytest tests/
```

### Building Documentation
```bash
cd docs/
make html
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - See [LICENSE](LICENSE) file for details

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for the upscaling models
- [BasicSR](https://github.com/xinntao/BasicSR) for architecture implementations
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the GUI framework

## Support

### Getting Help
1. Check the [Diagnostics tab](INSTALL.md#diagnostics-tab) in the app
2. Review the [Installation Guide](INSTALL.md)
3. Check [Troubleshooting section](INSTALL.md#troubleshooting)
4. Open an [issue on GitHub](https://github.com/mikelongjr/randotools/issues)

### Reporting Bugs
Please include:
- Fedora version
- GPU model
- Error messages from the Processing Log
- Output from the Diagnostics tab

## Changelog

### Version 2.0.0 (2026-03-16)
- ✨ Complete GUI rewrite with PyQt6
- ✨ Real-time GPU monitoring
- ✨ Pause/resume functionality
- ✨ System diagnostics panel
- ✨ Fedora-specific setup helpers
- ✨ Configuration persistence
- ✨ RPM package support

### Version 1.0.0 (Original)
- CLI-based upscaler
- Multi-GPU support
- ROCm compatibility
