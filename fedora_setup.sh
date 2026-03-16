#!/usr/bin/env bash
# =============================================================================
# fedora_setup.sh - Automated setup for RealESRGAN Upscaler on Fedora 43
# =============================================================================
# Usage:
#   chmod +x fedora_setup.sh
#   ./fedora_setup.sh                    # Auto-detect GPU
#   ./fedora_setup.sh --nvidia           # Force NVIDIA CUDA setup
#   ./fedora_setup.sh --amd              # Force AMD ROCm setup
#   ./fedora_setup.sh --cpu-only         # CPU-only mode (no GPU drivers)
# =============================================================================

set -euo pipefail

# ---- Colours ----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---- Configuration ----------------------------------------------------------
VENV_DIR="${HOME}/.venv/realesrgan"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_MODE="auto"   # auto | nvidia | amd | cpu-only

# ---- Argument parsing -------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --nvidia)   GPU_MODE="nvidia"   ;;
        --amd)      GPU_MODE="amd"      ;;
        --cpu-only) GPU_MODE="cpu-only" ;;
        --help|-h)
            echo "Usage: $0 [--nvidia|--amd|--cpu-only]"
            exit 0
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# =============================================================================
# 1. System prerequisites
# =============================================================================
install_system_deps() {
    info "Installing system packages (requires sudo)…"
    sudo dnf install -y \
        python3 python3-pip python3-devel python3-virtualenv \
        python3-qt6 python3-qt6-devel \
        gcc gcc-c++ make \
        git wget curl \
        libGL libGLU mesa-libGL \
        libXext libXrender libXtst libXi \
        xcb-util-image xcb-util-keysyms xcb-util-renderutil xcb-util-wm \
        || true
    success "System packages installed."
}

# =============================================================================
# 2. GPU driver detection and installation
# =============================================================================
detect_gpu() {
    if [[ "$GPU_MODE" != "auto" ]]; then
        echo "$GPU_MODE"
        return
    fi

    if lspci 2>/dev/null | grep -qi 'NVIDIA'; then
        echo "nvidia"
    elif lspci 2>/dev/null | grep -qi 'AMD\|Radeon\|Advanced Micro'; then
        echo "amd"
    else
        echo "cpu-only"
    fi
}

setup_nvidia() {
    info "Setting up NVIDIA drivers and CUDA…"

    # Check if nvidia-smi is already available
    if command -v nvidia-smi &>/dev/null; then
        success "NVIDIA driver already installed: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        return
    fi

    warn "NVIDIA drivers not found. Attempting installation via RPM Fusion…"
    # Enable RPM Fusion Free and Non-Free
    sudo dnf install -y \
        "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm" \
        || true

    sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda || {
        error "NVIDIA driver installation failed. Please install manually:"
        error "  https://rpmfusion.org/Howto/NVIDIA"
        return 1
    }
    success "NVIDIA drivers installed. A reboot may be required."
}

setup_amd() {
    info "Setting up AMD ROCm…"

    # Check for existing ROCm installation
    if command -v rocm-smi &>/dev/null; then
        success "ROCm already installed: $(rocm-smi --version 2>/dev/null | head -1 || echo 'version unknown')"
    else
        warn "ROCm not found. Installing ROCm via dnf…"

        # Add ROCm repository
        cat <<'EOF' | sudo tee /etc/yum.repos.d/rocm.repo
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel9/6.0/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

        sudo dnf install -y rocm-hip rocm-opencl rocm-smi rocminfo || {
            error "ROCm installation failed. Manual steps:"
            error "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/linux/rhel.html"
            return 1
        }
        success "ROCm installed."
    fi

    # Configure group permissions
    CURRENT_USER="$(whoami)"
    for grp in render video; do
        if ! groups "$CURRENT_USER" 2>/dev/null | grep -qw "$grp"; then
            info "Adding $CURRENT_USER to the '$grp' group…"
            sudo usermod -aG "$grp" "$CURRENT_USER" && \
                warn "You must log out and back in (or reboot) for group changes to take effect."
        else
            success "User already in '$grp' group."
        fi
    done

    # Check device nodes
    for dev in /dev/kfd /dev/dri; do
        if [[ -e "$dev" ]]; then
            success "Device node found: $dev"
        else
            warn "Device node not found: $dev  (ROCm may not fully work yet)"
        fi
    done
}

# =============================================================================
# 3. Python virtual environment
# =============================================================================
setup_venv() {
    info "Creating Python virtual environment at $VENV_DIR…"
    python3 -m venv "$VENV_DIR"
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip wheel setuptools
    success "Virtual environment ready."
}

# =============================================================================
# 4. Python dependencies
# =============================================================================
install_python_deps() {
    local detected_gpu="$1"
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"

    info "Installing base Python dependencies…"
    pip install -r "${REPO_DIR}/requirements.txt" --no-deps 2>/dev/null || true

    info "Installing PyTorch for $detected_gpu…"
    case "$detected_gpu" in
        nvidia)
            pip install torch torchvision \
                --index-url https://download.pytorch.org/whl/cu121 \
                || pip install torch torchvision   # fall back to CPU wheel
            ;;
        amd)
            pip install torch torchvision \
                --index-url https://download.pytorch.org/whl/rocm6.0 \
                || pip install torch torchvision
            ;;
        *)
            pip install torch torchvision
            ;;
    esac

    # Install remaining requirements
    pip install PyQt6 Pillow opencv-python py_real_esrgan tqdm

    # BasicSR (optional, requires build tools)
    if command -v gcc &>/dev/null && command -v g++ &>/dev/null; then
        info "Installing BasicSR (enables additional model variants)…"
        CUDA_VISIBLE_DEVICES='' pip install basicsr 2>/dev/null && \
            success "BasicSR installed." || \
            warn "BasicSR installation failed (optional — standard models still work)."
    fi

    success "Python dependencies installed."
}

# =============================================================================
# 5. Install the upscaler package
# =============================================================================
install_package() {
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    info "Installing upscaler package in development mode…"
    pip install -e "${REPO_DIR}"
    success "Package installed."
}

# =============================================================================
# 6. Create desktop launcher
# =============================================================================
create_desktop_entry() {
    local desktop_file="${HOME}/.local/share/applications/realesrgan-upscaler.desktop"
    mkdir -p "$(dirname "$desktop_file")"
    cat > "$desktop_file" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=RealESRGAN Upscaler
Comment=PyQt6-based AI image upscaler (NVIDIA/AMD GPU support)
Exec=${VENV_DIR}/bin/realesrgan-upscaler
Icon=image-x-generic
Terminal=false
Categories=Graphics;RasterGraphics;
Keywords=upscale;AI;image;GPU;
EOF
    chmod +x "$desktop_file"
    success "Desktop entry created: $desktop_file"
}

# =============================================================================
# 7. Verification
# =============================================================================
verify_installation() {
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    info "Running verification tests…"

    python3 - <<'PYEOF'
import sys
errors = []

try:
    import torch
    print(f"  PyTorch {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"  CUDA/ROCm available: {cuda}")
    if cuda:
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    errors.append(f"PyTorch import failed: {e}")

try:
    from PyQt6.QtWidgets import QApplication
    print("  PyQt6: OK")
except ImportError as e:
    errors.append(f"PyQt6 import failed: {e}")

try:
    from py_real_esrgan.model import RealESRGAN
    print("  py_real_esrgan: OK")
except ImportError as e:
    errors.append(f"py_real_esrgan import failed: {e}")

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    print("  basicsr: OK")
except ImportError:
    print("  basicsr: not installed (optional)")

try:
    from upscaler.gpu_manager import GPUManager
    mgr = GPUManager()
    gpus = mgr.get_available_gpus()
    print(f"  GPU Manager: detected {len(gpus)} device(s)")
    for g in gpus:
        print(f"    {g}")
except Exception as e:
    errors.append(f"GPU Manager failed: {e}")

if errors:
    print("\nVerification FAILED:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("\nAll checks passed!")
PYEOF
}

# =============================================================================
# 8. Print summary
# =============================================================================
print_summary() {
    local detected_gpu="$1"
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          RealESRGAN Upscaler - Setup Complete                ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "  GPU mode      : $detected_gpu"
    echo "  Virtual env   : $VENV_DIR"
    echo "  Launch GUI    : source $VENV_DIR/bin/activate && realesrgan-upscaler"
    echo "  CLI script    : source $VENV_DIR/bin/activate && python upscaler/upscale_frames.py"
    echo ""

    if [[ "$detected_gpu" == "amd" ]]; then
        echo -e "${YELLOW}AMD GPU note:${NC}"
        echo "  If PyTorch cannot see your GPU, try:"
        echo "    HSA_OVERRIDE_GFX_VERSION=10.3.0 realesrgan-upscaler"
        echo ""
    fi

    echo "  Documentation : $REPO_DIR/README.md"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo -e "${CYAN}"
    echo "  ╔═══════════════════════════════════════════════╗"
    echo "  ║  RealESRGAN Upscaler — Fedora 43 Setup Script ║"
    echo "  ╚═══════════════════════════════════════════════╝"
    echo -e "${NC}"

    local detected_gpu
    detected_gpu="$(detect_gpu)"
    info "Detected GPU type: $detected_gpu"

    install_system_deps

    case "$detected_gpu" in
        nvidia)   setup_nvidia ;;
        amd)      setup_amd    ;;
        cpu-only) warn "No GPU detected — CPU-only mode (performance will be limited)." ;;
    esac

    setup_venv
    install_python_deps "$detected_gpu"
    install_package
    create_desktop_entry
    verify_installation
    print_summary "$detected_gpu"
}

main "$@"
