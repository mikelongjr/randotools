"""Entry point for the RealESRGAN Upscaler GUI application."""

import sys


def main() -> None:
    """Launch the PyQt6 GUI application."""
    # Auto-apply HSA_OVERRIDE_GFX_VERSION for AMD GPUs whose GFX variant is
    # not compiled into the PyTorch ROCm wheel (e.g. gfx1032/gfx1035 need
    # the gfx1030-compiled kernels via override "10.3.0").  Must happen before
    # the first HIP kernel execution — i.e. before torch.cuda.* is called.
    if sys.platform == "linux":
        try:
            from upscaler.gpu_manager import GPUManager
            GPUManager.apply_rocm_gfx_override()
        except Exception:
            pass  # Non-fatal: user can set HSA_OVERRIDE_GFX_VERSION manually

    from PyQt6.QtWidgets import QApplication
    from upscaler.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("RealESRGAN Upscaler")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("randotools")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
