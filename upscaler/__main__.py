"""Entry point for the RealESRGAN Upscaler GUI application."""

import sys


def main() -> None:
    """Launch the PyQt6 GUI application."""
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
