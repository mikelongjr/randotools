import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog

class UpscalerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Upscaler")
        
        layout = QVBoxLayout()
        
        self.label = QLabel("Welcome to the Image Upscaler!")
        layout.addWidget(self.label)
        
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)
        
        self.setLayout(layout)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.label.setText(f"Loaded image: {file_name}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UpscalerApp()
    window.show()
    sys.exit(app.exec())
