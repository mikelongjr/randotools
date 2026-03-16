"""
Main Window - Professional PyQt6 main window for the RealESRGAN Upscaler.

Features:
  - File browser for selecting input images/videos
  - Model selection dropdown (RealESRGAN variants)
  - GPU selection with auto-detection (NVIDIA/AMD)
  - Batch processing queue with drag-and-drop
  - Real-time upscaling progress with ETA
  - Temperature monitoring widget
  - Output preview panel
  - Comprehensive error dialogs
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import (
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    QMimeData,
    QUrl,
    QSize,
)
from PyQt6.QtGui import (
    QAction,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QKeySequence,
    QPalette,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QSpinBox,
)

from upscaler.config import Config
from upscaler.gpu_manager import GPUInfo, GPUManager, GPUVendor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class UpscaleWorker(QThread):
    """
    Background QThread that runs the batch upscaling operation.

    Signals
    -------
    progress(completed, total, filename, eta_s)
    job_done(filename, success, error_msg)
    finished(succeeded, failed, elapsed_s)
    log_message(text)
    """

    progress = pyqtSignal(int, int, str, float)   # completed, total, filename, eta_s
    job_done = pyqtSignal(str, bool, str)          # filename, success, error
    finished = pyqtSignal(int, int, float)         # succeeded, failed, elapsed_s
    log_message = pyqtSignal(str)

    def __init__(
        self,
        input_files: List[str],
        output_dir: str,
        model_name: str,
        device_string: str,
        weights_path: str,
        use_half: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.input_files = input_files
        self.output_dir = output_dir
        self.model_name = model_name
        self.device_string = device_string
        self.weights_path = weights_path
        self.use_half = use_half
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from upscaler.upscale_engine import UpscaleEngine, UpscaleJob

        self.log_message.emit(f"Loading model '{self.model_name}' on {self.device_string}…")

        engine = UpscaleEngine(
            device_string=self.device_string,
            model_name=self.model_name,
            weights_path=self.weights_path,
            use_half_precision=self.use_half,
        )

        try:
            engine.load_model()
        except Exception as exc:
            self.log_message.emit(f"❌ Model load failed: {exc}")
            self.finished.emit(0, len(self.input_files), 0.0)
            return

        self.log_message.emit(f"✔ Model loaded. Processing {len(self.input_files)} file(s)…")

        os.makedirs(self.output_dir, exist_ok=True)
        t_start = time.perf_counter()
        succeeded = 0
        failed = 0
        times: List[float] = []

        for idx, src in enumerate(self.input_files):
            if self._cancel:
                self.log_message.emit("⛔ Batch cancelled by user.")
                break

            filename = os.path.basename(src)
            dst = os.path.join(self.output_dir, filename)

            job = UpscaleJob(
                input_path=src,
                output_path=dst,
                model_name=self.model_name,
            )

            t_job = time.perf_counter()
            engine.process_job(job)
            elapsed_job = time.perf_counter() - t_job
            times.append(elapsed_job)

            if job.success:
                succeeded += 1
            else:
                failed += 1

            self.job_done.emit(filename, job.success, job.error)

            # ETA calculation
            avg = sum(times) / len(times)
            remaining = len(self.input_files) - (idx + 1)
            eta = avg * remaining

            self.progress.emit(idx + 1, len(self.input_files), filename, eta)

        engine.unload_model()
        elapsed = time.perf_counter() - t_start
        self.finished.emit(succeeded, failed, elapsed)


# ---------------------------------------------------------------------------
# Queue list item widget
# ---------------------------------------------------------------------------


class QueueItem(QListWidgetItem):
    """Custom list item representing a queued file."""

    STATUS_PENDING = "⏳ Pending"
    STATUS_PROCESSING = "⚙ Processing…"
    STATUS_DONE = "✔ Done"
    STATUS_ERROR = "❌ Error"

    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self._status = self.STATUS_PENDING
        self._update_text()

    def set_status(self, status: str) -> None:
        self._status = status
        self._update_text()

    def _update_text(self) -> str:
        text = f"{self.filename}  [{self._status}]"
        self.setText(text)
        return text


# ---------------------------------------------------------------------------
# Temperature monitor widget
# ---------------------------------------------------------------------------


class ThermalWidget(QFrame):
    """Compact widget showing live GPU temperature and thermal state."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setFixedHeight(36)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)

        self._icon = QLabel("🌡")
        self._label = QLabel("GPU Temp: — ")
        self._state_label = QLabel("(monitoring off)")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self._icon)
        layout.addWidget(self._label)
        layout.addStretch()
        layout.addWidget(self._state_label)

    def update_temperature(self, temp_c: Optional[int], state_name: str = "") -> None:
        if temp_c is not None:
            self._label.setText(f"GPU Temp: {temp_c}°C")
            # Color-code by temperature
            if temp_c >= 95:
                color = "#ff4444"
            elif temp_c >= 85:
                color = "#ff8800"
            elif temp_c >= 75:
                color = "#ffcc00"
            else:
                color = "#44aa44"
            self._label.setStyleSheet(f"color: {color}; font-weight: bold;")
        else:
            self._label.setText("GPU Temp: N/A")
            self._label.setStyleSheet("")
        if state_name:
            self._state_label.setText(state_name)


# ---------------------------------------------------------------------------
# Preview panel
# ---------------------------------------------------------------------------


class PreviewPanel(QScrollArea):
    """Scrollable image preview panel."""

    def __init__(self, title: str = "Preview", parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(300)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._title = QLabel(f"<b>{title}</b>")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label = QLabel("No image loaded")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(280, 280)
        self._image_label.setStyleSheet("background: #1a1a2e; border-radius: 6px;")

        self._layout.addWidget(self._title)
        self._layout.addWidget(self._image_label)
        self.setWidget(self._container)

    def show_image(self, path: str) -> None:
        if not os.path.isfile(path):
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            QSize(self._image_label.width() - 8, self._image_label.height() - 8),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def clear(self) -> None:
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText("No image loaded")


# ---------------------------------------------------------------------------
# GPU Diagnostics dialog
# ---------------------------------------------------------------------------


class DiagnosticsDialog(QDialog):
    """Simple dialog showing GPU diagnostics report."""

    def __init__(self, report: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GPU Diagnostics")
        self.resize(600, 400)
        layout = QVBoxLayout(self)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("monospace", 10))
        text.setPlainText(report)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """
    Main application window for the RealESRGAN Upscaler.

    The window is organized into three main sections:
    - Left panel: queue list with drag-and-drop, controls
    - Center panel: options (model, GPU, output directory)
    - Right panel: input/output preview
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RealESRGAN Upscaler v2.0")
        self.setAcceptDrops(True)

        # Load config and detect GPUs
        self._config = Config.load()
        self._gpu_manager = GPUManager()
        self._gpus = self._gpu_manager.get_available_gpus()

        # Worker reference
        self._worker: Optional[UpscaleWorker] = None
        self._thermal_timer: Optional[QTimer] = None

        # Power manager (lazy-initialised on first run)
        self._power_mgr = None

        self._setup_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        self._populate_gpu_combo()
        self._apply_window_geometry()
        self._start_thermal_monitor()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 4)
        root_layout.setSpacing(6)

        # Thermal bar
        self._thermal = ThermalWidget()
        root_layout.addWidget(self._thermal)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # ---- Left: Queue panel ----------------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        queue_group = QGroupBox("Processing Queue")
        queue_inner = QVBoxLayout(queue_group)

        drop_hint = QLabel("Drop images here or use 'Add Files'")
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_hint.setStyleSheet("color: #888; font-style: italic; padding: 4px;")
        queue_inner.addWidget(drop_hint)

        self._queue_list = QListWidget()
        self._queue_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._queue_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._queue_list.itemClicked.connect(self._on_queue_item_clicked)
        queue_inner.addWidget(self._queue_list)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add Files…")
        self._btn_add.clicked.connect(self._add_files)
        self._btn_add_dir = QPushButton("Add Folder…")
        self._btn_add_dir.clicked.connect(self._add_folder)
        self._btn_remove = QPushButton("Remove")
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_clear = QPushButton("Clear All")
        self._btn_clear.clicked.connect(self._clear_queue)
        for btn in (self._btn_add, self._btn_add_dir, self._btn_remove, self._btn_clear):
            btn_row.addWidget(btn)
        queue_inner.addLayout(btn_row)
        left_layout.addWidget(queue_group)

        splitter.addWidget(left_panel)

        # ---- Center: Options + Progress + Log -------------------------
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(4, 0, 4, 0)

        # Options group
        opt_group = QGroupBox("Upscaling Options")
        opt_grid = QGridLayout(opt_group)
        opt_grid.setColumnStretch(1, 1)

        # Model
        opt_grid.addWidget(QLabel("Model:"), 0, 0)
        self._model_combo = QComboBox()
        for key, meta in Config.MODELS.items():
            self._model_combo.addItem(
                f"{key}  —  {meta['description']}", userData=key
            )
        # Select saved model
        saved_idx = self._model_combo.findData(self._config.model_name)
        if saved_idx >= 0:
            self._model_combo.setCurrentIndex(saved_idx)
        opt_grid.addWidget(self._model_combo, 0, 1)

        # GPU
        opt_grid.addWidget(QLabel("GPU:"), 1, 0)
        self._gpu_combo = QComboBox()
        opt_grid.addWidget(self._gpu_combo, 1, 1)
        self._btn_diag = QPushButton("Diagnostics…")
        self._btn_diag.setFixedWidth(120)
        self._btn_diag.clicked.connect(self._show_diagnostics)
        opt_grid.addWidget(self._btn_diag, 1, 2)

        # Output directory
        opt_grid.addWidget(QLabel("Output Dir:"), 2, 0)
        self._output_edit = QLabel(self._config.output_dir or "(not set)")
        self._output_edit.setStyleSheet("background:#1e1e2e; padding:3px; border-radius:3px;")
        opt_grid.addWidget(self._output_edit, 2, 1)
        self._btn_output = QPushButton("Browse…")
        self._btn_output.setFixedWidth(120)
        self._btn_output.clicked.connect(self._choose_output_dir)
        opt_grid.addWidget(self._btn_output, 2, 2)

        # Half-precision
        self._half_cb = QCheckBox("Use FP16 (half-precision) — faster, recommended for GPU")
        self._half_cb.setChecked(self._config.use_half_precision)
        opt_grid.addWidget(self._half_cb, 3, 0, 1, 3)

        center_layout.addWidget(opt_group)

        # Start/Cancel
        action_row = QHBoxLayout()
        self._btn_start = QPushButton("▶  Start Upscaling")
        self._btn_start.setFixedHeight(40)
        self._btn_start.setStyleSheet(
            "QPushButton { background: #2d6a4f; color: white; font-weight: bold; "
            "border-radius: 6px; } QPushButton:hover { background: #40916c; }"
        )
        self._btn_start.clicked.connect(self._start_processing)

        self._btn_cancel = QPushButton("⛔  Cancel")
        self._btn_cancel.setFixedHeight(40)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setStyleSheet(
            "QPushButton { background: #6b2737; color: white; font-weight: bold; "
            "border-radius: 6px; } QPushButton:hover { background: #9b2335; }"
            "QPushButton:disabled { background: #444; }"
        )
        self._btn_cancel.clicked.connect(self._cancel_processing)

        action_row.addWidget(self._btn_start)
        action_row.addWidget(self._btn_cancel)
        center_layout.addLayout(action_row)

        # Progress
        progress_group = QGroupBox("Progress")
        prog_layout = QVBoxLayout(progress_group)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%v / %m  (%p%)")
        prog_layout.addWidget(self._progress_bar)

        self._eta_label = QLabel("ETA: —")
        self._eta_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        prog_layout.addWidget(self._eta_label)
        center_layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("monospace", 9))
        self._log.setMaximumHeight(160)
        log_layout.addWidget(self._log)
        center_layout.addWidget(log_group)

        splitter.addWidget(center_panel)

        # ---- Right: Preview panels -----------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._preview_in = PreviewPanel("Input Preview")
        self._preview_out = PreviewPanel("Output Preview")

        right_preview_splitter = QSplitter(Qt.Orientation.Vertical)
        right_preview_splitter.addWidget(self._preview_in)
        right_preview_splitter.addWidget(self._preview_out)
        right_layout.addWidget(right_preview_splitter)

        splitter.addWidget(right_panel)

        # Splitter proportions: 30% / 45% / 25%
        splitter.setSizes([300, 450, 250])

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        act_add = QAction("&Add Files…", self)
        act_add.setShortcut(QKeySequence("Ctrl+O"))
        act_add.triggered.connect(self._add_files)
        file_menu.addAction(act_add)

        act_add_dir = QAction("Add &Folder…", self)
        act_add_dir.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_add_dir.triggered.connect(self._add_folder)
        file_menu.addAction(act_add_dir)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        act_diag = QAction("GPU &Diagnostics…", self)
        act_diag.triggered.connect(self._show_diagnostics)
        tools_menu.addAction(act_diag)

        act_cfg = QAction("Open Config &Folder…", self)
        act_cfg.triggered.connect(self._open_config_folder)
        tools_menu.addAction(act_cfg)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _setup_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))

        act_add = QAction("➕ Add Files", self)
        act_add.triggered.connect(self._add_files)
        tb.addAction(act_add)

        act_start = QAction("▶ Start", self)
        act_start.triggered.connect(self._start_processing)
        tb.addAction(act_start)

        act_cancel = QAction("⛔ Cancel", self)
        act_cancel.triggered.connect(self._cancel_processing)
        tb.addAction(act_cancel)

        tb.addSeparator()
        act_diag = QAction("🔍 Diagnostics", self)
        act_diag.triggered.connect(self._show_diagnostics)
        tb.addAction(act_diag)

    def _setup_statusbar(self) -> None:
        sb = self.statusBar()
        self._status_label = QLabel("Ready")
        sb.addWidget(self._status_label, 1)

        gpu_count = len([g for g in self._gpus if g.vendor != GPUVendor.CPU])
        gpu_text = (
            f"GPUs detected: {gpu_count}" if gpu_count else "No GPU (CPU mode)"
        )
        sb.addPermanentWidget(QLabel(gpu_text))

    # ------------------------------------------------------------------
    # GPU population
    # ------------------------------------------------------------------

    def _populate_gpu_combo(self) -> None:
        self._gpu_combo.clear()
        for gpu in self._gpus:
            label = str(gpu)
            self._gpu_combo.addItem(label, userData=gpu.device_string)

        # Restore saved GPU selection
        saved = self._config.preferred_gpu_index
        if 0 <= saved < self._gpu_combo.count():
            self._gpu_combo.setCurrentIndex(saved)

    def _current_device_string(self) -> str:
        idx = self._gpu_combo.currentIndex()
        data = self._gpu_combo.itemData(idx)
        return data if data else "cpu"

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                files.append(path)
            elif os.path.isdir(path):
                files.extend(self._scan_directory(path))
        self._enqueue_files(files)

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def _add_files(self) -> None:
        start_dir = self._config.input_dir or str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;All Files (*)",
        )
        if paths:
            self._enqueue_files(paths)
            self._config.add_recent_input(os.path.dirname(paths[0]))

    def _add_folder(self) -> None:
        start_dir = self._config.input_dir or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Select Folder", start_dir)
        if directory:
            files = self._scan_directory(directory)
            self._enqueue_files(files)
            self._config.add_recent_input(directory)

    def _enqueue_files(self, paths: List[str]) -> None:
        existing = {
            self._queue_list.item(i).filepath  # type: ignore[attr-defined]
            for i in range(self._queue_list.count())
        }
        added = 0
        for p in paths:
            if p not in existing:
                item = QueueItem(p)
                self._queue_list.addItem(item)
                added += 1
        if added:
            self._log_message(f"Added {added} file(s) to queue.")
            self._status_label.setText(f"Queue: {self._queue_list.count()} file(s)")

    def _remove_selected(self) -> None:
        for item in self._queue_list.selectedItems():
            self._queue_list.takeItem(self._queue_list.row(item))

    def _clear_queue(self) -> None:
        self._queue_list.clear()
        self._preview_in.clear()
        self._preview_out.clear()
        self._status_label.setText("Queue cleared")

    def _on_queue_item_clicked(self, item: QListWidgetItem) -> None:
        if isinstance(item, QueueItem):
            self._preview_in.show_image(item.filepath)

    @staticmethod
    def _scan_directory(directory: str) -> List[str]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
        results = []
        for fname in sorted(os.listdir(directory)):
            if Path(fname).suffix.lower() in exts:
                results.append(os.path.join(directory, fname))
        return results

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    def _choose_output_dir(self) -> None:
        start = self._config.output_dir or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", start)
        if directory:
            self._config.output_dir = directory
            self._output_edit.setText(directory)
            self._config.add_recent_output(directory)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _start_processing(self) -> None:
        if self._queue_list.count() == 0:
            QMessageBox.warning(self, "Empty Queue", "Please add images to the queue first.")
            return

        output_dir = self._config.output_dir
        if not output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory.")
            return

        model_key = self._model_combo.currentData()
        weights_path = self._config.resolve_weights_path(model_key)
        if not os.path.isfile(weights_path):
            QMessageBox.critical(
                self,
                "Weights Not Found",
                f"Model weights file not found:\n{weights_path}\n\n"
                "Please place the .pth file in the 'weights' directory.",
            )
            return

        device = self._current_device_string()
        files = [
            self._queue_list.item(i).filepath  # type: ignore[attr-defined]
            for i in range(self._queue_list.count())
        ]

        # Save settings
        self._config.model_name = model_key
        self._config.preferred_gpu_index = self._gpu_combo.currentIndex()
        self._config.use_half_precision = self._half_cb.isChecked()
        self._config.save()

        # Reset UI
        self._progress_bar.setRange(0, len(files))
        self._progress_bar.setValue(0)
        self._eta_label.setText("ETA: calculating…")
        self._btn_start.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._log_message(f"Starting batch: {len(files)} file(s) → {output_dir}")

        # Reset queue item statuses
        for i in range(self._queue_list.count()):
            item = self._queue_list.item(i)
            if isinstance(item, QueueItem):
                item.set_status(QueueItem.STATUS_PENDING)

        # Launch worker
        self._worker = UpscaleWorker(
            input_files=files,
            output_dir=output_dir,
            model_name=model_key,
            device_string=device,
            weights_path=weights_path,
            use_half=self._half_cb.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.job_done.connect(self._on_job_done)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.log_message.connect(self._log_message)
        self._worker.start()

    def _cancel_processing(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._log_message("Cancelling…")
            self._btn_cancel.setEnabled(False)

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    def _on_progress(self, completed: int, total: int, filename: str, eta_s: float) -> None:
        self._progress_bar.setValue(completed)
        self._progress_bar.setMaximum(total)
        eta_str = self._format_eta(eta_s)
        self._eta_label.setText(f"ETA: {eta_str}")
        self._status_label.setText(f"Processing {completed}/{total}: {filename}")

        # Mark current item
        for i in range(self._queue_list.count()):
            item = self._queue_list.item(i)
            if isinstance(item, QueueItem) and item.filename == filename:
                item.set_status(QueueItem.STATUS_PROCESSING)
                self._queue_list.scrollToItem(item)
                break

    def _on_job_done(self, filename: str, success: bool, error: str) -> None:
        # Mark item done/error
        for i in range(self._queue_list.count()):
            item = self._queue_list.item(i)
            if isinstance(item, QueueItem) and item.filename == filename:
                item.set_status(QueueItem.STATUS_DONE if success else QueueItem.STATUS_ERROR)
                # Show output preview for first completed file
                if success and self._preview_out:
                    out_path = os.path.join(self._config.output_dir, filename)
                    self._preview_out.show_image(out_path)
                break

        if not success:
            self._log_message(f"❌ {filename}: {error}")

    def _on_batch_finished(self, succeeded: int, failed: int, elapsed_s: float) -> None:
        self._btn_start.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        elapsed_str = self._format_eta(elapsed_s)
        msg = (
            f"✅ Batch complete in {elapsed_str}  —  "
            f"{succeeded} succeeded, {failed} failed."
        )
        self._log_message(msg)
        self._status_label.setText(f"Done: {succeeded}/{succeeded + failed} succeeded")
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._eta_label.setText("ETA: done")

        if failed > 0:
            QMessageBox.warning(
                self,
                "Batch Finished with Errors",
                f"{succeeded} file(s) upscaled successfully.\n{failed} file(s) failed.\n\n"
                "Check the Activity Log for details.",
            )

    # ------------------------------------------------------------------
    # Thermal monitoring
    # ------------------------------------------------------------------

    def _start_thermal_monitor(self) -> None:
        self._thermal_timer = QTimer(self)
        self._thermal_timer.setInterval(3000)  # 3 seconds
        self._thermal_timer.timeout.connect(self._poll_temperature)
        self._thermal_timer.start()

    def _poll_temperature(self) -> None:
        from upscaler.power_manager import PowerManager

        if self._power_mgr is None:
            device = self._current_device_string()
            gpu_idx = 0
            vendor = "auto"
            if "cuda:" in device:
                gpu_idx = int(device.split(":")[-1])
                # Determine vendor from GPU info
                idx = self._gpu_combo.currentIndex()
                if 0 <= idx < len(self._gpus):
                    vendor = "nvidia" if self._gpus[idx].vendor == GPUVendor.NVIDIA else "amd"
            try:
                self._power_mgr = PowerManager(gpu_index=gpu_idx, vendor=vendor)
            except Exception:
                return

        try:
            temp = self._power_mgr.read_temperature()
            state = self._power_mgr.current_state.name
            self._thermal.update_temperature(temp, state)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Misc actions
    # ------------------------------------------------------------------

    def _show_diagnostics(self) -> None:
        report = self._gpu_manager.run_diagnostics()
        dlg = DiagnosticsDialog(report, self)
        dlg.exec()

    def _open_config_folder(self) -> None:
        import subprocess as sp
        cfg_dir = str(Path.home() / ".config" / "realesrgan-upscaler")
        Path(cfg_dir).mkdir(parents=True, exist_ok=True)
        try:
            sp.Popen(["xdg-open", cfg_dir])
        except Exception:
            QMessageBox.information(self, "Config Folder", f"Config folder:\n{cfg_dir}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About RealESRGAN Upscaler",
            "<h2>RealESRGAN Upscaler v2.0</h2>"
            "<p>A production-ready image upscaling application powered by "
            "<b>Real-ESRGAN</b> and <b>PyQt6</b>.</p>"
            "<ul>"
            "<li>NVIDIA GPU support (CUDA)</li>"
            "<li>AMD GPU support (ROCm/HIP)</li>"
            "<li>Batch processing with drag-and-drop</li>"
            "<li>Real-time temperature monitoring</li>"
            "<li>Optimized for Fedora 43</li>"
            "</ul>"
            "<p>Repository: <a href='https://github.com/mikelongjr/randotools'>"
            "github.com/mikelongjr/randotools</a></p>",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds < 0:
            return "—"
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m {s % 60}s"
        return f"{s // 3600}h {(s % 3600) // 60}m"

    def _log_message(self, text: str) -> None:
        self._log.append(text)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _apply_window_geometry(self) -> None:
        self.resize(self._config.window_width, self._config.window_height)

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Save window size
        self._config.window_width = self.width()
        self._config.window_height = self.height()
        self._config.save()

        if self._thermal_timer:
            self._thermal_timer.stop()

        if self._power_mgr:
            try:
                self._power_mgr.stop()
            except Exception:
                pass

        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing in Progress",
                "Upscaling is still running. Cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._worker.cancel()
            self._worker.wait(5000)

        event.accept()
