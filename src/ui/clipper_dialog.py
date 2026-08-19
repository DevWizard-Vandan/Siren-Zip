"""One-Click GIF and WebM Clipper Modal Dialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.tools.clipper import VideoClipper


class ClipperDialog(QDialog):
    """Modal dialog allowing users to set A-B points and export GIFs/WebMs."""

    def __init__(self, parent: Optional[QWidget] = None, get_current_time_cb=None, get_video_path_cb=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("✂️ One-Click GIF / WebM Clipper")
        self.setFixedSize(440, 320)
        self.get_current_time_cb = get_current_time_cb
        self.get_video_path_cb = get_video_path_cb

        self.start_time: float = 0.0
        self.end_time: float = 5.0

        self.setup_ui()

    def setup_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #161b22;
                color: #e6edf3;
            }
            QLabel {
                color: #e6edf3;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                color: #c9d1d9;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #8b949e;
            }
            QPushButton#ExportBtn {
                background-color: #238636;
                color: #ffffff;
                font-weight: bold;
                border-color: #2ea043;
            }
            QPushButton#ExportBtn:hover {
                background-color: #2ea043;
            }
            QComboBox {
                background-color: #21262d;
                border: 1px solid #30363d;
                color: #c9d1d9;
                border-radius: 6px;
                padding: 4px 8px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(10)

        # Point A (Start)
        row_a = QHBoxLayout()
        self.lbl_start = QLabel("00:00.000")
        self.lbl_start.setStyleSheet("font-family: monospace; font-weight: bold; color: #58a6ff;")
        btn_set_a = QPushButton("📍 Set Point A (Start)")
        btn_set_a.clicked.connect(self.on_set_point_a)
        row_a.addWidget(self.lbl_start)
        row_a.addWidget(btn_set_a)
        form.addRow("Start Time (A):", row_a)

        # Point B (End)
        row_b = QHBoxLayout()
        self.lbl_end = QLabel("00:05.000")
        self.lbl_end.setStyleSheet("font-family: monospace; font-weight: bold; color: #58a6ff;")
        btn_set_b = QPushButton("📍 Set Point B (End)")
        btn_set_b.clicked.connect(self.on_set_point_b)
        row_b.addWidget(self.lbl_end)
        row_b.addWidget(btn_set_b)
        form.addRow("End Time (B):", row_b)

        # Format Selector
        self.cmb_format = QComboBox()
        self.cmb_format.addItems(["Animated GIF (for Discord / WhatsApp)", "WebM Video (Ultra-Compressed VP9)"])
        form.addRow("Export Format:", self.cmb_format)

        # Resolution Selector
        self.cmb_res = QComboBox()
        self.cmb_res.addItems(["480p (Standard)", "360p (Smallest File Size)", "720p (HD)"])
        form.addRow("Resolution:", self.cmb_res)

        layout.addLayout(form)
        layout.addStretch()

        # Action Buttons
        row_actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        self.btn_export = QPushButton("🚀 Export Clip")
        self.btn_export.setObjectName("ExportBtn")
        self.btn_export.clicked.connect(self.on_export)

        row_actions.addStretch()
        row_actions.addWidget(btn_cancel)
        row_actions.addWidget(self.btn_export)
        layout.addLayout(row_actions)

    def on_set_point_a(self) -> None:
        if self.get_current_time_cb:
            self.start_time = float(self.get_current_time_cb())
            m, s = int(self.start_time // 60), self.start_time % 60
            self.lbl_start.setText(f"{m:02d}:{s:06.3f}")

    def on_set_point_b(self) -> None:
        if self.get_current_time_cb:
            self.end_time = float(self.get_current_time_cb())
            if self.end_time <= self.start_time:
                self.end_time = self.start_time + 3.0
            m, s = int(self.end_time // 60), self.end_time % 60
            self.lbl_end.setText(f"{m:02d}:{s:06.3f}")

    def on_export(self) -> None:
        video_p = self.get_video_path_cb() if self.get_video_path_cb else None
        if not video_p:
            QMessageBox.warning(self, "No Video", "Please load a video file first to clip.")
            return

        is_gif = "GIF" in self.cmb_format.currentText()
        ext = ".gif" if is_gif else ".webm"
        filter_str = "GIF Files (*.gif)" if is_gif else "WebM Files (*.webm)"

        save_p, _ = QFileDialog.getSaveFileName(self, "Save Clip", f"clip{ext}", filter_str)
        if not save_p:
            return

        w_map = {"360p (Smallest File Size)": 360, "480p (Standard)": 480, "720p (HD)": 720}
        target_w = w_map.get(self.cmb_res.currentText(), 480)

        self.btn_export.setEnabled(False)
        self.btn_export.setText("⏳ Rendering...")
        try:
            if is_gif:
                res = VideoClipper.clip_to_gif(video_p, save_p, self.start_time, self.end_time, width=target_w)
            else:
                res = VideoClipper.clip_to_webm(video_p, save_p, self.start_time, self.end_time, width=target_w)

            QMessageBox.information(
                self,
                "Clip Exported!",
                f"Successfully exported {res['format'].upper()} clip!\n\n"
                f"File: {res['output_path']}\n"
                f"Size: {res['size_kb']:.1f} KB\n"
                f"Duration: {res['duration']:.2f}s",
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export clip:\n{e}")
        finally:
            self.btn_export.setEnabled(True)
            self.btn_export.setText("🚀 Export Clip")
