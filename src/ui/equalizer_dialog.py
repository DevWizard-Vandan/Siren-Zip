"""Audio & Video Adjustments / Equalizer Panel."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.filters.video_fx import VideoFXFilter


class EqualizerDialog(QDialog):
    """Floating Equalizer and Video Adjustments panel with real-time feedback."""

    filter_changed = Signal(VideoFXFilter)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("🎛️ Siren-VLC: Adjustments & Equalizer")
        self.resize(380, 420)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0d1117;
                color: #e6edf3;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #58a6ff;
            }
            QLabel {
                color: #c9d1d9;
                font-size: 12px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #21262d;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #58a6ff;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #58a6ff;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #c9d1d9;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #30363d;
                color: #ffffff;
            }
            """
        )

        main_layout = QVBoxLayout(self)

        # 1. Video Effects Group
        grp_video = QGroupBox("Video Image Adjustments")
        form_layout = QFormLayout(grp_video)
        form_layout.setSpacing(10)

        # Brightness (-1.0 to 1.0 -> Slider 0 to 200, center 100)
        self.sld_brightness = self._create_slider(0, 200, 100)
        self.lbl_brightness = QLabel("0.00")
        form_layout.addRow("Brightness:", self._wrap_slider(self.sld_brightness, self.lbl_brightness))

        # Contrast (0.0 to 2.0 -> Slider 0 to 200, center 100)
        self.sld_contrast = self._create_slider(0, 200, 100)
        self.lbl_contrast = QLabel("1.00")
        form_layout.addRow("Contrast:", self._wrap_slider(self.sld_contrast, self.lbl_contrast))

        # Saturation (0.0 to 2.0 -> Slider 0 to 200, center 100)
        self.sld_saturation = self._create_slider(0, 200, 100)
        self.lbl_saturation = QLabel("1.00")
        form_layout.addRow("Saturation:", self._wrap_slider(self.sld_saturation, self.lbl_saturation))

        # Gamma (0.2 to 3.0 -> Slider 20 to 300, center 100)
        self.sld_gamma = self._create_slider(20, 300, 100)
        self.lbl_gamma = QLabel("1.00")
        form_layout.addRow("Gamma:", self._wrap_slider(self.sld_gamma, self.lbl_gamma))

        # SIREN Detail Booster (0.0 to 3.0 -> Slider 0 to 300, center 0)
        self.sld_detail = self._create_slider(0, 300, 0)
        self.lbl_detail = QLabel("0.00")
        form_layout.addRow("⚡ SIREN Detail Boost:", self._wrap_slider(self.sld_detail, self.lbl_detail))

        main_layout.addWidget(grp_video)

        # Reset Buttons
        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("↺ Reset to Default")
        self.btn_reset.clicked.connect(self.reset_defaults)
        btn_layout.addWidget(self.btn_reset)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)

        main_layout.addLayout(btn_layout)

        # Connect signals
        self.sld_brightness.valueChanged.connect(self._on_values_changed)
        self.sld_contrast.valueChanged.connect(self._on_values_changed)
        self.sld_saturation.valueChanged.connect(self._on_values_changed)
        self.sld_gamma.valueChanged.connect(self._on_values_changed)
        self.sld_detail.valueChanged.connect(self._on_values_changed)

    def _create_slider(self, min_val: int, max_val: int, default_val: int) -> QSlider:
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(min_val, max_val)
        sld.setValue(default_val)
        return sld

    def _wrap_slider(self, sld: QSlider, lbl: QLabel) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(sld, stretch=1)
        lbl.setFixedWidth(40)
        h.addWidget(lbl)
        return w

    def _on_values_changed(self) -> None:
        b = (self.sld_brightness.value() - 100) / 100.0
        c = self.sld_contrast.value() / 100.0
        s = self.sld_saturation.value() / 100.0
        g = self.sld_gamma.value() / 100.0
        d = self.sld_detail.value() / 100.0

        self.lbl_brightness.setText(f"{b:+.2f}")
        self.lbl_contrast.setText(f"{c:.2f}")
        self.lbl_saturation.setText(f"{s:.2f}")
        self.lbl_gamma.setText(f"{g:.2f}")
        self.lbl_detail.setText(f"{d:.2f}")

        fx = VideoFXFilter(brightness=b, contrast=c, saturation=s, gamma=g, detail_boost=d)
        self.filter_changed.emit(fx)

    def reset_defaults(self) -> None:
        self.sld_brightness.setValue(100)
        self.sld_contrast.setValue(100)
        self.sld_saturation.setValue(100)
        self.sld_gamma.setValue(100)
        self.sld_detail.setValue(0)
        self._on_values_changed()
