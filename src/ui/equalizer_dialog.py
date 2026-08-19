"""VLC-Style Adjustments and Effects Dialog (Audio EQ, Video FX, and Track Synchronization)."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.filters.video_fx import VideoFXFilter

VLC_EQ_PRESETS: Dict[str, List[int]] = {
    "Flat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Classical": [0, 0, 0, 0, 0, 0, -4, -4, -4, -6],
    "Club": [0, 0, 2, 4, 4, 4, 2, 0, 0, 0],
    "Dance": [6, 5, 2, 0, 0, -3, -4, -4, 0, 0],
    "Full Bass": [6, 6, 6, 4, 2, -3, -6, -7, -8, -8],
    "Headphones": [3, 7, 4, -2, -3, 1, 3, 6, 8, 9],
    "Live": [-3, 0, 3, 4, 4, 4, 3, 2, 2, 2],
    "Party": [5, 5, 0, 0, 0, 0, 0, 0, 5, 5],
    "Pop": [-1, 1, 4, 5, 4, -1, -2, -2, -1, -1],
    "Reggae": [0, 0, -1, -4, 0, 4, 4, 1, 0, 0],
    "Rock": [5, 3, -4, -6, -2, 3, 6, 7, 7, 7],
    "Ska": [-2, -3, -3, -1, 3, 4, 6, 7, 7, 6],
    "Soft": [3, 1, -1, -2, -1, 3, 6, 7, 8, 8],
    "Techno": [5, 4, 0, -4, -3, 0, 5, 6, 6, 5],
    "Vocal / Dialogue Booster": [-2, -3, -3, 1, 5, 6, 6, 4, 2, 0],
}


class EqualizerDialog(QDialog):
    """VLC-Authentic 'Adjustments & Effects' Panel with Audio EQ, Video Colors, and Sync."""

    filter_changed = Signal(VideoFXFilter)
    audio_delay_changed = Signal(float)  # ms
    subtitle_delay_changed = Signal(float)  # ms
    night_mode_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None, initial_fx: Optional[VideoFXFilter] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("🎛️ Adjustments and Effects - Siren-VLC")
        self.resize(520, 480)
        self.fx = initial_fx or VideoFXFilter()

        self.setup_ui()

    def setup_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #181a1f;
                color: #e6edf3;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                background-color: #21252b;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #181a1f;
                color: #8b949e;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 600;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background: #21252b;
                color: #ff8800;
                border: 1px solid #30363d;
                border-bottom: 1px solid #21252b;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 14px;
                font-weight: bold;
                color: #ff8800;
                font-size: 12px;
            }
            QLabel {
                color: #c9d1d9;
                font-size: 11px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #30363d;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #ff8800;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #ff8800;
                width: 12px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 6px;
            }
            QSlider::groove:vertical {
                width: 4px;
                background: #30363d;
                border-radius: 2px;
            }
            QSlider::sub-page:vertical {
                background: #ff8800;
                border-radius: 2px;
            }
            QSlider::handle:vertical {
                background: #ffffff;
                border: 2px solid #ff8800;
                height: 12px;
                margin-left: -4px;
                margin-right: -4px;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #282c34;
                border: 1px solid #3e4451;
                border-radius: 4px;
                color: #abb2bf;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #353b45;
                color: #ffffff;
                border-color: #ff8800;
            }
            QComboBox {
                background-color: #282c34;
                border: 1px solid #3e4451;
                border-radius: 4px;
                color: #abb2bf;
                padding: 4px 8px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.tabs = QTabWidget()

        # Tab 1: Audio Effects (Graphic Equalizer)
        tab_audio = self._build_audio_tab()
        self.tabs.addTab(tab_audio, "🎵 Audio Effects")

        # Tab 2: Video Effects (Color Adjustments & Sharpen)
        tab_video = self._build_video_tab()
        self.tabs.addTab(tab_video, "🎬 Video Effects")

        # Tab 3: Track Synchronization
        tab_sync = self._build_sync_tab()
        self.tabs.addTab(tab_sync, "⏱️ Synchronization")

        layout.addWidget(self.tabs)

        # Bottom Action Buttons
        row_actions = QHBoxLayout()
        btn_reset = QPushButton("↺ Reset All")
        btn_reset.clicked.connect(self.reset_defaults)
        row_actions.addWidget(btn_reset)
        row_actions.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        row_actions.addWidget(btn_close)

        layout.addLayout(row_actions)

    def _build_audio_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Preset Selector
        row_preset = QHBoxLayout()
        row_preset.addWidget(QLabel("Preset:"))
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(VLC_EQ_PRESETS.keys()))
        self.cmb_preset.currentTextChanged.connect(self._on_preset_changed)
        row_preset.addWidget(self.cmb_preset, stretch=1)

        self.chk_night = QCheckBox("🌙 Night Mode (Dynamic Compressor)")
        self.chk_night.toggled.connect(self.night_mode_toggled.emit)
        row_preset.addWidget(self.chk_night)
        layout.addLayout(row_preset)

        # 10-Band Graphic Equalizer Sliders
        grp_bands = QGroupBox("10-Band Graphic Equalizer")
        h_bands = QHBoxLayout(grp_bands)
        h_bands.setSpacing(8)

        self.eq_sliders: List[QSlider] = []
        band_labels = ["60Hz", "170Hz", "310Hz", "600Hz", "1kHz", "3kHz", "6kHz", "12kHz", "14kHz", "16kHz"]

        for lbl_text in band_labels:
            col = QVBoxLayout()
            col.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_val = QLabel("0dB")
            lbl_val.setStyleSheet("font-size: 9px; color: #8b949e;")

            sld = QSlider(Qt.Orientation.Vertical)
            sld.setRange(-20, 20)
            sld.setValue(0)
            sld.setFixedHeight(120)
            sld.valueChanged.connect(lambda v, l=lbl_val: l.setText(f"{v:+d}dB"))

            lbl_name = QLabel(lbl_text)
            lbl_name.setStyleSheet("font-size: 9px; font-weight: bold; color: #c9d1d9;")

            col.addWidget(lbl_val, alignment=Qt.AlignmentFlag.AlignCenter)
            col.addWidget(sld, alignment=Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl_name, alignment=Qt.AlignmentFlag.AlignCenter)
            h_bands.addLayout(col)
            self.eq_sliders.append(sld)

        layout.addWidget(grp_bands)
        return w

    def _build_video_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Image Adjustments Group
        grp_img = QGroupBox("Image Adjust")
        form = QFormLayout(grp_img)
        form.setSpacing(8)

        # Brightness
        self.sld_brightness = QSlider(Qt.Orientation.Horizontal)
        self.sld_brightness.setRange(0, 200)
        self.sld_brightness.setValue(100)
        self.lbl_brightness = QLabel("0.00")
        form.addRow("Brightness:", self._wrap_slider(self.sld_brightness, self.lbl_brightness))

        # Contrast
        self.sld_contrast = QSlider(Qt.Orientation.Horizontal)
        self.sld_contrast.setRange(0, 200)
        self.sld_contrast.setValue(100)
        self.lbl_contrast = QLabel("1.00")
        form.addRow("Contrast:", self._wrap_slider(self.sld_contrast, self.lbl_contrast))

        # Saturation
        self.sld_saturation = QSlider(Qt.Orientation.Horizontal)
        self.sld_saturation.setRange(0, 200)
        self.sld_saturation.setValue(100)
        self.lbl_saturation = QLabel("1.00")
        form.addRow("Saturation:", self._wrap_slider(self.sld_saturation, self.lbl_saturation))

        # Gamma
        self.sld_gamma = QSlider(Qt.Orientation.Horizontal)
        self.sld_gamma.setRange(20, 300)
        self.sld_gamma.setValue(100)
        self.lbl_gamma = QLabel("1.00")
        form.addRow("Gamma:", self._wrap_slider(self.sld_gamma, self.lbl_gamma))

        # Detail Boost / Sharpen
        self.sld_detail = QSlider(Qt.Orientation.Horizontal)
        self.sld_detail.setRange(0, 300)
        self.sld_detail.setValue(0)
        self.lbl_detail = QLabel("0.00")
        form.addRow("⚡ SIREN Detail Boost:", self._wrap_slider(self.sld_detail, self.lbl_detail))

        layout.addWidget(grp_img)

        # Connect signals
        self.sld_brightness.valueChanged.connect(self._on_video_fx_changed)
        self.sld_contrast.valueChanged.connect(self._on_video_fx_changed)
        self.sld_saturation.valueChanged.connect(self._on_video_fx_changed)
        self.sld_gamma.valueChanged.connect(self._on_video_fx_changed)
        self.sld_detail.valueChanged.connect(self._on_video_fx_changed)

        return w

    def _build_sync_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        grp_audio_sync = QGroupBox("Audio Track Synchronization")
        form_a = QFormLayout(grp_audio_sync)
        self.sld_audio_delay = QSlider(Qt.Orientation.Horizontal)
        self.sld_audio_delay.setRange(-500, 500)
        self.sld_audio_delay.setValue(0)
        self.lbl_audio_delay = QLabel("0.0 ms")
        self.sld_audio_delay.valueChanged.connect(lambda v: self._on_sync_changed(v, self.lbl_audio_delay, self.audio_delay_changed))
        form_a.addRow("Audio Desync Compensation:", self._wrap_slider(self.sld_audio_delay, self.lbl_audio_delay))
        layout.addWidget(grp_audio_sync)

        grp_sub_sync = QGroupBox("Subtitle Track Synchronization")
        form_s = QFormLayout(grp_sub_sync)
        self.sld_sub_delay = QSlider(Qt.Orientation.Horizontal)
        self.sld_sub_delay.setRange(-5000, 5000)
        self.sld_sub_delay.setValue(0)
        self.lbl_sub_delay = QLabel("0.0 ms")
        self.sld_sub_delay.valueChanged.connect(lambda v: self._on_sync_changed(v, self.lbl_sub_delay, self.subtitle_delay_changed))
        form_s.addRow("Subtitle Delay Compensation:", self._wrap_slider(self.sld_sub_delay, self.lbl_sub_delay))
        layout.addWidget(grp_sub_sync)

        layout.addStretch()
        return w

    def _wrap_slider(self, sld: QSlider, lbl: QLabel) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(sld, stretch=1)
        lbl.setFixedWidth(50)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(lbl)
        return w

    def _on_preset_changed(self, name: str) -> None:
        if name in VLC_EQ_PRESETS:
            gains = VLC_EQ_PRESETS[name]
            for sld, g in zip(self.eq_sliders, gains):
                sld.setValue(g)

    def _on_video_fx_changed(self) -> None:
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

    def _on_sync_changed(self, val: int, lbl: QLabel, sig: Signal) -> None:
        lbl.setText(f"{val:+d} ms")
        sig.emit(float(val))

    def reset_defaults(self) -> None:
        self.cmb_preset.setCurrentText("Flat")
        self.chk_night.setChecked(False)
        self.sld_brightness.setValue(100)
        self.sld_contrast.setValue(100)
        self.sld_saturation.setValue(100)
        self.sld_gamma.setValue(100)
        self.sld_detail.setValue(0)
        self.sld_audio_delay.setValue(0)
        self.sld_sub_delay.setValue(0)
        self._on_video_fx_changed()
