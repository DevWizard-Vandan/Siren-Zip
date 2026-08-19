"""Modern Dark-Theme Desktop GUI for Siren-Zip Video INR Codec."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from src.player.engine import PlayerEngine, RenderRequest, RenderResult, ViewportBounds
from src.player.neura_reader import NeuraReader
from src.ui.split_view import SplitComparisonView
from src.ui.video_canvas import ContinuousVideoCanvas


DARK_STYLESHEET = """
QMainWindow {
    background-color: #090d13;
    color: #e6edf3;
}
QWidget {
    background-color: #090d13;
    color: #e6edf3;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QFrame#ControlCard {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px;
}
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #c9d1d9;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #161b22;
}
QPushButton#PrimaryBtn {
    background-color: #238636;
    border: 1px solid #2ea043;
    color: #ffffff;
}
QPushButton#PrimaryBtn:hover {
    background-color: #2ea043;
}
QPushButton#PrimaryBtn:pressed {
    background-color: #1f7f34;
}
QSlider::groove:horizontal {
    border: 1px solid #30363d;
    height: 6px;
    background: #21262d;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #00e676;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #00e676;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #00e676;
    border-color: #ffffff;
}
QComboBox {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 10px;
    color: #c9d1d9;
    font-weight: 600;
}
QComboBox:hover {
    border-color: #8b949e;
}
QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    selection-background-color: #00e676;
    selection-color: #000000;
}
QLabel#HUDLabel {
    color: #00e676;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    font-weight: bold;
}
QLabel#TimeLabel {
    color: #8b949e;
    font-family: 'Consolas', monospace;
    font-size: 12px;
}
QCheckBox {
    color: #c9d1d9;
    font-size: 12px;
}
"""


class SirenPlayerWindow(QMainWindow):
    """Main desktop application window for Siren-Zip continuous video playback."""

    def __init__(
        self,
        neura_path: Optional[str] = None,
        baseline_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Siren Player - Implicit Neural Compression (.neura)")
        self.resize(1280, 840)
        self.setStyleSheet(DARK_STYLESHEET)

        # State Variables
        self.engine: Optional[PlayerEngine] = None
        self.neura_meta: Dict[str, Any] = {}
        self.discrete_frames: Optional[list[np.ndarray]] = None
        self.baseline_path: Optional[str] = baseline_path

        # Playback State
        self.is_playing: bool = False
        self.current_t: float = -1.0  # Range [-1.0, 1.0]
        self.playback_speed: float = 1.0
        self.is_looping: bool = True
        self.is_split_mode: bool = False

        # Current Viewport Bounds
        self.current_viewport = ViewportBounds()
        self.current_zoom: float = 1.0

        # Performance & HUD tracking
        self.last_render_time_ms: float = 0.0
        self.last_fps: float = 0.0
        self.fps_frame_count: int = 0
        self.fps_start_time: float = time.perf_counter()

        # Build UI layout
        self.setup_ui()

        # Setup 60 FPS Playback Timer
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(16)  # ~60 FPS (16ms)
        self.playback_timer.timeout.connect(self.on_timer_tick)

        # Load initial files if supplied via CLI
        if neura_path and os.path.exists(neura_path):
            self.load_neura_file(neura_path)
        if baseline_path and os.path.exists(baseline_path):
            self.load_baseline_video(baseline_path)

    def setup_ui(self) -> None:
        """Construct central widgets and control cards."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Top Header Bar
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)

        self.btn_open_neura = QPushButton("📂 Open .neura")
        self.btn_open_neura.setObjectName("PrimaryBtn")
        self.btn_open_neura.clicked.connect(self.on_open_neura_dialog)
        header_layout.addWidget(self.btn_open_neura)

        self.btn_open_baseline = QPushButton("🎬 Open Baseline MP4")
        self.btn_open_baseline.clicked.connect(self.on_open_baseline_dialog)
        header_layout.addWidget(self.btn_open_baseline)

        self.btn_toggle_split = QPushButton("🔀 Split Comparison: OFF")
        self.btn_toggle_split.clicked.connect(self.toggle_split_mode)
        header_layout.addWidget(self.btn_toggle_split)

        self.btn_reset_zoom = QPushButton("🔍 Reset Zoom (1.0x)")
        self.btn_reset_zoom.clicked.connect(self.on_reset_zoom)
        header_layout.addWidget(self.btn_reset_zoom)

        header_layout.addStretch()

        self.lbl_file_info = QLabel("No .neura file loaded")
        self.lbl_file_info.setStyleSheet("color: #8b949e; font-size: 12px;")
        header_layout.addWidget(self.lbl_file_info)

        main_layout.addLayout(header_layout)

        # 2. Central Display Stack (Full View vs Split View)
        self.view_stack = QStackedWidget(self)

        # View 0: Single Continuous Canvas
        self.full_canvas = ContinuousVideoCanvas(self)
        self.full_canvas.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.full_canvas)

        # View 1: Split Comparison View
        self.split_view = SplitComparisonView(self)
        self.split_view.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.split_view)

        main_layout.addWidget(self.view_stack, stretch=1)

        # 3. Diagnostic HUD Bar
        hud_card = QFrame()
        hud_card.setObjectName("ControlCard")
        hud_layout = QHBoxLayout(hud_card)
        hud_layout.setContentsMargins(12, 6, 12, 6)

        self.lbl_hud_stats = QLabel("HUD: Ready")
        self.lbl_hud_stats.setObjectName("HUDLabel")
        hud_layout.addWidget(self.lbl_hud_stats)
        hud_layout.addStretch()

        self.lbl_culling = QLabel("Dynamic Viewport Culling: Active (0.0% compute saved)")
        self.lbl_culling.setStyleSheet("color: #58a6ff; font-family: Consolas; font-size: 11px;")
        hud_layout.addWidget(self.lbl_culling)

        main_layout.addWidget(hud_card)

        # 4. Bottom Playback Control Panel
        control_card = QFrame()
        control_card.setObjectName("ControlCard")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(12, 10, 12, 10)
        control_layout.setSpacing(10)

        # Timeline Scrubber Row
        timeline_layout = QHBoxLayout()
        timeline_layout.setSpacing(12)

        self.lbl_time = QLabel("Frame: 0 / 0 | t = -1.0000")
        self.lbl_time.setObjectName("TimeLabel")
        timeline_layout.addWidget(self.lbl_time)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 10000)
        self.timeline_slider.setValue(0)
        self.timeline_slider.valueChanged.connect(self.on_timeline_slider_changed)
        timeline_layout.addWidget(self.timeline_slider, stretch=1)

        control_layout.addLayout(timeline_layout)

        # Buttons & Controls Row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_restart = QPushButton("⏮ Restart")
        self.btn_restart.clicked.connect(self.on_restart)
        btn_row.addWidget(self.btn_restart)

        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setObjectName("PrimaryBtn")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        btn_row.addWidget(self.btn_play_pause)

        self.chk_loop = QCheckBox("🔁 Loop")
        self.chk_loop.setChecked(True)
        self.chk_loop.toggled.connect(lambda v: setattr(self, "is_looping", v))
        btn_row.addWidget(self.chk_loop)

        btn_row.addSpacing(20)

        lbl_speed = QLabel("Playback Speed:")
        lbl_speed.setStyleSheet("color: #c9d1d9; font-weight: 600;")
        btn_row.addWidget(lbl_speed)

        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.1x (Ultra Slow-Mo)", "0.25x (Slow-Mo)", "0.5x", "1.0x (Normal)", "2.0x", "4.0x"])
        self.cmb_speed.setCurrentIndex(3)  # 1.0x
        self.cmb_speed.currentIndexChanged.connect(self.on_speed_changed)
        btn_row.addWidget(self.cmb_speed)

        btn_row.addSpacing(20)

        lbl_zoom = QLabel("Analytical Zoom:")
        lbl_zoom.setStyleSheet("color: #c9d1d9; font-weight: 600;")
        btn_row.addWidget(lbl_zoom)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 400)
        self.zoom_slider.setValue(1)
        self.zoom_slider.setFixedWidth(140)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        btn_row.addWidget(self.zoom_slider)

        self.lbl_zoom_val = QLabel("1.0x")
        self.lbl_zoom_val.setObjectName("HUDLabel")
        btn_row.addWidget(self.lbl_zoom_val)

        btn_row.addStretch()
        control_layout.addLayout(btn_row)

        main_layout.addWidget(control_card)

    # --- Loading Handlers ---

    def on_open_neura_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open .neura Container", "", "Neura Video Containers (*.neura);;All Files (*)"
        )
        if filepath:
            self.load_neura_file(filepath)

    def on_open_baseline_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Baseline Video", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
        )
        if filepath:
            self.load_baseline_video(filepath)

    def load_neura_file(self, filepath: str) -> None:
        """Instantiate SirenVideo from .neura container on GPU."""
        try:
            model, meta = NeuraReader.load(filepath, device="cuda")
            self.engine = PlayerEngine(model, meta, device="cuda")
            self.neura_meta = meta

            fc = meta.get("frame_count", 96)
            w = meta.get("width", 1280)
            h = meta.get("height", 720)
            size_kb = meta.get("file_size_kb", 0.0)

            self.lbl_file_info.setText(f"📦 {os.path.basename(filepath)} ({size_kb:.1f} KB | {w}x{h} | {fc} frames)")
            self.current_t = -1.0
            self.render_current_frame()
        except Exception as e:
            QMessageBox.critical(self, "Error Loading .neura", f"Failed to load container:\n{e}")

    def load_baseline_video(self, filepath: str) -> None:
        """Cache baseline MP4 frames for real-time discrete split comparison."""
        try:
            cap = cv2.VideoCapture(filepath)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if frames:
                self.discrete_frames = frames
                self.baseline_path = filepath
                self.btn_open_baseline.setText(f"🎬 Baseline: {os.path.basename(filepath)}")
                self.render_current_frame()
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load baseline video:\n{e}")

    # --- Viewport & Navigation Handlers ---

    def on_viewport_changed(self, viewport: ViewportBounds, zoom: float) -> None:
        self.current_viewport = viewport
        self.current_zoom = zoom
        self.lbl_zoom_val.setText(f"{zoom:.1f}x")
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(zoom))
        self.zoom_slider.blockSignals(False)
        self.render_current_frame()

    def on_zoom_slider_changed(self, val: int) -> None:
        zoom = float(val)
        self.full_canvas.set_zoom(zoom)
        self.split_view.set_zoom(zoom)

    def on_reset_zoom(self) -> None:
        self.full_canvas.reset_view()
        self.split_view.reset_view()

    def toggle_split_mode(self) -> None:
        self.is_split_mode = not self.is_split_mode
        if self.is_split_mode:
            self.view_stack.setCurrentIndex(1)
            self.btn_toggle_split.setText("🔀 Split Comparison: ON")
            self.btn_toggle_split.setStyleSheet("background-color: #00e676; color: #000000; font-weight: bold;")
        else:
            self.view_stack.setCurrentIndex(0)
            self.btn_toggle_split.setText("🔀 Split Comparison: OFF")
            self.btn_toggle_split.setStyleSheet("")
        self.render_current_frame()

    # --- Playback Logic ---

    def toggle_play_pause(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.btn_play_pause.setText("▶ Play")
            self.btn_play_pause.setObjectName("PrimaryBtn")
        else:
            if self.engine is None:
                QMessageBox.information(self, "Open .neura", "Please open a .neura container first.")
                return
            self.is_playing = True
            self.playback_timer.start()
            self.btn_play_pause.setText("⏸ Pause")
            self.btn_play_pause.setObjectName("")
        self.style().polish(self.btn_play_pause)

    def on_restart(self) -> None:
        self.current_t = -1.0
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.render_current_frame()

    def on_speed_changed(self, index: int) -> None:
        speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        if 0 <= index < len(speeds):
            self.playback_speed = speeds[index]

    def on_timeline_slider_changed(self, value: int) -> None:
        """Map slider position [0, 10000] to continuous timestamp t in [-1.0, 1.0]."""
        alpha = value / 10000.0
        self.current_t = -1.0 + 2.0 * alpha
        self.render_current_frame()

    def on_timer_tick(self) -> None:
        if not self.is_playing or self.engine is None:
            return

        fc = max(2, self.neura_meta.get("frame_count", 96))
        native_fps = float(self.neura_meta.get("fps", 24.0))

        # Advance continuous timestamp t
        dt = (2.0 / (fc - 1)) * (native_fps / 60.0) * self.playback_speed
        self.current_t += dt

        if self.current_t > 1.0:
            if self.is_looping:
                self.current_t = -1.0
            else:
                self.current_t = 1.0
                self.toggle_play_pause()
                return

        # Update slider without recursive trigger
        alpha = (self.current_t + 1.0) / 2.0
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(int(alpha * 10000))
        self.timeline_slider.blockSignals(False)

        self.render_current_frame()

    # --- Frame Rendering & Display ---

    def render_current_frame(self) -> None:
        if self.engine is None:
            return

        fc = max(2, self.neura_meta.get("frame_count", 96))
        fps = float(self.neura_meta.get("fps", 24.0))

        # Current frame index & time string
        alpha = (self.current_t + 1.0) / 2.0
        curr_frame_float = alpha * (fc - 1)
        curr_sec = curr_frame_float / fps
        total_sec = (fc - 1) / fps

        self.lbl_time.setText(
            f"Frame: {curr_frame_float:5.1f} / {fc} | Time: {curr_sec:.3f}s / {total_sec:.3f}s | t = {self.current_t:+.4f}"
        )

        # Viewport dimensions
        target_widget = self.split_view if self.is_split_mode else self.full_canvas
        w = max(128, target_widget.width())
        h = max(72, target_widget.height())

        # Forward pass through SIREN Viewport Culling Engine
        result: RenderResult = self.engine.render_viewport(
            t_val=self.current_t,
            viewport=self.current_viewport,
            render_width=w,
            render_height=h,
            lod_fast=False,
        )

        # Measure FPS
        self.fps_frame_count += 1
        elapsed = time.perf_counter() - self.fps_start_time
        if elapsed >= 0.5:
            self.last_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.perf_counter()

        hud_str = (
            f"⚡ SIREN-ZIP (.neura INT8: {self.neura_meta.get('file_size_kb', 0.0):.1f} KB) | "
            f"FPS: {self.last_fps:.1f} | Latency: {result.compute_time_ms:.1f}ms | Zoom: {self.current_zoom:.1f}x"
        )
        self.lbl_hud_stats.setText(hud_str)
        self.lbl_culling.setText(f"Dynamic Viewport Culling: {result.culling_ratio_pct:.1f}% compute saved")

        if self.is_split_mode:
            discrete_frame = None
            if self.discrete_frames:
                nearest_idx = int(round(curr_frame_float))
                if nearest_idx < len(self.discrete_frames):
                    discrete_frame = self.discrete_frames[nearest_idx]

            self.split_view.update_buffers(
                siren_rgb=result.rgb_numpy,
                discrete_full_frame=discrete_frame,
                viewport=self.current_viewport,
                hud_text="",
            )
        else:
            self.full_canvas.set_frame_buffer(result.rgb_numpy, hud_text="")


def launch_player_app(
    neura_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
) -> None:
    """Entrypoint to launch Qt GUI application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = SirenPlayerWindow(neura_path=neura_path, baseline_path=baseline_path)
    window.show()
    app.exec()
