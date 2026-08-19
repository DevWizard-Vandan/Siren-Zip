"""Modern Dark-Theme Desktop GUI for Siren-Zip Cinema INR Codec with Audio & HDR."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont
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
    QVBoxLayout,
    QWidget,
)

from src.audio.audio_player import AudioMasterClock
from src.player.engine import PlayerEngine, RenderResult, ViewportBounds
from src.player.neura_reader import NeuraReader
from src.streaming.stream_engine import StreamEngine, StreamRenderResult
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
    """Universal cinema desktop player with Audio Master Clock & HDR10+ tone mapping."""

    def __init__(
        self,
        neura_path: Optional[str] = None,
        baseline_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Siren Player 2.0 - Implicit Neural Cinema Codec (.neura)")
        self.resize(1280, 860)
        self.setStyleSheet(DARK_STYLESHEET)

        # Engine & Readers
        self.stream_engine: Optional[StreamEngine] = None
        self.single_engine: Optional[PlayerEngine] = None
        self.container_version: int = 1
        self.meta: Dict[str, Any] = {}

        # Audio Master Clock
        self.audio_clock = AudioMasterClock(self)
        self.audio_clock.playback_ended.connect(self.on_audio_ended)

        # Baseline MP4
        self.discrete_frames: Optional[list[np.ndarray]] = None
        self.baseline_path: Optional[str] = baseline_path

        # Playback State
        self.is_playing: bool = False
        self.current_global_time: float = 0.0
        self.total_duration: float = 0.0
        self.playback_speed: float = 1.0
        self.is_looping: bool = True
        self.is_split_mode: bool = False
        self.tone_map_mode: str = "aces"

        # Viewport Navigation
        self.current_viewport = ViewportBounds()
        self.current_zoom: float = 1.0

        # Performance & HUD tracking
        self.fps_frame_count: int = 0
        self.fps_start_time: float = time.perf_counter()
        self.last_fps: float = 0.0

        self.setup_ui()

        # 60 FPS Render Loop Timer
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(16)
        self.playback_timer.timeout.connect(self.on_timer_tick)

        if neura_path and os.path.exists(neura_path):
            self.load_neura_container(neura_path)
        if baseline_path and os.path.exists(baseline_path):
            self.load_baseline_video(baseline_path)

    def setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        # 1. Header Toolbar
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

        header_layout.addSpacing(15)

        lbl_hdr = QLabel("HDR / Color:")
        lbl_hdr.setStyleSheet("color: #c9d1d9; font-weight: 600;")
        header_layout.addWidget(lbl_hdr)

        self.cmb_tone_map = QComboBox()
        self.cmb_tone_map.addItems(["ACES Filmic (HDR)", "Reinhard-Jodie", "Reinhard", "SDR (Linear Clamp)"])
        self.cmb_tone_map.currentIndexChanged.connect(self.on_tone_map_changed)
        header_layout.addWidget(self.cmb_tone_map)

        header_layout.addStretch()

        self.lbl_file_info = QLabel("No .neura container loaded")
        self.lbl_file_info.setStyleSheet("color: #8b949e; font-size: 12px;")
        header_layout.addWidget(self.lbl_file_info)

        main_layout.addLayout(header_layout)

        # 2. Central Viewport Stack
        self.view_stack = QStackedWidget(self)

        self.full_canvas = ContinuousVideoCanvas(self)
        self.full_canvas.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.full_canvas)

        self.split_view = SplitComparisonView(self)
        self.split_view.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.split_view)

        main_layout.addWidget(self.view_stack, stretch=1)

        # 3. HUD Stats Bar
        hud_card = QFrame()
        hud_card.setObjectName("ControlCard")
        hud_layout = QHBoxLayout(hud_card)
        hud_layout.setContentsMargins(12, 6, 12, 6)

        self.lbl_hud_stats = QLabel("HUD: Ready")
        self.lbl_hud_stats.setObjectName("HUDLabel")
        hud_layout.addWidget(self.lbl_hud_stats)
        hud_layout.addStretch()

        self.lbl_sync_drift = QLabel("Audio Master Sync: 0.0ms drift")
        self.lbl_sync_drift.setStyleSheet("color: #58a6ff; font-family: Consolas; font-size: 11px;")
        hud_layout.addWidget(self.lbl_sync_drift)

        main_layout.addWidget(hud_card)

        # 4. Playback Controls Card
        control_card = QFrame()
        control_card.setObjectName("ControlCard")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(12, 10, 12, 10)
        control_layout.setSpacing(10)

        # Timeline Scrubber
        timeline_layout = QHBoxLayout()
        timeline_layout.setSpacing(12)

        self.lbl_time = QLabel("00:00.000 / 00:00.000")
        self.lbl_time.setObjectName("TimeLabel")
        timeline_layout.addWidget(self.lbl_time)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 10000)
        self.timeline_slider.setValue(0)
        self.timeline_slider.valueChanged.connect(self.on_timeline_slider_changed)
        timeline_layout.addWidget(self.timeline_slider, stretch=1)

        control_layout.addLayout(timeline_layout)

        # Buttons, Speed, Volume & Zoom Controls
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

        btn_row.addSpacing(15)

        # Audio Volume Controls
        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(38)
        self.btn_mute.clicked.connect(self.toggle_mute)
        btn_row.addWidget(self.btn_mute)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(90)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        btn_row.addWidget(self.volume_slider)

        btn_row.addSpacing(15)

        lbl_speed = QLabel("Speed:")
        lbl_speed.setStyleSheet("color: #c9d1d9; font-weight: 600;")
        btn_row.addWidget(lbl_speed)

        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.1x", "0.25x", "0.5x", "1.0x", "2.0x", "4.0x"])
        self.cmb_speed.setCurrentIndex(3)
        self.cmb_speed.currentIndexChanged.connect(self.on_speed_changed)
        btn_row.addWidget(self.cmb_speed)

        btn_row.addSpacing(15)

        lbl_zoom = QLabel("Analytical Zoom:")
        lbl_zoom.setStyleSheet("color: #c9d1d9; font-weight: 600;")
        btn_row.addWidget(lbl_zoom)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 400)
        self.zoom_slider.setValue(1)
        self.zoom_slider.setFixedWidth(120)
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
            self, "Open .neura Container", "", "Neura Cinema Containers (*.neura);;All Files (*)"
        )
        if filepath:
            self.load_neura_container(filepath)

    def on_open_baseline_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Baseline Video", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
        )
        if filepath:
            self.load_baseline_video(filepath)

    def load_neura_container(self, filepath: str) -> None:
        """Load .neura 1.0 or .neura 2.0 container."""
        try:
            ver = NeuraReader.detect_version(filepath)
            self.container_version = ver

            if ver == 2:
                self.stream_engine = StreamEngine(filepath, device="cuda")
                self.single_engine = None
                self.meta = self.stream_engine.header._asdict()
                self.total_duration = float(self.stream_engine.header.total_duration)

                # Load Audio Payload
                audio_bytes, codec_id, s_rate, ch = self.stream_engine.reader.get_audio_payload()
                codec_str = "aac" if codec_id == 1 else ("opus" if codec_id == 2 else "mp3")
                has_audio = self.audio_clock.load_audio_data(audio_bytes, codec_type=codec_str)

                audio_desc = f" | 🔊 {ch}ch {codec_str.upper()}" if has_audio else " | 🔇 Video Only"
                fc = self.stream_engine.header.total_chunks
                size_kb = os.path.getsize(filepath) / 1024.0
                w = self.stream_engine.header.native_width
                h = self.stream_engine.header.native_height
                self.lbl_file_info.setText(f"📦 .neura 2.0: {os.path.basename(filepath)} ({size_kb:.1f} KB | {w}x{h} | {fc} GOPs{audio_desc})")
            else:
                model, meta = NeuraReader.load(filepath, device="cuda")
                self.single_engine = PlayerEngine(model, meta, device="cuda")
                self.stream_engine = None
                self.meta = meta
                self.total_duration = float(meta.get("frame_count", 96) / meta.get("fps", 24.0))
                size_kb = meta.get("file_size_kb", 0.0)
                w = meta.get("width", 1280)
                h = meta.get("height", 720)
                self.lbl_file_info.setText(f"📦 .neura 1.0: {os.path.basename(filepath)} ({size_kb:.1f} KB | {w}x{h})")

            self.current_global_time = 0.0
            self.render_frame_at_time(0.0)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading .neura", f"Failed to load container:\n{e}")

    def load_baseline_video(self, filepath: str) -> None:
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
                self.render_frame_at_time(self.current_global_time)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load baseline video:\n{e}")

    # --- Viewport Navigation Handlers ---

    def on_viewport_changed(self, viewport: ViewportBounds, zoom: float) -> None:
        self.current_viewport = viewport
        self.current_zoom = zoom
        self.lbl_zoom_val.setText(f"{zoom:.1f}x")
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(zoom))
        self.zoom_slider.blockSignals(False)
        self.render_frame_at_time(self.current_global_time)

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
        self.render_frame_at_time(self.current_global_time)

    def on_tone_map_changed(self, index: int) -> None:
        modes = ["aces", "reinhard_jodie", "reinhard", "linear"]
        if 0 <= index < len(modes):
            self.tone_map_mode = modes[index]
            self.render_frame_at_time(self.current_global_time)

    # --- Audio Controls ---

    def toggle_mute(self) -> None:
        is_muted = self.volume_slider.value() == 0
        if is_muted:
            self.volume_slider.setValue(100)
            self.audio_clock.set_muted(False)
            self.btn_mute.setText("🔊")
        else:
            self.volume_slider.setValue(0)
            self.audio_clock.set_muted(True)
            self.btn_mute.setText("🔇")

    def on_volume_changed(self, val: int) -> None:
        vol = val / 100.0
        self.audio_clock.set_volume(vol)
        if val == 0:
            self.btn_mute.setText("🔇")
        else:
            self.btn_mute.setText("🔊")

    # --- Playback Logic ---

    def toggle_play_pause(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.audio_clock.pause()
            self.btn_play_pause.setText("▶ Play")
            self.btn_play_pause.setObjectName("PrimaryBtn")
        else:
            if self.stream_engine is None and self.single_engine is None:
                QMessageBox.information(self, "Open .neura", "Please open a .neura container first.")
                return
            self.is_playing = True
            self.audio_clock.play()
            self.playback_timer.start()
            self.btn_play_pause.setText("⏸ Pause")
            self.btn_play_pause.setObjectName("")
        self.style().polish(self.btn_play_pause)

    def on_restart(self) -> None:
        self.current_global_time = 0.0
        self.audio_clock.seek(0.0)
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.render_frame_at_time(0.0)

    def on_speed_changed(self, index: int) -> None:
        speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        if 0 <= index < len(speeds):
            self.playback_speed = speeds[index]

    def on_timeline_slider_changed(self, value: int) -> None:
        alpha = value / 10000.0
        t_target = alpha * max(0.001, self.total_duration)
        self.current_global_time = t_target
        self.audio_clock.seek(t_target)
        self.render_frame_at_time(t_target)

    def on_audio_ended(self) -> None:
        if self.is_looping:
            self.on_restart()
            if self.is_playing:
                self.audio_clock.play()
        else:
            self.toggle_play_pause()

    def on_timer_tick(self) -> None:
        if not self.is_playing:
            return

        # Query Authoritative Audio Master Clock
        if self.audio_clock.is_loaded:
            t_master = self.audio_clock.get_master_time()
            self.current_global_time = t_master
        else:
            dt = (1.0 / 60.0) * self.playback_speed
            self.current_global_time += dt

        if self.current_global_time > self.total_duration:
            if self.is_looping:
                self.current_global_time = 0.0
                self.audio_clock.seek(0.0)
            else:
                self.current_global_time = self.total_duration
                self.toggle_play_pause()
                return

        alpha = min(1.0, max(0.0, self.current_global_time / max(0.001, self.total_duration)))
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(int(alpha * 10000))
        self.timeline_slider.blockSignals(False)

        self.render_frame_at_time(self.current_global_time)

    # --- Frame Rendering ---

    def render_frame_at_time(self, t_sec: float) -> None:
        target_widget = self.split_view if self.is_split_mode else self.full_canvas
        w = max(128, target_widget.width())
        h = max(72, target_widget.height())

        curr_m = int(t_sec // 60)
        curr_s = t_sec % 60
        tot_m = int(self.total_duration // 60)
        tot_s = self.total_duration % 60
        self.lbl_time.setText(f"{curr_m:02d}:{curr_s:06.3f} / {tot_m:02d}:{tot_s:06.3f}")

        if self.stream_engine is not None:
            res: StreamRenderResult = self.stream_engine.render_at_time(
                t_global=t_sec,
                viewport=self.current_viewport,
                render_width=w,
                render_height=h,
                tone_map_mode=self.tone_map_mode,
                lod_fast=False,
            )
            rgb_output = res.rgb_numpy
            hdr_str = " | HDR: Rec.2020 PQ" if res.is_hdr_source else ""
            hud_str = (
                f"⚡ SIREN-ZIP 2.0 (Chunk [{res.chunk_idx+1:02d}/{res.total_chunks:02d}]) | "
                f"FPS: {self.last_fps:.1f} | Latency: {res.eval_time_ms:.1f}ms | Zoom: {self.current_zoom:.1f}x{hdr_str}"
            )
        elif self.single_engine is not None:
            # Map global sec to [-1.0, 1.0]
            alpha = min(1.0, max(0.0, t_sec / max(0.001, self.total_duration)))
            t_norm = -1.0 + 2.0 * alpha
            res_single: RenderResult = self.single_engine.render_viewport(
                t_val=t_norm,
                viewport=self.current_viewport,
                render_width=w,
                render_height=h,
                lod_fast=False,
            )
            rgb_output = res_single.rgb_numpy
            hud_str = f"⚡ SIREN-ZIP 1.0 | FPS: {self.last_fps:.1f} | Latency: {res_single.compute_time_ms:.1f}ms | Zoom: {self.current_zoom:.1f}x"
        else:
            return

        # Measure FPS
        self.fps_frame_count += 1
        elapsed = time.perf_counter() - self.fps_start_time
        if elapsed >= 0.5:
            self.last_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.perf_counter()

        self.lbl_hud_stats.setText(hud_str)

        if self.is_split_mode:
            discrete_frame = None
            if self.discrete_frames:
                fps = float(self.meta.get("fps", 24.0))
                idx = min(len(self.discrete_frames) - 1, max(0, int(round(t_sec * fps))))
                discrete_frame = self.discrete_frames[idx]

            self.split_view.update_buffers(
                siren_rgb=rgb_output,
                discrete_full_frame=discrete_frame,
                viewport=self.current_viewport,
            )
        else:
            self.full_canvas.set_frame_buffer(rgb_output)


def launch_player_app(
    neura_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = SirenPlayerWindow(neura_path=neura_path, baseline_path=baseline_path)
    window.show()
    app.exec()
