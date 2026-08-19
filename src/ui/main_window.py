"""Siren-VLC: Feature-Complete Universal Media Player for Implicit Neural Media (.neura)."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
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
from src.filters.video_fx import VideoFXFilter
from src.player.engine import PlayerEngine, RenderResult, ViewportBounds
from src.player.neura_reader import NeuraReader
from src.sharing.share_packer import SharePacker
from src.streaming.stream_engine import StreamEngine, StreamRenderResult
from src.subtitles.subtitle_engine import SubtitleEngine
from src.ui.equalizer_dialog import EqualizerDialog
from src.ui.osd_overlay import OSDOverlay
from src.ui.playlist_widget import PlaylistWidget
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
    padding: 6px 10px;
}
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #c9d1d9;
    padding: 6px 12px;
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
QPushButton#SpecialBtn {
    background-color: #1f6feb;
    border: 1px solid #388bfd;
    color: #ffffff;
}
QPushButton#SpecialBtn:hover {
    background-color: #388bfd;
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
    padding: 4px 8px;
    color: #c9d1d9;
    font-weight: 600;
    font-size: 12px;
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
    """Siren-VLC: Feature-Complete Universal Media Player for Neural Containers."""

    def __init__(
        self,
        neura_path: Optional[str] = None,
        baseline_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Siren-VLC 2.0 - Universal Cinema Media Player (.neura)")
        self.resize(1340, 880)
        self.setStyleSheet(DARK_STYLESHEET)

        # Engines & Decoders
        self.stream_engine: Optional[StreamEngine] = None
        self.single_engine: Optional[PlayerEngine] = None
        self.container_version: int = 2
        self.active_file_path: Optional[str] = None
        self.meta: Dict[str, Any] = {}

        # Audio Master Clock
        self.audio_clock = AudioMasterClock(self)
        self.audio_clock.playback_ended.connect(self.on_audio_ended)

        # Subtitle Engine
        self.subtitle_engine = SubtitleEngine()

        # Video FX Equalizer
        self.video_fx = VideoFXFilter()
        self.equalizer_dlg = EqualizerDialog(self)
        self.equalizer_dlg.filter_changed.connect(self.on_filter_changed)

        # Baseline Comparison
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
        self.is_fullscreen: bool = False

        # Viewport Navigation
        self.current_viewport = ViewportBounds()
        self.current_zoom: float = 1.0

        # Performance tracking
        self.fps_frame_count: int = 0
        self.fps_start_time: float = time.perf_counter()
        self.last_fps: float = 0.0

        self.setup_ui()

        # OSD Overlay
        self.osd = OSDOverlay(self.view_stack)

        # 60 FPS Render Loop Timer
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(16)
        self.playback_timer.timeout.connect(self.on_timer_tick)

        if neura_path and os.path.exists(neura_path):
            self.load_media_file(neura_path)
            self.playlist_dock.add_file(neura_path)
        if baseline_path and os.path.exists(baseline_path):
            self.load_baseline_video(baseline_path)

    def setup_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(8)

        # 1. Header Toolbar
        self.header_frame = QFrame()
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.btn_open_neura = QPushButton("📂 Open Media")
        self.btn_open_neura.setObjectName("PrimaryBtn")
        self.btn_open_neura.clicked.connect(self.on_open_file_dialog)
        header_layout.addWidget(self.btn_open_neura)

        self.btn_open_subtitles = QPushButton("💬 Subtitles")
        self.btn_open_subtitles.clicked.connect(self.on_open_subtitles_dialog)
        header_layout.addWidget(self.btn_open_subtitles)

        self.btn_open_baseline = QPushButton("🎬 Baseline MP4")
        self.btn_open_baseline.clicked.connect(self.on_open_baseline_dialog)
        header_layout.addWidget(self.btn_open_baseline)

        self.btn_toggle_split = QPushButton("🔀 Split View: OFF")
        self.btn_toggle_split.clicked.connect(self.toggle_split_mode)
        header_layout.addWidget(self.btn_toggle_split)

        self.btn_equalizer = QPushButton("🎛️ Equalizer (E)")
        self.btn_equalizer.clicked.connect(self.toggle_equalizer)
        header_layout.addWidget(self.btn_equalizer)

        self.btn_share = QPushButton("📤 Share (WhatsApp)")
        self.btn_share.setObjectName("SpecialBtn")
        self.btn_share.clicked.connect(self.on_share_package)
        header_layout.addWidget(self.btn_share)

        header_layout.addSpacing(10)

        lbl_hdr = QLabel("HDR / Color:")
        lbl_hdr.setStyleSheet("color: #c9d1d9; font-weight: 600;")
        header_layout.addWidget(lbl_hdr)

        self.cmb_tone_map = QComboBox()
        self.cmb_tone_map.addItems(["ACES Filmic (HDR)", "Reinhard-Jodie", "Reinhard", "SDR (Linear)"])
        self.cmb_tone_map.currentIndexChanged.connect(self.on_tone_map_changed)
        header_layout.addWidget(self.cmb_tone_map)

        header_layout.addStretch()

        self.btn_playlist = QPushButton("📋 Playlist (P)")
        self.btn_playlist.clicked.connect(self.toggle_playlist)
        header_layout.addWidget(self.btn_playlist)

        self.btn_fullscreen = QPushButton("⛶ Fullscreen (F)")
        self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        header_layout.addWidget(self.btn_fullscreen)

        self.main_layout.addWidget(self.header_frame)

        # 2. Central Viewport Stack
        self.view_stack = QStackedWidget(self)

        self.full_canvas = ContinuousVideoCanvas(self)
        self.full_canvas.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.full_canvas)

        self.split_view = SplitComparisonView(self)
        self.split_view.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.split_view)

        self.main_layout.addWidget(self.view_stack, stretch=1)

        # 3. HUD Stats Bar
        self.hud_card = QFrame()
        self.hud_card.setObjectName("ControlCard")
        hud_layout = QHBoxLayout(self.hud_card)
        hud_layout.setContentsMargins(10, 4, 10, 4)

        self.lbl_hud_stats = QLabel("⚡ Siren-VLC: Ready")
        self.lbl_hud_stats.setObjectName("HUDLabel")
        hud_layout.addWidget(self.lbl_hud_stats)
        hud_layout.addStretch()

        self.lbl_sync_drift = QLabel("Audio Master Sync: 0.0ms drift")
        self.lbl_sync_drift.setStyleSheet("color: #58a6ff; font-family: Consolas; font-size: 11px;")
        hud_layout.addWidget(self.lbl_sync_drift)

        self.main_layout.addWidget(self.hud_card)

        # 4. Playback Controls Card
        self.control_card = QFrame()
        self.control_card.setObjectName("ControlCard")
        control_layout = QVBoxLayout(self.control_card)
        control_layout.setContentsMargins(10, 8, 10, 8)
        control_layout.setSpacing(6)

        # Timeline Scrubber
        timeline_layout = QHBoxLayout()
        timeline_layout.setSpacing(10)

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
        btn_row.setSpacing(8)

        self.btn_prev = QPushButton("⏮ Prev")
        self.btn_prev.clicked.connect(self.on_play_previous)
        btn_row.addWidget(self.btn_prev)

        self.btn_play_pause = QPushButton("▶ Play (Space)")
        self.btn_play_pause.setObjectName("PrimaryBtn")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        btn_row.addWidget(self.btn_play_pause)

        self.btn_next = QPushButton("⏭ Next")
        self.btn_next.clicked.connect(self.on_play_next)
        btn_row.addWidget(self.btn_next)

        self.chk_loop = QCheckBox("🔁 Loop")
        self.chk_loop.setChecked(True)
        self.chk_loop.toggled.connect(lambda v: setattr(self, "is_looping", v))
        btn_row.addWidget(self.chk_loop)

        btn_row.addSpacing(10)

        # Audio Volume Controls
        self.btn_mute = QPushButton("🔊 (M)")
        self.btn_mute.setFixedWidth(54)
        self.btn_mute.clicked.connect(self.toggle_mute)
        btn_row.addWidget(self.btn_mute)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        btn_row.addWidget(self.volume_slider)

        btn_row.addSpacing(10)

        lbl_speed = QLabel("Speed:")
        lbl_speed.setStyleSheet("color: #c9d1d9; font-weight: 600;")
        btn_row.addWidget(lbl_speed)

        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.1x", "0.25x", "0.5x", "1.0x", "2.0x", "4.0x"])
        self.cmb_speed.setCurrentIndex(3)
        self.cmb_speed.currentIndexChanged.connect(self.on_speed_changed)
        btn_row.addWidget(self.cmb_speed)

        btn_row.addSpacing(10)

        lbl_zoom = QLabel("400X Zoom:")
        lbl_zoom.setStyleSheet("color: #c9d1d9; font-weight: 600;")
        btn_row.addWidget(lbl_zoom)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 400)
        self.zoom_slider.setValue(1)
        self.zoom_slider.setFixedWidth(90)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        btn_row.addWidget(self.zoom_slider)

        self.lbl_zoom_val = QLabel("1.0x")
        self.lbl_zoom_val.setObjectName("HUDLabel")
        btn_row.addWidget(self.lbl_zoom_val)

        btn_row.addStretch()

        self.btn_screenshot = QPushButton("📸 4K Shot (S)")
        self.btn_screenshot.clicked.connect(self.take_high_res_screenshot)
        btn_row.addWidget(self.btn_screenshot)

        control_layout.addLayout(btn_row)
        self.main_layout.addWidget(self.control_card)

        # 5. Playlist Dock
        self.playlist_dock = PlaylistWidget(self)
        self.playlist_dock.file_selected.connect(self.load_media_file)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.playlist_dock)
        self.playlist_dock.hide()

    # --- Keyboard Shortcuts ---

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif key == Qt.Key.Key_Left:
            dt = 1.0 if modifiers & Qt.KeyboardModifier.ShiftModifier else 5.0
            self.seek_relative(-dt)
        elif key == Qt.Key.Key_Right:
            dt = 1.0 if modifiers & Qt.KeyboardModifier.ShiftModifier else 5.0
            self.seek_relative(+dt)
        elif key == Qt.Key.Key_Up:
            self.adjust_volume(+5)
        elif key == Qt.Key.Key_Down:
            self.adjust_volume(-5)
        elif key in (Qt.Key.Key_F, Qt.Key.Key_F11):
            self.toggle_fullscreen()
        elif key == Qt.Key.Key_M:
            self.toggle_mute()
        elif key == Qt.Key.Key_S:
            self.take_high_res_screenshot()
        elif key == Qt.Key.Key_P:
            self.toggle_playlist()
        elif key == Qt.Key.Key_E:
            self.toggle_equalizer()
        elif key == Qt.Key.Key_Escape and self.is_fullscreen:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

    # --- Loading Handlers ---

    def on_open_file_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Media File", "", "Siren & Video Files (*.neura *.mp4 *.mkv *.avi);;All Files (*)"
        )
        if filepath:
            self.load_media_file(filepath)
            self.playlist_dock.add_file(filepath)

    def on_open_subtitles_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Subtitles", "", "Subtitle Files (*.srt *.vtt);;All Files (*)"
        )
        if filepath:
            if self.subtitle_engine.load_file(filepath):
                self.osd.show_notification(f"💬 Subtitles Loaded: {os.path.basename(filepath)}")
                self.btn_open_subtitles.setText(f"💬 {os.path.basename(filepath)[:12]}")
                self.render_frame_at_time(self.current_global_time)

    def on_open_baseline_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Baseline Video", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)"
        )
        if filepath:
            self.load_baseline_video(filepath)

    def load_media_file(self, filepath: str) -> None:
        """Universal loader for .neura 2.0, .neura 1.0, and standard MP4/MKV files."""
        try:
            self.active_file_path = filepath
            if filepath.endswith(".neura"):
                ver = NeuraReader.detect_version(filepath)
                self.container_version = ver

                if ver == 2:
                    self.stream_engine = StreamEngine(filepath, device="cuda")
                    self.single_engine = None
                    self.meta = self.stream_engine.header._asdict()
                    self.total_duration = float(self.stream_engine.header.total_duration)

                    audio_bytes, codec_id, s_rate, ch = self.stream_engine.reader.get_audio_payload()
                    codec_str = "aac" if codec_id == 1 else ("opus" if codec_id == 2 else "mp3")
                    self.audio_clock.load_audio_data(audio_bytes, codec_type=codec_str)

                    fc = self.stream_engine.header.total_chunks
                    size_kb = os.path.getsize(filepath) / 1024.0
                    w = self.stream_engine.header.native_width
                    h = self.stream_engine.header.native_height
                    self.osd.show_notification(f"🎬 Loaded .neura 2.0 ({w}x{h} | {size_kb/1024:.2f} MB)")
                else:
                    model, meta = NeuraReader.load(filepath, device="cuda")
                    self.single_engine = PlayerEngine(model, meta, device="cuda")
                    self.stream_engine = None
                    self.meta = meta
                    self.total_duration = float(meta.get("frame_count", 96) / meta.get("fps", 24.0))
                    self.osd.show_notification(f"🎬 Loaded .neura 1.0 ({meta.get('file_size_kb', 0):.1f} KB)")
            else:
                # Video file
                self.load_baseline_video(filepath)
                self.osd.show_notification(f"🎬 Baseline Loaded: {os.path.basename(filepath)}")

            self.current_global_time = 0.0
            self.render_frame_at_time(0.0)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Media", f"Could not load media:\n{e}")

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
                self.btn_open_baseline.setText(f"🎬 Baseline: {os.path.basename(filepath)[:10]}")
                self.render_frame_at_time(self.current_global_time)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load baseline video:\n{e}")

    # --- UI Toggles & Actions ---

    def toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.header_frame.hide()
            self.hud_card.hide()
            self.showFullScreen()
            self.osd.show_notification("⛶ Fullscreen (Press F or Esc to exit)")
        else:
            self.header_frame.show()
            self.hud_card.show()
            self.showNormal()
            self.osd.show_notification("⛶ Windowed Mode")

    def toggle_playlist(self) -> None:
        self.playlist_dock.setVisible(not self.playlist_dock.isVisible())

    def toggle_equalizer(self) -> None:
        self.equalizer_dlg.show()
        self.equalizer_dlg.raise_()

    def on_filter_changed(self, fx: VideoFXFilter) -> None:
        self.video_fx = fx
        self.render_frame_at_time(self.current_global_time)

    def on_play_next(self) -> None:
        next_file = self.playlist_dock.get_next_file()
        if next_file:
            self.load_media_file(next_file)

    def on_play_previous(self) -> None:
        prev_file = self.playlist_dock.get_previous_file()
        if prev_file:
            self.load_media_file(prev_file)

    def on_share_package(self) -> None:
        if not self.active_file_path or not self.active_file_path.endswith(".neura"):
            QMessageBox.information(self, "Share Package", "Please load an active .neura container first.")
            return

        zip_path, _ = QFileDialog.getSaveFileName(
            self, "Save WhatsApp / Cold-Storage Bundle", "cinema_whatsapp_bundle.zip", "ZIP Archives (*.zip)"
        )
        if zip_path:
            res = SharePacker.create_share_bundle(self.active_file_path, zip_path, platform="whatsapp")
            QMessageBox.information(
                self,
                "WhatsApp Bundle Created",
                f"Bundle saved to: {os.path.basename(zip_path)}\n"
                f"Bundle Size: {res['bundle_size_mb']:.2f} MB\n"
                f"Status: {res['compliance']['status']}\n\n"
                "You can now attach this ZIP directly in WhatsApp or Telegram!",
            )

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
        self.osd.show_notification(f"🔍 Zoom: {zoom:.1f}x")

    def toggle_split_mode(self) -> None:
        self.is_split_mode = not self.is_split_mode
        if self.is_split_mode:
            self.view_stack.setCurrentIndex(1)
            self.btn_toggle_split.setText("🔀 Split View: ON")
            self.btn_toggle_split.setStyleSheet("background-color: #00e676; color: #000000; font-weight: bold;")
            self.osd.show_notification("🔀 Split Comparison: ON")
        else:
            self.view_stack.setCurrentIndex(0)
            self.btn_toggle_split.setText("🔀 Split View: OFF")
            self.btn_toggle_split.setStyleSheet("")
            self.osd.show_notification("🔀 Split Comparison: OFF")
        self.render_frame_at_time(self.current_global_time)

    def on_tone_map_changed(self, index: int) -> None:
        modes = ["aces", "reinhard_jodie", "reinhard", "linear"]
        if 0 <= index < len(modes):
            self.tone_map_mode = modes[index]
            self.osd.show_notification(f"🎨 Color: {self.cmb_tone_map.currentText()}")
            self.render_frame_at_time(self.current_global_time)

    # --- Audio Controls ---

    def toggle_mute(self) -> None:
        is_muted = self.volume_slider.value() == 0
        if is_muted:
            self.volume_slider.setValue(100)
            self.audio_clock.set_muted(False)
            self.btn_mute.setText("🔊 (M)")
            self.osd.show_notification("🔊 Unmuted (100%)")
        else:
            self.volume_slider.setValue(0)
            self.audio_clock.set_muted(True)
            self.btn_mute.setText("🔇 (M)")
            self.osd.show_notification("🔇 Muted")

    def adjust_volume(self, delta: int) -> None:
        curr = self.volume_slider.value()
        new_val = max(0, min(100, curr + delta))
        self.volume_slider.setValue(new_val)
        self.osd.show_notification(f"🔊 Volume: {new_val}%")

    def on_volume_changed(self, val: int) -> None:
        vol = val / 100.0
        self.audio_clock.set_volume(vol)
        if val == 0:
            self.btn_mute.setText("🔇 (M)")
        else:
            self.btn_mute.setText("🔊 (M)")

    # --- Playback Logic ---

    def toggle_play_pause(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.audio_clock.pause()
            self.btn_play_pause.setText("▶ Play (Space)")
            self.btn_play_pause.setObjectName("PrimaryBtn")
            self.osd.show_notification("⏸ Paused")
        else:
            if self.stream_engine is None and self.single_engine is None:
                QMessageBox.information(self, "Open Media", "Please open a media file or playlist item first.")
                return
            self.is_playing = True
            self.audio_clock.play()
            self.playback_timer.start()
            self.btn_play_pause.setText("⏸ Pause (Space)")
            self.btn_play_pause.setObjectName("")
            self.osd.show_notification("▶ Playing")
        self.style().polish(self.btn_play_pause)

    def seek_relative(self, delta_sec: float) -> None:
        new_time = max(0.0, min(self.total_duration, self.current_global_time + delta_sec))
        self.current_global_time = new_time
        self.audio_clock.seek(new_time)
        self.render_frame_at_time(new_time)
        sign = "+" if delta_sec > 0 else ""
        self.osd.show_notification(f"⏩ Seek {sign}{delta_sec:.1f}s ({new_time:.1f}s / {self.total_duration:.1f}s)")

    def on_speed_changed(self, index: int) -> None:
        speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
        if 0 <= index < len(speeds):
            self.playback_speed = speeds[index]
            self.osd.show_notification(f"⚡ Speed: {self.playback_speed:.2f}x")

    def on_timeline_slider_changed(self, value: int) -> None:
        alpha = value / 10000.0
        t_target = alpha * max(0.001, self.total_duration)
        self.current_global_time = t_target
        self.audio_clock.seek(t_target)
        self.render_frame_at_time(t_target)

    def on_audio_ended(self) -> None:
        if self.is_looping:
            self.current_global_time = 0.0
            self.audio_clock.seek(0.0)
            if self.is_playing:
                self.audio_clock.play()
        else:
            self.on_play_next()

    def on_timer_tick(self) -> None:
        if not self.is_playing:
            return

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
                self.on_play_next()
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
            raw_rgb = res.rgb_numpy
            pre_str = " (Prefetched)" if res.is_prefetched else ""
            hud_str = (
                f"⚡ SIREN-ZIP 2.0 (Chunk [{res.chunk_idx+1:02d}/{res.total_chunks:02d}]{pre_str}) | "
                f"FPS: {self.last_fps:.1f} | Latency: {res.eval_time_ms:.1f}ms | Zoom: {self.current_zoom:.1f}x"
            )
        elif self.single_engine is not None:
            alpha = min(1.0, max(0.0, t_sec / max(0.001, self.total_duration)))
            t_norm = -1.0 + 2.0 * alpha
            res_single: RenderResult = self.single_engine.render_viewport(
                t_val=t_norm,
                viewport=self.current_viewport,
                render_width=w,
                render_height=h,
                lod_fast=False,
            )
            raw_rgb = res_single.rgb_numpy
            hud_str = f"⚡ SIREN-ZIP 1.0 | FPS: {self.last_fps:.1f} | Latency: {res_single.compute_time_ms:.1f}ms | Zoom: {self.current_zoom:.1f}x"
        else:
            return

        # 1. Apply Video Equalizer / Color FX
        processed_rgb = self.video_fx.apply(raw_rgb)

        # 2. Render Vector Subtitles if active
        final_rgb = self.subtitle_engine.render_overlay(processed_rgb, current_sec=t_sec)

        # 3. Measure FPS
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
                siren_rgb=final_rgb,
                discrete_full_frame=discrete_frame,
                viewport=self.current_viewport,
            )
        else:
            self.full_canvas.set_frame_buffer(final_rgb)

    def take_high_res_screenshot(self) -> None:
        """Render continuous mathematical coordinate field at full 4K UHD (3840x2160) and save to runs/."""
        if self.stream_engine is None and self.single_engine is None:
            return

        os.makedirs("runs", exist_ok=True)
        shot_path = os.path.join("runs", f"screenshot_{int(time.time())}.png")

        # Evaluate at 4K resolution (3840x2160)
        shot_w, shot_h = 3840, 2160
        self.osd.show_notification(f"📸 Generating 4K Continuous Shot...")
        QApplication.processEvents()

        if self.stream_engine is not None:
            res = self.stream_engine.render_at_time(
                t_global=self.current_global_time,
                viewport=self.current_viewport,
                render_width=shot_w,
                render_height=shot_h,
                tone_map_mode=self.tone_map_mode,
            )
            raw = res.rgb_numpy
        else:
            alpha = min(1.0, max(0.0, self.current_global_time / max(0.001, self.total_duration)))
            res = self.single_engine.render_viewport(
                t_val=-1.0 + 2.0 * alpha,
                viewport=self.current_viewport,
                render_width=shot_w,
                render_height=shot_h,
            )
            raw = res.rgb_numpy

        processed = self.video_fx.apply(raw)
        bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(shot_path, bgr)

        self.osd.show_notification(f"📸 4K Shot Saved: {os.path.basename(shot_path)}", duration_ms=2500)


def launch_player_app(
    neura_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = SirenPlayerWindow(neura_path=neura_path, baseline_path=baseline_path)
    window.show()
    app.exec()
