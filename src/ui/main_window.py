"""Siren-VLC: Universal Implicit Neural Media Player with Classic VLC UI/UX."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QImage, QKeyEvent, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMenuBar,
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
from src.ui.clipper_dialog import ClipperDialog
from src.ui.equalizer_dialog import EqualizerDialog
from src.ui.open_url_dialog import OpenURLDialog
from src.ui.osd_overlay import OSDOverlay
from src.ui.playlist_widget import PlaylistWidget
from src.ui.split_view import SplitComparisonView
from src.ui.telemetry_overlay import TelemetryOverlay
from src.ui.video_canvas import ContinuousVideoCanvas

DARK_VLC_STYLESHEET = """
QMainWindow {
    background-color: #0b0f14;
    color: #e6edf3;
}
QMenuBar {
    background-color: #161b22;
    color: #c9d1d9;
    border-bottom: 1px solid #30363d;
    font-size: 12px;
    padding: 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #21262d;
    color: #58a6ff;
}
QMenu {
    background-color: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #00e676;
    color: #000000;
    font-weight: bold;
}
QMenu::separator {
    height: 1px;
    background-color: #30363d;
    margin: 4px 8px;
}
QWidget {
    background-color: #0b0f14;
    color: #e6edf3;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QFrame#VLCBottomBar {
    background-color: #161b22;
    border-top: 1px solid #30363d;
    padding: 6px 12px;
}
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    border-radius: 5px;
    padding: 5px 10px;
    font-weight: 600;
    font-size: 11px;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
    color: #ffffff;
}
QPushButton#PlayBtn {
    background-color: #238636;
    color: #ffffff;
    font-weight: bold;
    border-color: #2ea043;
    min-width: 70px;
}
QPushButton#PlayBtn:hover {
    background-color: #2ea043;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #30363d;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #00e676;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #00e676;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #00e676;
    border-color: #ffffff;
}
QLabel#TimeLabel {
    color: #8b949e;
    font-family: 'Consolas', monospace;
    font-size: 11px;
}
QLabel#HUDLabel {
    color: #00e676;
    font-family: 'Consolas', monospace;
    font-size: 11px;
    font-weight: bold;
}
"""


class SirenPlayerWindow(QMainWindow):
    """Siren-VLC: Universal Implicit Neural Media Player with Classic VLC Simplicity."""

    def __init__(
        self,
        neura_path: Optional[str] = None,
        baseline_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Siren-VLC Media Player")
        self.resize(1280, 800)
        self.setStyleSheet(DARK_VLC_STYLESHEET)

        # State & Engines
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

        # Video Equalizer
        self.video_fx = VideoFXFilter()
        self.equalizer_dlg = EqualizerDialog(self, initial_fx=self.video_fx)
        self.equalizer_dlg.filter_changed.connect(self.on_filter_changed)

        # Baseline comparison
        self.baseline_cap: Optional[cv2.VideoCapture] = None
        self.baseline_fps: float = 24.0
        self.baseline_path: Optional[str] = baseline_path
        self.is_split_mode: bool = False

        # Playback Timeline State
        self.is_playing: bool = False
        self.is_looping: bool = True
        self.current_global_time: float = 0.0
        self.total_duration: float = 12.0
        self.playback_speed: float = 1.0
        self.tone_map_mode: str = "aces"
        self.is_fullscreen: bool = False
        self.is_pip_mode: bool = False
        self.normal_geometry = self.geometry()

        # Viewport Navigation
        self.current_viewport = ViewportBounds()
        self.current_zoom: float = 1.0

        # Performance tracking
        self.fps_frame_count: int = 0
        self.fps_start_time: float = time.perf_counter()
        self.last_fps: float = 0.0

        self.setup_ui()

        # OSD Overlay & Examiner Telemetry HUD
        self.osd = OSDOverlay(self.view_stack)
        self.telemetry = TelemetryOverlay(self.view_stack)

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
        # 1. Classic VLC Menu Bar
        self.setup_menu_bar()

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 2. Central Clean Video Canvas
        self.view_stack = QStackedWidget(self)
        self.full_canvas = ContinuousVideoCanvas(self)
        self.full_canvas.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.full_canvas)

        self.split_view = SplitComparisonView(self)
        self.split_view.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.split_view)

        self.main_layout.addWidget(self.view_stack, stretch=1)

        # 3. Classic VLC Bottom Control Bar
        self.bottom_bar = QFrame()
        self.bottom_bar.setObjectName("VLCBottomBar")
        bar_layout = QVBoxLayout(self.bottom_bar)
        bar_layout.setContentsMargins(12, 6, 12, 8)
        bar_layout.setSpacing(6)

        # Line 1: Timeline Scrubber + Time Label
        row_time = QHBoxLayout()
        row_time.setSpacing(8)
        self.lbl_time = QLabel("00:00.000 / 00:00.000")
        self.lbl_time.setObjectName("TimeLabel")
        row_time.addWidget(self.lbl_time)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 10000)
        self.timeline_slider.setValue(0)
        self.timeline_slider.valueChanged.connect(self.on_timeline_slider_changed)
        row_time.addWidget(self.timeline_slider, stretch=1)
        bar_layout.addLayout(row_time)

        # Line 2: VLC Control Buttons
        row_controls = QHBoxLayout()
        row_controls.setSpacing(6)

        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setObjectName("PlayBtn")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        row_controls.addWidget(self.btn_play_pause)

        btn_stop = QPushButton("⏹")
        btn_stop.setFixedWidth(32)
        btn_stop.clicked.connect(self.stop_playback)
        row_controls.addWidget(btn_stop)

        btn_prev = QPushButton("⏮")
        btn_prev.setFixedWidth(32)
        btn_prev.clicked.connect(self.on_play_previous)
        row_controls.addWidget(btn_prev)

        btn_next = QPushButton("⏭")
        btn_next.setFixedWidth(32)
        btn_next.clicked.connect(self.on_play_next)
        row_controls.addWidget(btn_next)

        row_controls.addSpacing(6)

        btn_fs = QPushButton("⛶")
        btn_fs.setToolTip("Fullscreen (F)")
        btn_fs.clicked.connect(self.toggle_fullscreen)
        row_controls.addWidget(btn_fs)

        btn_pip = QPushButton("📌 PiP")
        btn_pip.setToolTip("Picture-in-Picture (Ctrl+P)")
        btn_pip.clicked.connect(self.toggle_pip_mode)
        row_controls.addWidget(btn_pip)

        btn_eq = QPushButton("🎛️ EQ")
        btn_eq.setToolTip("Equalizer & Video FX (Ctrl+E)")
        btn_eq.clicked.connect(self.toggle_equalizer)
        row_controls.addWidget(btn_eq)

        btn_clip = QPushButton("✂️ Clip")
        btn_clip.setToolTip("One-Click GIF/WebM Clipper (Ctrl+R)")
        btn_clip.clicked.connect(self.open_clipper_dialog)
        row_controls.addWidget(btn_clip)

        btn_web = QPushButton("🌐 URL")
        btn_web.setToolTip("Open YouTube / Web Stream (Ctrl+N)")
        btn_web.clicked.connect(self.open_network_stream_dialog)
        row_controls.addWidget(btn_web)

        btn_list = QPushButton("📋 Queue")
        btn_list.setToolTip("Playlist Dock (Ctrl+L)")
        btn_list.clicked.connect(self.toggle_playlist)
        row_controls.addWidget(btn_list)

        row_controls.addSpacing(10)

        # Volume Controls
        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(34)
        self.btn_mute.clicked.connect(self.toggle_mute)
        row_controls.addWidget(self.btn_mute)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(75)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        row_controls.addWidget(self.volume_slider)

        row_controls.addStretch()

        # HUD Status Badge
        self.lbl_hud_stats = QLabel("⚡ SIREN-ZIP 2.0 | 60 FPS | A/V 0.0ms")
        self.lbl_hud_stats.setObjectName("HUDLabel")
        row_controls.addWidget(self.lbl_hud_stats)

        bar_layout.addLayout(row_controls)
        self.main_layout.addWidget(self.bottom_bar)

        # 4. Playlist Dock (hidden by default)
        self.playlist_dock = PlaylistWidget(self)
        self.playlist_dock.file_selected.connect(self.load_media_file)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.playlist_dock)
        self.playlist_dock.hide()

    def setup_menu_bar(self) -> None:
        menubar = self.menuBar()

        # --- &Media Menu ---
        media_menu = menubar.addMenu("&Media")
        
        act_open_file = QAction("Open &File...", self)
        act_open_file.setShortcut(QKeySequence("Ctrl+O"))
        act_open_file.triggered.connect(self.on_open_file_dialog)
        media_menu.addAction(act_open_file)

        act_open_url = QAction("Open &Network Stream (YouTube)...", self)
        act_open_url.setShortcut(QKeySequence("Ctrl+N"))
        act_open_url.triggered.connect(self.open_network_stream_dialog)
        media_menu.addAction(act_open_url)

        act_open_base = QAction("Open Baseline &MP4 for Comparison...", self)
        act_open_base.triggered.connect(self.on_open_baseline_dialog)
        media_menu.addAction(act_open_base)

        media_menu.addSeparator()

        act_clip = QAction("Convert / &Clip GIF & WebM...", self)
        act_clip.setShortcut(QKeySequence("Ctrl+R"))
        act_clip.triggered.connect(self.open_clipper_dialog)
        media_menu.addAction(act_clip)

        act_share = QAction("&Save / Share WhatsApp Bundle...", self)
        act_share.setShortcut(QKeySequence("Ctrl+S"))
        act_share.triggered.connect(self.on_share_package)
        media_menu.addAction(act_share)

        media_menu.addSeparator()

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)
        media_menu.addAction(act_quit)

        # --- &Playback Menu ---
        play_menu = menubar.addMenu("&Playback")

        act_play = QAction("&Play / Pause", self)
        act_play.setShortcut(QKeySequence("Space"))
        act_play.triggered.connect(self.toggle_play_pause)
        play_menu.addAction(act_play)

        act_stop = QAction("&Stop", self)
        act_stop.triggered.connect(self.stop_playback)
        play_menu.addAction(act_stop)

        act_prev = QAction("&Previous Track", self)
        act_prev.setShortcut(QKeySequence("P"))
        act_prev.triggered.connect(self.on_play_previous)
        play_menu.addAction(act_prev)

        act_next = QAction("&Next Track", self)
        act_next.setShortcut(QKeySequence("N"))
        act_next.triggered.connect(self.on_play_next)
        play_menu.addAction(act_next)

        play_menu.addSeparator()

        speed_menu = play_menu.addMenu("&Speed")
        for sp in [0.25, 0.5, 1.0, 2.0, 4.0]:
            act_sp = QAction(f"{sp}x", self)
            act_sp.triggered.connect(lambda _, s=sp: self.set_playback_speed(s))
            speed_menu.addAction(act_sp)

        # --- &Audio Menu ---
        audio_menu = menubar.addMenu("&Audio")

        act_mute = QAction("&Mute", self)
        act_mute.setShortcut(QKeySequence("M"))
        act_mute.triggered.connect(self.toggle_mute)
        audio_menu.addAction(act_mute)

        act_night = QAction("&Night Mode (Dialogue Booster / Limiter)", self)
        act_night.setShortcut(QKeySequence("Ctrl+D"))
        act_night.setCheckable(True)
        act_night.triggered.connect(self.toggle_night_mode)
        audio_menu.addAction(act_night)

        # --- &Video Menu ---
        video_menu = menubar.addMenu("&Video")

        act_fs = QAction("&Fullscreen", self)
        act_fs.setShortcut(QKeySequence("F"))
        act_fs.triggered.connect(self.toggle_fullscreen)
        video_menu.addAction(act_fs)

        act_pip = QAction("&Picture-in-Picture (Floating Mode)", self)
        act_pip.setShortcut(QKeySequence("Ctrl+P"))
        act_pip.triggered.connect(self.toggle_pip_mode)
        video_menu.addAction(act_pip)

        act_shot = QAction("Take &4K UHD Screenshot", self)
        act_shot.setShortcut(QKeySequence("S"))
        act_shot.triggered.connect(self.take_high_res_screenshot)
        video_menu.addAction(act_shot)

        act_split = QAction("Toggle &Split Comparison View", self)
        act_split.setShortcut(QKeySequence("Ctrl+T"))
        act_split.triggered.connect(self.toggle_split_mode)
        video_menu.addAction(act_split)

        video_menu.addSeparator()

        hdr_menu = video_menu.addMenu("&HDR / Color Tone-Mapping")
        for mode, name in [("aces", "ACES Filmic (HDR)"), ("reinhard_jodie", "Reinhard-Jodie"), ("reinhard", "Reinhard"), ("linear", "SDR Linear")]:
            act_hdr = QAction(name, self)
            act_hdr.triggered.connect(lambda _, m=mode, n=name: self.set_tone_mapping(m, n))
            hdr_menu.addAction(act_hdr)

        # --- &Subtitle Menu ---
        sub_menu = menubar.addMenu("&Subtitle")

        act_add_sub = QAction("&Add Subtitle File...", self)
        act_add_sub.triggered.connect(self.on_open_subtitles_dialog)
        sub_menu.addAction(act_add_sub)

        # --- &Tools Menu ---
        tools_menu = menubar.addMenu("&Tools")

        act_eq = QAction("&Effects and Filters / Equalizer", self)
        act_eq.setShortcut(QKeySequence("Ctrl+E"))
        act_eq.triggered.connect(self.toggle_equalizer)
        tools_menu.addAction(act_eq)

        act_telem = QAction("Examiner &Telemetry HUD", self)
        act_telem.setShortcut(QKeySequence("F12"))
        act_telem.triggered.connect(self.toggle_telemetry)
        tools_menu.addAction(act_telem)

        # --- &View Menu ---
        view_menu = menubar.addMenu("&View")

        act_list = QAction("&Playlist", self)
        act_list.setShortcut(QKeySequence("Ctrl+L"))
        act_list.triggered.connect(self.toggle_playlist)
        view_menu.addAction(act_list)

        # --- &Help Menu ---
        help_menu = menubar.addMenu("&Help")

        act_about = QAction("&About Siren-VLC", self)
        act_about.triggered.connect(self.show_about_dialog)
        help_menu.addAction(act_about)

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
        elif key == Qt.Key.Key_F12:
            self.toggle_telemetry()
        elif key == Qt.Key.Key_Escape:
            if self.is_fullscreen:
                self.toggle_fullscreen()
            elif self.is_pip_mode:
                self.toggle_pip_mode()
        else:
            super().keyPressEvent(event)

    # --- Dialog Openers ---

    def open_clipper_dialog(self) -> None:
        dlg = ClipperDialog(
            self,
            get_current_time_cb=lambda: self.current_global_time,
            get_video_path_cb=lambda: self.baseline_path or self.active_file_path,
        )
        dlg.exec()

    def open_network_stream_dialog(self) -> None:
        dlg = OpenURLDialog(self)
        if dlg.exec():
            if dlg.resolved_stream_url:
                self.load_media_file(dlg.resolved_stream_url)
                if dlg.resolved_title:
                    self.osd.show_notification(f"🌐 Stream: {dlg.resolved_title[:24]}")

    def show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About Siren-VLC",
            "<h3>⚡ Siren-VLC 2.0 (Implicit Neural Media Player)</h3>"
            "<p><b>Lead Architect:</b> Vandan Patel</p>"
            "<p>Replacing discrete pixels with continuous spatio-temporal calculus.</p>"
            "<ul>"
            "<li>Zero-drift hardware master clock A/V sync</li>"
            "<li>Asynchronous double-buffered CUDA streaming</li>"
            "<li>400X infinite continuous analytical zoom</li>"
            "<li>10-bit SMPTE ST.2084 PQ & ACES Filmic HDR</li>"
            "</ul>"
            "<p>Licensed under MIT Open Source.</p>",
        )

    # --- Media Loading Handlers ---

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
                self.osd.show_notification(f"💬 Subtitles: {os.path.basename(filepath)}")
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

                    size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
                    w = self.stream_engine.header.native_width
                    h = self.stream_engine.header.native_height
                    self.osd.show_notification(f"🎬 Loaded .neura 2.0 ({w}x{h} | {size_mb:.2f} MB)")
                else:
                    model, meta = NeuraReader.load(filepath, device="cuda")
                    self.single_engine = PlayerEngine(model, meta, device="cuda")
                    self.stream_engine = None
                    self.meta = meta
                    self.total_duration = float(meta.get("frame_count", 96) / meta.get("fps", 24.0))
                    self.osd.show_notification(f"🎬 Loaded .neura 1.0 ({meta.get('file_size_kb', 0):.1f} KB)")
            else:
                self.load_baseline_video(filepath)
                self.osd.show_notification(f"🎬 Loaded: {os.path.basename(filepath)}")

            self.current_global_time = 0.0
            self.render_frame_at_time(0.0)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Media", f"Could not load media:\n{e}")

    def load_baseline_video(self, filepath: str) -> None:
        try:
            if self.baseline_cap is not None:
                self.baseline_cap.release()
            self.baseline_cap = cv2.VideoCapture(filepath)
            if self.baseline_cap.isOpened():
                self.baseline_fps = float(self.baseline_cap.get(cv2.CAP_PROP_FPS) or 24.0)
                self.baseline_path = filepath
                self.render_frame_at_time(self.current_global_time)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load baseline video:\n{e}")

    # --- UI Toggles & Actions ---

    def toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.menuBar().hide()
            self.bottom_bar.hide()
            self.showFullScreen()
            self.osd.show_notification("⛶ Fullscreen (Press F or Esc to exit)")
        else:
            self.menuBar().show()
            self.bottom_bar.show()
            self.showNormal()
            self.osd.show_notification("⛶ Windowed Mode")

    def toggle_pip_mode(self) -> None:
        """Toggle True Borderless Picture-in-Picture floating mode."""
        self.is_pip_mode = not self.is_pip_mode
        if self.is_pip_mode:
            self.normal_geometry = self.geometry()
            self.menuBar().hide()
            self.bottom_bar.hide()
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
            self.resize(480, 270)
            self.show()
            self.osd.show_notification("📌 Picture-in-Picture Mode: ON")
        else:
            self.setWindowFlags(Qt.WindowType.Window)
            self.menuBar().show()
            self.bottom_bar.show()
            self.setGeometry(self.normal_geometry)
            self.show()
            self.osd.show_notification("📌 Picture-in-Picture Mode: OFF")

    def toggle_night_mode(self, enabled: bool) -> None:
        self.audio_clock.set_night_mode(enabled)
        state_str = "ON (Voice Boost + Limiter)" if enabled else "OFF"
        self.osd.show_notification(f"🌙 Night Mode: {state_str}")

    def set_tone_mapping(self, mode: str, name: str) -> None:
        self.tone_map_mode = mode
        self.osd.show_notification(f"🎨 Color: {name}")
        self.render_frame_at_time(self.current_global_time)

    def set_playback_speed(self, speed: float) -> None:
        self.playback_speed = float(speed)
        self.osd.show_notification(f"⚡ Speed: {self.playback_speed:.2f}x")

    def toggle_playlist(self) -> None:
        self.playlist_dock.setVisible(not self.playlist_dock.isVisible())

    def toggle_equalizer(self) -> None:
        self.equalizer_dlg.show()
        self.equalizer_dlg.raise_()

    def toggle_telemetry(self) -> None:
        is_vis = self.telemetry.isVisible()
        self.telemetry.setVisible(not is_vis)
        state_str = "ON" if not is_vis else "OFF"
        self.osd.show_notification(f"🔬 Examiner Telemetry: {state_str}")
        self.render_frame_at_time(self.current_global_time)

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
        self.render_frame_at_time(self.current_global_time)

    def toggle_split_mode(self) -> None:
        self.is_split_mode = not self.is_split_mode
        if self.is_split_mode:
            self.view_stack.setCurrentIndex(1)
            self.osd.show_notification("🔀 Split Comparison: ON")
        else:
            self.view_stack.setCurrentIndex(0)
            self.osd.show_notification("🔀 Split Comparison: OFF")
        self.render_frame_at_time(self.current_global_time)

    # --- Audio Controls ---

    def toggle_mute(self) -> None:
        is_muted = self.volume_slider.value() == 0
        if is_muted:
            self.volume_slider.setValue(100)
            self.audio_clock.set_muted(False)
            self.btn_mute.setText("🔊")
            self.osd.show_notification("🔊 Unmuted (100%)")
        else:
            self.volume_slider.setValue(0)
            self.audio_clock.set_muted(True)
            self.btn_mute.setText("🔇")
            self.osd.show_notification("🔇 Muted")

    def adjust_volume(self, delta: int) -> None:
        curr = self.volume_slider.value()
        new_val = max(0, min(100, curr + delta))
        self.volume_slider.setValue(new_val)
        self.osd.show_notification(f"🔊 Volume: {new_val}%")

    def on_volume_changed(self, val: int) -> None:
        vol = val / 100.0
        self.audio_clock.set_volume(vol)
        self.btn_mute.setText("🔇" if val == 0 else "🔊")

    # --- Playback Logic ---

    def toggle_play_pause(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.audio_clock.pause()
            self.btn_play_pause.setText("▶ Play")
            self.btn_play_pause.setObjectName("PlayBtn")
            self.osd.show_notification("⏸ Paused")
        else:
            if self.stream_engine is None and self.single_engine is None and self.baseline_cap is None:
                QMessageBox.information(self, "Open Media", "Please open a media file or stream first.")
                return
            self.is_playing = True
            self.audio_clock.play()
            self.playback_timer.start()
            self.btn_play_pause.setText("⏸ Pause")
            self.btn_play_pause.setObjectName("")
            self.osd.show_notification("▶ Playing")
        self.style().polish(self.btn_play_pause)

    def stop_playback(self) -> None:
        self.is_playing = False
        self.playback_timer.stop()
        self.audio_clock.stop()
        self.current_global_time = 0.0
        self.timeline_slider.setValue(0)
        self.btn_play_pause.setText("▶ Play")
        self.btn_play_pause.setObjectName("PlayBtn")
        self.style().polish(self.btn_play_pause)
        self.render_frame_at_time(0.0)
        self.osd.show_notification("⏹ Stopped")

    def seek_relative(self, delta_sec: float) -> None:
        new_time = max(0.0, min(self.total_duration, self.current_global_time + delta_sec))
        self.current_global_time = new_time
        self.audio_clock.seek(new_time)
        self.render_frame_at_time(new_time)
        sign = "+" if delta_sec > 0 else ""
        self.osd.show_notification(f"⏩ Seek {sign}{delta_sec:.1f}s")

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
                lod_fast=self.is_playing,
            )
            raw_rgb = res.rgb_numpy
            pre_str = " (Prefetched)" if res.is_prefetched else ""
            hud_str = (
                f"⚡ SIREN-ZIP 2.0 [{res.chunk_idx+1:02d}/{res.total_chunks:02d}{pre_str}] | "
                f"{self.last_fps:.1f} FPS | {res.eval_time_ms:.1f}ms | A/V: 0.0ms"
            )

            if self.telemetry.isVisible():
                sz_mb = os.path.getsize(self.active_file_path) / (1024.0 * 1024.0) if self.active_file_path else 3.4
                self.telemetry.update_telemetry(
                    chunk_idx=res.chunk_idx,
                    total_chunks=res.total_chunks,
                    model_size_mb=sz_mb,
                    master_time_sec=t_sec,
                    local_time_norm=res.t_local,
                    zoom_factor=self.current_zoom,
                    culling_saved_pct=res.culling_saved_pct,
                    paging_ms=res.paging_time_ms,
                    eval_ms=res.eval_time_ms,
                    omega_xy=float(self.meta.get("omega_xy", 30.0)),
                    omega_t=float(self.meta.get("omega_t", 10.0)),
                )
        elif self.single_engine is not None:
            alpha = min(1.0, max(0.0, t_sec / max(0.001, self.total_duration)))
            t_norm = -1.0 + 2.0 * alpha
            res_single: RenderResult = self.single_engine.render_viewport(
                t_val=t_norm,
                viewport=self.current_viewport,
                render_width=w,
                render_height=h,
                lod_fast=self.is_playing,
            )
            raw_rgb = res_single.rgb_numpy
            hud_str = f"⚡ SIREN-ZIP 1.0 | {self.last_fps:.1f} FPS | {res_single.compute_time_ms:.1f}ms"
        elif self.baseline_cap is not None and self.baseline_cap.isOpened():
            pos_msec = t_sec * 1000.0
            self.baseline_cap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
            ret, bgr = self.baseline_cap.read()
            if ret and bgr is not None:
                raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                return
            hud_str = f"🎬 Baseline Video | {self.last_fps:.1f} FPS"
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
            if self.baseline_cap is not None and self.baseline_cap.isOpened():
                pos_msec = t_sec * 1000.0
                self.baseline_cap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
                ret, bgr = self.baseline_cap.read()
                if ret and bgr is not None:
                    discrete_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            self.split_view.update_buffers(
                siren_rgb=final_rgb,
                discrete_full_frame=discrete_frame,
                viewport=self.current_viewport,
            )
        else:
            self.full_canvas.set_frame_buffer(final_rgb)

    def take_high_res_screenshot(self) -> None:
        """Render continuous mathematical coordinate field at full 4K UHD (3840x2160) and save to runs/."""
        if self.stream_engine is None and self.single_engine is None and self.baseline_cap is None:
            return

        os.makedirs("runs", exist_ok=True)
        shot_path = os.path.join("runs", f"screenshot_{int(time.time())}.png")
        shot_w, shot_h = 3840, 2160
        self.osd.show_notification("📸 Generating 4K Continuous Shot...")
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
        elif self.single_engine is not None:
            alpha = min(1.0, max(0.0, self.current_global_time / max(0.001, self.total_duration)))
            res = self.single_engine.render_viewport(
                t_val=-1.0 + 2.0 * alpha,
                viewport=self.current_viewport,
                render_width=shot_w,
                render_height=shot_h,
            )
            raw = res.rgb_numpy
        else:
            pos_msec = self.current_global_time * 1000.0
            self.baseline_cap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
            ret, bgr = self.baseline_cap.read()
            raw = cv2.cvtColor(cv2.resize(bgr, (shot_w, shot_h)), cv2.COLOR_BGR2RGB) if ret else np.zeros((shot_h, shot_w, 3), dtype=np.uint8)

        processed = self.video_fx.apply(raw)
        bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(shot_path, bgr)
        self.osd.show_notification(f"📸 4K Shot Saved: {os.path.basename(shot_path)}", duration_ms=2500)

    def closeEvent(self, event) -> None:
        self.playback_timer.stop()
        if self.baseline_cap is not None:
            self.baseline_cap.release()
            self.baseline_cap = None
        self.audio_clock.cleanup()
        if self.stream_engine is not None:
            self.stream_engine.close()
        super().closeEvent(event)


def launch_player_app(
    neura_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = SirenPlayerWindow(neura_path=neura_path, baseline_path=baseline_path)
    window.show()
    app.exec()
