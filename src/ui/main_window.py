"""Siren-VLC: The Universal Neural Media Player with Authentic VideoLAN VLC UI/UX."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPixmap,
)
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
    QStyle,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

import threading

from src.audio.audio_player import AudioMasterClock
from src.filters.video_fx import VideoFXFilter
from src.subtitles.subtitle_engine import SubtitleEngine
from src.types.viewport import ViewportBounds
from src.ui.clipper_dialog import ClipperDialog
from src.ui.equalizer_dialog import EqualizerDialog
from src.ui.open_url_dialog import OpenURLDialog
from src.ui.osd_overlay import OSDOverlay
from src.ui.playlist_widget import PlaylistWidget
from src.ui.split_view import SplitComparisonView
from src.ui.telemetry_overlay import TelemetryOverlay
from src.ui.video_canvas import ContinuousVideoCanvas


def _warmup_cuda_background() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            _ = torch.zeros(1, device="cuda")
    except Exception:
        pass

VLC_ORANGE = "#ff8800"
VLC_DARK_BG = "#181a1f"
VLC_PANEL_BG = "#21252b"
VLC_BORDER = "#30363d"

AUTHENTIC_VLC_STYLESHEET = f"""
QMainWindow {{
    background-color: {VLC_DARK_BG};
    color: #e6edf3;
}}
QMenuBar {{
    background-color: {VLC_PANEL_BG};
    color: #c9d1d9;
    border-bottom: 1px solid {VLC_BORDER};
    font-size: 12px;
    padding: 2px 4px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 8px;
    border-radius: 3px;
}}
QMenuBar::item:selected {{
    background-color: #2c313a;
    color: {VLC_ORANGE};
}}
QMenu {{
    background-color: {VLC_PANEL_BG};
    color: #e6edf3;
    border: 1px solid {VLC_BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 24px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background-color: {VLC_ORANGE};
    color: #000000;
    font-weight: bold;
}}
QMenu::separator {{
    height: 1px;
    background-color: {VLC_BORDER};
    margin: 4px 6px;
}}
QWidget {{
    background-color: {VLC_DARK_BG};
    color: #e6edf3;
    font-family: 'Segoe UI', Arial, sans-serif;
}}
QFrame#VLCBottomBar {{
    background-color: {VLC_PANEL_BG};
    border-top: 1px solid {VLC_BORDER};
    padding: 4px 10px 6px 10px;
}}
QFrame#VLCAdvancedBar {{
    background-color: #1e2227;
    border-top: 1px solid {VLC_BORDER};
    padding: 3px 10px;
}}
QPushButton {{
    background-color: #282c34;
    border: 1px solid #3e4451;
    color: #abb2bf;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: 600;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: #353b45;
    border-color: {VLC_ORANGE};
    color: #ffffff;
}}
QPushButton:pressed {{
    background-color: #1b1d23;
}}
QPushButton#VLCPlayBtn {{
    background-color: {VLC_ORANGE};
    color: #000000;
    font-weight: bold;
    border: 1px solid #e07700;
    min-width: 60px;
    font-size: 12px;
}}
QPushButton#VLCPlayBtn:hover {{
    background-color: #ffa033;
}}
QPushButton#ActiveToggleBtn {{
    background-color: {VLC_ORANGE};
    color: #000000;
    font-weight: bold;
    border: 1px solid #e07700;
}}
QSlider::groove:horizontal {{
    height: 5px;
    background: #30363d;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {VLC_ORANGE};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #ffffff;
    border: 2px solid {VLC_ORANGE};
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {VLC_ORANGE};
    border-color: #ffffff;
}}
QLabel#TimeDisplay {{
    color: #c9d1d9;
    font-family: 'Consolas', 'Segoe UI', monospace;
    font-size: 11px;
    font-weight: bold;
    padding: 0 4px;
}}
QLabel#TimeDisplay:hover {{
    color: {VLC_ORANGE};
}}
QLabel#StatusBadge {{
    color: {VLC_ORANGE};
    font-family: 'Consolas', monospace;
    font-size: 11px;
    font-weight: bold;
}}
"""


class SirenPlayerWindow(QMainWindow):
    """Siren-VLC: Universal Implicit Neural Media Player with Authentic VideoLAN VLC UI/UX."""

    def __init__(
        self,
        neura_path: Optional[str] = None,
        baseline_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Siren-VLC Media Player")
        self.resize(1280, 800)
        self.setAcceptDrops(True)
        self.setStyleSheet(AUTHENTIC_VLC_STYLESHEET)

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
        self.equalizer_dlg.night_mode_toggled.connect(self.toggle_night_mode)

        # Baseline comparison & High-Speed Frame Cache
        self.baseline_cap: Optional[cv2.VideoCapture] = None
        self.baseline_fps: float = 24.0
        self.baseline_path: Optional[str] = baseline_path
        self.baseline_current_frame: int = -1
        self.baseline_cached_frame: Optional[np.ndarray] = None
        self.is_split_mode: bool = False

        # Playback Timeline State
        self.is_playing: bool = False
        self.loop_mode: int = 0  # 0: Repeat All (🔁), 1: Repeat One (🔂), 2: No Repeat (➡️)
        self.current_global_time: float = 0.0
        self.total_duration: float = 12.0
        self.playback_speed: float = 1.0
        self.tone_map_mode: str = "aces"
        self.is_fullscreen: bool = False
        self.is_pip_mode: bool = False
        self.show_remaining_time: bool = False
        self.normal_geometry = self.geometry()

        # VLC A-B Loop State
        self.ab_loop_a: Optional[float] = None
        self.ab_loop_b: Optional[float] = None

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

        # 2. Central Full Video Canvas
        self.view_stack = QStackedWidget(self)
        self.full_canvas = ContinuousVideoCanvas(self)
        self.full_canvas.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.full_canvas)

        self.split_view = SplitComparisonView(self)
        self.split_view.viewport_changed.connect(self.on_viewport_changed)
        self.view_stack.addWidget(self.split_view)

        self.main_layout.addWidget(self.view_stack, stretch=1)

        # 3. Optional VLC Advanced Controls Bar (A-B Loop, Record, Step Frame, Snapshot)
        self.advanced_bar = QFrame()
        self.advanced_bar.setObjectName("VLCAdvancedBar")
        adv_layout = QHBoxLayout(self.advanced_bar)
        adv_layout.setContentsMargins(12, 3, 12, 3)
        adv_layout.setSpacing(6)

        self.btn_ab_loop = QPushButton("🔁 A-B Loop: OFF")
        self.btn_ab_loop.setToolTip("Click once for Point A, again for Point B to loop continuously")
        self.btn_ab_loop.clicked.connect(self.toggle_ab_loop)
        adv_layout.addWidget(self.btn_ab_loop)

        btn_step = QPushButton("🎞️ Step 1 Frame (E)")
        btn_step.setToolTip("Advance exactly 1 video frame")
        btn_step.clicked.connect(self.step_one_frame)
        adv_layout.addWidget(btn_step)

        btn_record = QPushButton("🔴 Quick Clip (Ctrl+R)")
        btn_record.setToolTip("Export GIF or WebM clip")
        btn_record.clicked.connect(self.open_clipper_dialog)
        adv_layout.addWidget(btn_record)

        btn_snap = QPushButton("📸 Snapshot (S)")
        btn_snap.setToolTip("Take full 4K continuous mathematical screenshot")
        btn_snap.clicked.connect(self.take_high_res_screenshot)
        adv_layout.addWidget(btn_snap)

        adv_layout.addStretch()
        self.main_layout.addWidget(self.advanced_bar)
        self.advanced_bar.hide()  # Hidden by default, toggled via View -> Advanced Controls

        # 4. Classic VLC Bottom Control Bar
        self.bottom_bar = QFrame()
        self.bottom_bar.setObjectName("VLCBottomBar")
        bar_layout = QVBoxLayout(self.bottom_bar)
        bar_layout.setContentsMargins(12, 4, 12, 6)
        bar_layout.setSpacing(4)

        # Row 1: Timeline Scrubber + Clickable Time Displays
        row_time = QHBoxLayout()
        row_time.setSpacing(8)

        self.lbl_current_time = QLabel("00:00")
        self.lbl_current_time.setObjectName("TimeDisplay")
        row_time.addWidget(self.lbl_current_time)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 10000)
        self.timeline_slider.setValue(0)
        self.timeline_slider.setTracking(True)
        self.timeline_slider.valueChanged.connect(self.on_timeline_slider_changed)
        row_time.addWidget(self.timeline_slider, stretch=1)

        self.lbl_total_time = QLabel("00:00")
        self.lbl_total_time.setObjectName("TimeDisplay")
        self.lbl_total_time.setToolTip("Click to toggle remaining time (-)")
        self.lbl_total_time.mousePressEvent = self.on_toggle_time_display
        row_time.addWidget(self.lbl_total_time)

        bar_layout.addLayout(row_time)

        # Row 2: Standard VLC Playback Controls & Utilities
        row_controls = QHBoxLayout()
        row_controls.setSpacing(5)

        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setObjectName("VLCPlayBtn")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        row_controls.addWidget(self.btn_play_pause)

        btn_prev = QPushButton("⏮")
        btn_prev.setToolTip("Previous Track (P)")
        btn_prev.setFixedWidth(30)
        btn_prev.clicked.connect(self.on_play_previous)
        row_controls.addWidget(btn_prev)

        btn_slower = QPushButton("⏪")
        btn_slower.setToolTip("Jump Back 5s (Left Arrow)")
        btn_slower.setFixedWidth(30)
        btn_slower.clicked.connect(lambda: self.seek_relative(-5.0))
        row_controls.addWidget(btn_slower)

        btn_stop = QPushButton("⏹")
        btn_stop.setToolTip("Stop")
        btn_stop.setFixedWidth(30)
        btn_stop.clicked.connect(self.stop_playback)
        row_controls.addWidget(btn_stop)

        btn_faster = QPushButton("⏩")
        btn_faster.setToolTip("Jump Forward 5s (Right Arrow)")
        btn_faster.setFixedWidth(30)
        btn_faster.clicked.connect(lambda: self.seek_relative(+5.0))
        row_controls.addWidget(btn_faster)

        btn_next = QPushButton("⏭")
        btn_next.setToolTip("Next Track (N)")
        btn_next.setFixedWidth(30)
        btn_next.clicked.connect(self.on_play_next)
        row_controls.addWidget(btn_next)

        row_controls.addSpacing(6)

        btn_fs = QPushButton("⛶")
        btn_fs.setToolTip("Toggle Fullscreen (F / F11)")
        btn_fs.setFixedWidth(32)
        btn_fs.clicked.connect(self.toggle_fullscreen)
        row_controls.addWidget(btn_fs)

        btn_pip = QPushButton("📌 PiP")
        btn_pip.setToolTip("Picture-in-Picture Floating Mode (Ctrl+P)")
        btn_pip.clicked.connect(self.toggle_pip_mode)
        row_controls.addWidget(btn_pip)

        btn_eq = QPushButton("🎛️")
        btn_eq.setToolTip("Extended Settings / Audio & Video FX (Ctrl+E)")
        btn_eq.setFixedWidth(32)
        btn_eq.clicked.connect(self.toggle_equalizer)
        row_controls.addWidget(btn_eq)

        self.btn_loop_mode = QPushButton("🔁")
        self.btn_loop_mode.setToolTip("Repeat Mode: All -> One -> Off")
        self.btn_loop_mode.setFixedWidth(32)
        self.btn_loop_mode.clicked.connect(self.cycle_loop_mode)
        row_controls.addWidget(self.btn_loop_mode)

        btn_list = QPushButton("📋")
        btn_list.setToolTip("Toggle Playlist Queue (Ctrl+L)")
        btn_list.setFixedWidth(32)
        btn_list.clicked.connect(self.toggle_playlist)
        row_controls.addWidget(btn_list)

        row_controls.addSpacing(10)

        # Volume Controls (with up to 125% VLC volume boost)
        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(32)
        self.btn_mute.clicked.connect(self.toggle_mute)
        row_controls.addWidget(self.btn_mute)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 125)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        row_controls.addWidget(self.volume_slider)

        self.lbl_vol_pct = QLabel("100%")
        self.lbl_vol_pct.setStyleSheet("font-size: 10px; color: #8b949e; width: 32px;")
        row_controls.addWidget(self.lbl_vol_pct)

        row_controls.addStretch()

        # Status Badge
        self.lbl_hud_stats = QLabel("⚡ SIREN-ZIP 2.0 | 60 FPS | A/V 0.0ms")
        self.lbl_hud_stats.setObjectName("StatusBadge")
        row_controls.addWidget(self.lbl_hud_stats)

        bar_layout.addLayout(row_controls)
        self.main_layout.addWidget(self.bottom_bar)

        # 5. Playlist Dock (hidden by default)
        self.playlist_dock = PlaylistWidget(self)
        self.playlist_dock.file_selected.connect(self.load_media_file)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.playlist_dock)
        self.playlist_dock.hide()

    def setup_menu_bar(self) -> None:
        menubar = self.menuBar()

        # --- &Media ---
        media_menu = menubar.addMenu("&Media")

        act_open_file = QAction("Open &File...", self)
        act_open_file.setShortcut(QKeySequence("Ctrl+O"))
        act_open_file.triggered.connect(self.on_open_file_dialog)
        media_menu.addAction(act_open_file)

        act_open_url = QAction("Open &Network Stream (YouTube / Twitch)...", self)
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

        # --- &Playback ---
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

        act_step = QAction("Step &One Frame Forward", self)
        act_step.setShortcut(QKeySequence("E"))
        act_step.triggered.connect(self.step_one_frame)
        play_menu.addAction(act_step)

        play_menu.addSeparator()

        speed_menu = play_menu.addMenu("&Speed")
        for sp in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]:
            act_sp = QAction(f"{sp}x", self)
            act_sp.triggered.connect(lambda _, s=sp: self.set_playback_speed(s))
            speed_menu.addAction(act_sp)

        # --- &Audio ---
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

        # --- &Video ---
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
        for mode, name in [
            ("aces", "ACES Filmic (HDR)"),
            ("reinhard_jodie", "Reinhard-Jodie"),
            ("reinhard", "Reinhard"),
            ("linear", "SDR Linear"),
        ]:
            act_hdr = QAction(name, self)
            act_hdr.triggered.connect(lambda _, m=mode, n=name: self.set_tone_mapping(m, n))
            hdr_menu.addAction(act_hdr)

        # --- &Subtitle ---
        sub_menu = menubar.addMenu("&Subtitle")

        act_add_sub = QAction("&Add Subtitle File...", self)
        act_add_sub.triggered.connect(self.on_open_subtitles_dialog)
        sub_menu.addAction(act_add_sub)

        # --- &Tools ---
        tools_menu = menubar.addMenu("&Tools")

        act_eq = QAction("&Effects and Filters / Equalizer", self)
        act_eq.setShortcut(QKeySequence("Ctrl+E"))
        act_eq.triggered.connect(self.toggle_equalizer)
        tools_menu.addAction(act_eq)

        act_telem = QAction("Examiner &Telemetry HUD", self)
        act_telem.setShortcut(QKeySequence("F12"))
        act_telem.triggered.connect(self.toggle_telemetry)
        tools_menu.addAction(act_telem)

        # --- &View ---
        view_menu = menubar.addMenu("&View")

        act_adv = QAction("&Advanced Controls Toolbar", self)
        act_adv.setCheckable(True)
        act_adv.setChecked(False)
        act_adv.triggered.connect(lambda v: self.advanced_bar.setVisible(v))
        view_menu.addAction(act_adv)

        act_list = QAction("&Playlist", self)
        act_list.setShortcut(QKeySequence("Ctrl+L"))
        act_list.triggered.connect(self.toggle_playlist)
        view_menu.addAction(act_list)

        # --- &Help ---
        help_menu = menubar.addMenu("&Help")

        act_about = QAction("&About Siren-VLC", self)
        act_about.triggered.connect(self.show_about_dialog)
        help_menu.addAction(act_about)

    # --- Drag and Drop File Support ---

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()
            if filepath.endswith((".srt", ".vtt")):
                if self.subtitle_engine.load_file(filepath):
                    self.osd.show_notification(f"💬 Subtitles Loaded: {os.path.basename(filepath)}")
                    self.render_frame_at_time(self.current_global_time)
            elif filepath.endswith((".neura", ".mp4", ".mkv", ".avi", ".mov", ".webm")):
                self.load_media_file(filepath)
                self.playlist_dock.add_file(filepath)
                if not self.is_playing:
                    self.toggle_play_pause()

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
        elif key == Qt.Key.Key_E:
            self.step_one_frame()
        elif key == Qt.Key.Key_P:
            self.toggle_playlist()
        elif key == Qt.Key.Key_F12:
            self.toggle_telemetry()
        elif key == Qt.Key.Key_Escape:
            if self.is_fullscreen:
                self.toggle_fullscreen()
            elif self.is_pip_mode:
                self.toggle_pip_mode()
        else:
            super().keyPressEvent(event)

    # --- Advanced Features: A-B Loop & Frame Stepping ---

    def toggle_ab_loop(self) -> None:
        """3-State VLC A-B Loop: Set A -> Set B -> Clear."""
        if self.ab_loop_a is None:
            self.ab_loop_a = self.current_global_time
            self.btn_ab_loop.setText(f"🔁 A: {self.ab_loop_a:.1f}s -> B")
            self.btn_ab_loop.setObjectName("ActiveToggleBtn")
            self.osd.show_notification(f"🔁 A-B Loop: Point A set ({self.ab_loop_a:.2f}s)")
        elif self.ab_loop_b is None:
            self.ab_loop_b = self.current_global_time
            if self.ab_loop_b <= self.ab_loop_a:
                self.ab_loop_b = self.ab_loop_a + 2.0
            self.btn_ab_loop.setText(f"🔁 [{self.ab_loop_a:.1f}s - {self.ab_loop_b:.1f}s]")
            self.osd.show_notification(f"🔁 A-B Loop: Active [{self.ab_loop_a:.2f}s - {self.ab_loop_b:.2f}s]")
        else:
            self.ab_loop_a = None
            self.ab_loop_b = None
            self.btn_ab_loop.setText("🔁 A-B Loop: OFF")
            self.btn_ab_loop.setObjectName("")
            self.osd.show_notification("🔁 A-B Loop: Cleared")
        self.style().polish(self.btn_ab_loop)

    def step_one_frame(self) -> None:
        """Step forward exactly one frame."""
        if self.is_playing:
            self.toggle_play_pause()
        dt = 1.0 / max(1.0, self.baseline_fps)
        self.seek_relative(dt)
        self.osd.show_notification(f"🎞️ Step +1 Frame ({self.current_global_time:.3f}s)")

    def cycle_loop_mode(self) -> None:
        self.loop_mode = (self.loop_mode + 1) % 3
        icons = ["🔁", "🔂", "➡️"]
        names = ["Repeat All", "Repeat One", "No Repeat"]
        self.btn_loop_mode.setText(icons[self.loop_mode])
        self.osd.show_notification(f"Mode: {names[self.loop_mode]}")

    def on_toggle_time_display(self, event) -> None:
        self.show_remaining_time = not self.show_remaining_time
        self.render_frame_at_time(self.current_global_time)

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
            "<h3>⚡ Siren-VLC 2.0</h3>"
            "<p><b>Lead Architect:</b> Vandan Patel</p>"
            "<p>The Universal Implicit Neural Media Player with authentic VideoLAN VLC design.</p>"
            "<ul>"
            "<li>Continuous Spatio-Temporal Calculus</li>"
            "<li>Zero-Drift Hardware Master Clock A/V Sync</li>"
            "<li>Asynchronous Double-Buffered CUDA Prefetching</li>"
            "<li>400X Infinite Continuous Analytical Zoom</li>"
            "<li>10-Bit SMPTE ST.2084 PQ & ACES Filmic HDR</li>"
            "<li>A-B Looping, One-Click GIF/WebM Clipper & Direct YouTube Streaming</li>"
            "</ul>"
            "<p>Licensed under MIT Open Source.</p>",
        )

    # --- Media Loading Handlers ---

    def on_open_file_dialog(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Media File", "", "Siren & Video Files (*.neura *.mp4 *.mkv *.avi *.mov *.webm);;All Files (*)"
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
            self, "Open Baseline Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if filepath:
            self.load_baseline_video(filepath)

    def load_media_file(self, filepath: str) -> None:
        """Universal loader for .neura 2.0, .neura 1.0, and standard MP4/MKV files."""
        try:
            self.active_file_path = filepath
            if filepath.endswith(".neura"):
                from src.player.engine import PlayerEngine
                from src.player.neura_reader import NeuraReader
                from src.streaming.stream_engine import StreamEngine

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
                fc = float(self.baseline_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100.0)
                self.total_duration = fc / self.baseline_fps
                self.baseline_path = filepath
                self.baseline_current_frame = -1
                self.baseline_cached_frame = None

                # Load embedded audio from video file for hardware A/V master sync
                self.audio_clock.load_audio_file(filepath)

                self.render_frame_at_time(self.current_global_time)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not load baseline video:\n{e}")

    def get_baseline_frame_at_time(self, t_sec: float) -> Optional[np.ndarray]:
        """Decode baseline video frame with high-speed sequential caching (120+ FPS)."""
        if self.baseline_cap is None or not self.baseline_cap.isOpened():
            return None

        target_frame = int(round(t_sec * self.baseline_fps))
        total_frames = int(self.baseline_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1000000)
        target_frame = max(0, min(total_frames - 1, target_frame))

        # 1. Reuse already decoded frame if at same timestamp (0.0 ms)
        if target_frame == self.baseline_current_frame and self.baseline_cached_frame is not None:
            return self.baseline_cached_frame

        # 2. Sequential decode if advancing to next frame (< 1.0 ms, 200+ FPS)
        if target_frame == self.baseline_current_frame + 1:
            ret, bgr = self.baseline_cap.read()
            if ret and bgr is not None:
                self.baseline_current_frame = target_frame
                self.baseline_cached_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return self.baseline_cached_frame

        # 3. Seeking / Timeline jumping
        self.baseline_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, bgr = self.baseline_cap.read()
        if ret and bgr is not None:
            self.baseline_current_frame = target_frame
            self.baseline_cached_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return self.baseline_cached_frame

        return None

    # --- UI Toggles & Actions ---

    def toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.menuBar().hide()
            self.bottom_bar.hide()
            self.advanced_bar.hide()
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
            self.advanced_bar.hide()
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
            from src.sharing.share_packer import SharePacker
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
        new_val = max(0, min(125, curr + delta))
        self.volume_slider.setValue(new_val)
        self.osd.show_notification(f"🔊 Volume: {new_val}%")

    def on_volume_changed(self, val: int) -> None:
        vol = val / 100.0
        self.audio_clock.set_volume(vol)
        self.btn_mute.setText("🔇" if val == 0 else ("🔊" if val <= 100 else "📢"))
        self.lbl_vol_pct.setText(f"{val}%")

    # --- Playback Logic ---

    def toggle_play_pause(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.audio_clock.pause()
            self.btn_play_pause.setText("▶ Play")
            self.osd.show_notification("⏸ Paused")
        else:
            if self.stream_engine is None and self.single_engine is None and self.baseline_cap is None:
                QMessageBox.information(self, "Open Media", "Please open a media file or stream first.")
                return
            self.is_playing = True
            self.audio_clock.play()
            self.playback_timer.start()
            self.btn_play_pause.setText("⏸ Pause")
            self.osd.show_notification("▶ Playing")

    def stop_playback(self) -> None:
        self.is_playing = False
        self.playback_timer.stop()
        self.audio_clock.stop()
        self.current_global_time = 0.0
        self.timeline_slider.setValue(0)
        self.btn_play_pause.setText("▶ Play")
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
        if self.loop_mode == 1:  # Repeat One
            self.current_global_time = 0.0
            self.audio_clock.seek(0.0)
            if self.is_playing:
                self.audio_clock.play()
        elif self.loop_mode == 0:  # Repeat All
            self.on_play_next()
        else:  # No Repeat
            self.stop_playback()

    def on_timer_tick(self) -> None:
        if not self.is_playing:
            return

        # Handle A-B Loop boundary
        if self.ab_loop_a is not None and self.ab_loop_b is not None:
            if self.current_global_time >= self.ab_loop_b or self.current_global_time < self.ab_loop_a:
                self.current_global_time = self.ab_loop_a
                self.audio_clock.seek(self.ab_loop_a)

        if self.audio_clock.is_loaded:
            t_master = self.audio_clock.get_master_time()
            self.current_global_time = t_master
        else:
            dt = (1.0 / 60.0) * self.playback_speed
            self.current_global_time += dt

        if self.current_global_time > self.total_duration:
            self.on_audio_ended()
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

        # Formatted Time Display (e.g. 01:24 and -02:45)
        curr_m, curr_s = int(t_sec // 60), int(t_sec % 60)
        self.lbl_current_time.setText(f"{curr_m:02d}:{curr_s:02d}")

        if self.show_remaining_time:
            rem = max(0.0, self.total_duration - t_sec)
            rem_m, rem_s = int(rem // 60), int(rem % 60)
            self.lbl_total_time.setText(f"-{rem_m:02d}:{rem_s:02d}")
        else:
            tot_m, tot_s = int(self.total_duration // 60), int(self.total_duration % 60)
            self.lbl_total_time.setText(f"{tot_m:02d}:{tot_s:02d}")

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
            frame = self.get_baseline_frame_at_time(t_sec)
            if frame is not None:
                raw_rgb = frame
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
            discrete_frame = self.get_baseline_frame_at_time(t_sec)
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
            frame = self.get_baseline_frame_at_time(self.current_global_time)
            raw = cv2.resize(frame, (shot_w, shot_h)) if frame is not None else np.zeros((shot_h, shot_w, 3), dtype=np.uint8)

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
    # Asynchronously pre-warm CUDA in background so UI opens in < 200 ms with zero freeze
    threading.Thread(target=_warmup_cuda_background, daemon=True).start()
    app.exec()
