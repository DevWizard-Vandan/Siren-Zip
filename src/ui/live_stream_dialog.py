"""Siren-Cast Live Stream & Neural Network Broadcast Dialog."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.live.broadcast_server import NeuralBroadcastServer
from src.tools.web_streamer import WebStreamer


class LiveStreamDialog(QDialog):
    """Multi-tab modal dialog for Web Video Streaming, Siren-Cast Live Client, and Live Broadcaster."""

    # Signal emitted when client connects: (stream_url: str, mode: str)
    stream_selected = Signal(str, str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        broadcast_server_ref: Optional[NeuralBroadcastServer] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("🌐 Open Network Stream & Siren-Cast Live")
        self.setFixedSize(540, 360)

        self.resolved_stream_url: Optional[str] = None
        self.resolved_title: str = ""
        self.stream_mode: str = "url"  # 'url', 'siren_cast', 'broadcast'
        self.broadcast_server = broadcast_server_ref

        self.setup_ui()

        # Telemetry Timer for Broadcast Server Tab
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.setInterval(500)
        self.telemetry_timer.timeout.connect(self._update_broadcast_telemetry)
        self.telemetry_timer.start()

    def setup_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #161b22;
                color: #e6edf3;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                background-color: #0d1117;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #21262d;
                color: #8b949e;
                border: 1px solid #30363d;
                padding: 6px 14px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 500;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background-color: #0d1117;
                color: #58a6ff;
                border-bottom-color: #0d1117;
                font-weight: bold;
            }
            QLabel {
                color: #e6edf3;
                font-size: 11px;
            }
            QLineEdit, QSpinBox, QComboBox {
                background-color: #161b22;
                border: 1px solid #30363d;
                color: #e6edf3;
                border-radius: 5px;
                padding: 6px 8px;
                font-size: 11px;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border-color: #58a6ff;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                color: #c9d1d9;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #30363d;
                color: #ffffff;
            }
            QPushButton#PrimaryBtn {
                background-color: #238636;
                color: #ffffff;
                border-color: #2ea043;
            }
            QPushButton#PrimaryBtn:hover {
                background-color: #2ea043;
            }
            QPushButton#LiveBtn {
                background-color: #da3633;
                color: #ffffff;
                border-color: #f85149;
            }
            QPushButton#LiveBtn:hover {
                background-color: #f85149;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #58a6ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding-left: 8px;
            }
            """
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        self.tabs = QTabWidget()

        # Tab 1: Web URL (YouTube / Twitch)
        self.tab_url = QWidget()
        self._setup_url_tab()
        self.tabs.addTab(self.tab_url, "🌐 Web URL (YouTube / Twitch)")

        # Tab 2: Siren-Cast Live Network
        self.tab_siren_cast = QWidget()
        self._setup_siren_cast_tab()
        self.tabs.addTab(self.tab_siren_cast, "🔴 Siren-Cast Live Stream")

        # Tab 3: Broadcast Camera / Video Server
        self.tab_broadcast = QWidget()
        self._setup_broadcast_tab()
        self.tabs.addTab(self.tab_broadcast, "📡 Broadcast Server")

        main_layout.addWidget(self.tabs)

        # Bottom Buttons
        row_bottom = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        row_bottom.addStretch()
        row_bottom.addWidget(btn_close)
        main_layout.addLayout(row_bottom)

    # --- Tab 1: Web URL ---
    def _setup_url_tab(self) -> None:
        layout = QVBoxLayout(self.tab_url)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        lbl = QLabel("Enter Web Stream URL (YouTube, Twitch, or Direct .mp4/.m3u8):")
        layout.addWidget(lbl)

        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText("https://www.youtube.com/watch?v=...")
        layout.addWidget(self.txt_url)

        layout.addStretch()

        row_actions = QHBoxLayout()
        self.btn_play_url = QPushButton("▶ Play Web Stream")
        self.btn_play_url.setObjectName("PrimaryBtn")
        self.btn_play_url.clicked.connect(self._on_play_url)

        row_actions.addStretch()
        row_actions.addWidget(self.btn_play_url)
        layout.addLayout(row_actions)

    def _on_play_url(self) -> None:
        url = self.txt_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid network URL.")
            return

        self.btn_play_url.setText("⏳ Resolving Stream...")
        self.btn_play_url.setEnabled(False)

        res = WebStreamer.resolve_url(url)
        self.btn_play_url.setText("▶ Play Web Stream")
        self.btn_play_url.setEnabled(True)

        if not res.get("success"):
            QMessageBox.critical(self, "Stream Error", f"Failed to resolve URL:\n{res.get('error')}")
            return

        self.resolved_stream_url = res.get("stream_url")
        self.resolved_title = res.get("title", "Web Stream")
        self.stream_mode = "url"
        self.accept()

    # --- Tab 2: Siren-Cast Live Client ---
    def _setup_siren_cast_tab(self) -> None:
        layout = QVBoxLayout(self.tab_siren_cast)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        lbl = QLabel("Connect to a Live Neural Mathematical Stream (WebSockets):")
        layout.addWidget(lbl)

        self.txt_ws_url = QLineEdit("ws://localhost:8765")
        self.txt_ws_url.setPlaceholderText("ws://<host>:<port>")
        layout.addWidget(self.txt_ws_url)

        row_opts = QHBoxLayout()
        self.chk_auto_reconnect = QCheckBox("Auto-Reconnect on Packet Loss")
        self.chk_auto_reconnect.setChecked(True)
        self.chk_gpu_eval = QCheckBox("GPU Continuous Evaluation (60 FPS)")
        self.chk_gpu_eval.setChecked(True)
        row_opts.addWidget(self.chk_auto_reconnect)
        row_opts.addWidget(self.chk_gpu_eval)
        layout.addLayout(row_opts)

        info_box = QLabel(
            "⚡ <b>Neural WebSockets:</b> Receives differential weight tensors (Δθ ~30-60 KB/s)<br>"
            "and evaluates the continuous mathematical field live on your GPU with 0.0ms lip-sync drift."
        )
        info_box.setStyleSheet("color: #8b949e; font-size: 10px; padding: 4px;")
        layout.addWidget(info_box)

        layout.addStretch()

        row_actions = QHBoxLayout()
        self.btn_connect_live = QPushButton("🔴 Connect & Stream Live")
        self.btn_connect_live.setObjectName("LiveBtn")
        self.btn_connect_live.clicked.connect(self._on_connect_siren_cast)

        row_actions.addStretch()
        row_actions.addWidget(self.btn_connect_live)
        layout.addLayout(row_actions)

    def _on_connect_siren_cast(self) -> None:
        ws_url = self.txt_ws_url.text().strip()
        if not ws_url.startswith("ws://") and not ws_url.startswith("wss://"):
            QMessageBox.warning(self, "Invalid WebSocket URL", "URL must start with ws:// or wss://")
            return

        self.resolved_stream_url = ws_url
        self.resolved_title = f"🔴 Siren-Cast Live ({ws_url})"
        self.stream_mode = "siren_cast"
        self.accept()

    # --- Tab 3: Broadcast Server ---
    def _setup_broadcast_tab(self) -> None:
        layout = QVBoxLayout(self.tab_broadcast)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Source Selection
        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Broadcast Source:"))
        self.cmb_source_type = QComboBox()
        self.cmb_source_type.addItems(["Webcam (Device 0)", "Webcam (Device 1)", "Video File (.mp4/.mkv)"])
        self.cmb_source_type.currentIndexChanged.connect(self._on_source_type_changed)
        row_src.addWidget(self.cmb_source_type, 1)

        self.btn_browse_video = QPushButton("Browse...")
        self.btn_browse_video.setVisible(False)
        self.btn_browse_video.clicked.connect(self._on_browse_video)
        row_src.addWidget(self.btn_browse_video)
        layout.addLayout(row_src)

        self.lbl_selected_file = QLabel("")
        self.lbl_selected_file.setStyleSheet("color: #58a6ff; font-size: 10px;")
        self.lbl_selected_file.setVisible(False)
        layout.addWidget(self.lbl_selected_file)

        # Port & Duration Configuration
        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("Port:"))
        self.spn_port = QSpinBox()
        self.spn_port.setRange(1024, 65535)
        self.spn_port.setValue(8765)
        row_cfg.addWidget(self.spn_port)

        row_cfg.addWidget(QLabel("Chunk Window:"))
        self.spn_chunk = QSpinBox()
        self.spn_chunk.setRange(1, 10)
        self.spn_chunk.setValue(2)
        self.spn_chunk.setSuffix(" s")
        row_cfg.addWidget(self.spn_chunk)

        self.chk_fast_hash = QCheckBox("Instant-NGP Fast Hash Grid")
        self.chk_fast_hash.setChecked(True)
        row_cfg.addWidget(self.chk_fast_hash)
        layout.addLayout(row_cfg)

        # Telemetry Status Box
        self.grp_telemetry = QGroupBox("Live Broadcaster Telemetry")
        layout_tel = QVBoxLayout(self.grp_telemetry)
        self.lbl_broadcast_status = QLabel("Server Status: ⚪ IDLE (Not Broadcasting)")
        self.lbl_broadcast_metrics = QLabel("Viewers: 0 | Bitrate: 0 kbps | Loss: 0.0000 | PSNR: 0.0 dB")
        layout_tel.addWidget(self.lbl_broadcast_status)
        layout_tel.addWidget(self.lbl_broadcast_metrics)
        layout.addWidget(self.grp_telemetry)

        layout.addStretch()

        # Action Buttons
        row_actions = QHBoxLayout()
        self.btn_toggle_broadcast = QPushButton("📡 Start Live Broadcast Server")
        self.btn_toggle_broadcast.setObjectName("PrimaryBtn")
        self.btn_toggle_broadcast.clicked.connect(self._on_toggle_broadcast)

        row_actions.addStretch()
        row_actions.addWidget(self.btn_toggle_broadcast)
        layout.addLayout(row_actions)

    def _on_source_type_changed(self, index: int) -> None:
        is_file = (index == 2)
        self.btn_browse_video.setVisible(is_file)
        self.lbl_selected_file.setVisible(is_file)

    def _on_browse_video(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Video File to Broadcast", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm);;All Files (*)"
        )
        if filepath:
            self.lbl_selected_file.setText(os.path.basename(filepath))
            self.lbl_selected_file.setProperty("fullpath", filepath)

    def _on_toggle_broadcast(self) -> None:
        if self.broadcast_server and self.broadcast_server.is_running:
            # Stop server
            self.broadcast_server.stop()
            self.btn_toggle_broadcast.setText("📡 Start Live Broadcast Server")
            self.btn_toggle_broadcast.setObjectName("PrimaryBtn")
            self.btn_toggle_broadcast.setStyleSheet("")
            self.lbl_broadcast_status.setText("Server Status: ⚪ IDLE (Stopped)")
            return

        # Determine source
        idx = self.cmb_source_type.currentIndex()
        if idx == 0:
            source = 0
        elif idx == 1:
            source = 1
        else:
            filepath = self.lbl_selected_file.property("fullpath")
            if not filepath or not os.path.exists(filepath):
                QMessageBox.warning(self, "No Video Selected", "Please select a video file to broadcast.")
                return
            source = filepath

        port = self.spn_port.value()
        chunk_duration = float(self.spn_chunk.value())
        use_hash = self.chk_fast_hash.isChecked()

        try:
            self.broadcast_server = NeuralBroadcastServer(
                source=source,
                host="0.0.0.0",
                port=port,
                chunk_duration=chunk_duration,
                epochs_per_chunk=80,
                use_hash_grid=use_hash,
            )
            self.broadcast_server.start()

            self.btn_toggle_broadcast.setText("⏹ Stop Live Broadcast Server")
            self.btn_toggle_broadcast.setObjectName("LiveBtn")
            self.btn_toggle_broadcast.setStyleSheet("background-color: #da3633; color: white;")
            self.lbl_broadcast_status.setText(f"Server Status: 🔴 BROADCASTING on ws://0.0.0.0:{port}")
        except Exception as e:
            QMessageBox.critical(self, "Broadcast Error", f"Failed to start broadcast server:\n{e}")

    def _update_broadcast_telemetry(self) -> None:
        if self.broadcast_server and self.broadcast_server.is_running:
            st = self.broadcast_server.get_status()
            self.lbl_broadcast_status.setText(
                f"Server Status: 🔴 BROADCASTING (Chunk #{st['chunk_idx']}) on ws://0.0.0.0:{st['port']}"
            )
            self.lbl_broadcast_metrics.setText(
                f"Viewers: 👥 {st['viewers']} | Bitrate: ⚡ {st['bitrate_kbps']:.1f} kbps | "
                f"PSNR: 🎯 {st['psnr_db']:.1f} dB | Sent: {st['total_bytes_mb']:.2f} MB"
            )
