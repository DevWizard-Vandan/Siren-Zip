"""Open Network Stream / YouTube URL Modal Dialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.tools.web_streamer import WebStreamer


class OpenURLDialog(QDialog):
    """Modal dialog allowing users to paste YouTube, Twitch, or direct web video streams."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("🌐 Open Network Stream (YouTube / Web)")
        self.setFixedSize(480, 180)
        self.resolved_stream_url: Optional[str] = None
        self.resolved_title: str = ""

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
            QLineEdit {
                background-color: #0d1117;
                border: 1px solid #30363d;
                color: #e6edf3;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #58a6ff;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                color: #c9d1d9;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton#PlayBtn {
                background-color: #238636;
                color: #ffffff;
                font-weight: bold;
                border-color: #2ea043;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        lbl = QLabel("Enter Network Stream URL (YouTube, Twitch, or Direct .mp4/.m3u8):")
        layout.addWidget(lbl)

        self.txt_url = QLineEdit()
        self.txt_url.setPlaceholderText("https://www.youtube.com/watch?v=...")
        layout.addWidget(self.txt_url)

        layout.addStretch()

        row_btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        self.btn_play = QPushButton("▶ Play Stream")
        self.btn_play.setObjectName("PlayBtn")
        self.btn_play.clicked.connect(self.on_resolve_and_play)

        row_btns.addStretch()
        row_btns.addWidget(btn_cancel)
        row_btns.addWidget(self.btn_play)
        layout.addLayout(row_btns)

    def on_resolve_and_play(self) -> None:
        url = self.txt_url.text().strip()
        if not url:
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid video stream URL.")
            return

        self.btn_play.setEnabled(False)
        self.btn_play.setText("🔍 Resolving with yt-dlp...")
        try:
            res = WebStreamer.resolve_stream_url(url)
            self.resolved_stream_url = res["stream_url"]
            self.resolved_title = res["title"]
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Stream Error", f"Could not resolve video stream:\n{e}")
        finally:
            self.btn_play.setEnabled(True)
            self.btn_play.setText("▶ Play Stream")
