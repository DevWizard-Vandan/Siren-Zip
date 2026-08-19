"""Sleek On-Screen Display (OSD) HUD Notification Popup."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QLabel, QWidget


class OSDOverlay(QLabel):
    """Semi-transparent floating notification overlay."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            """
            QLabel {
                background-color: rgba(15, 23, 42, 210);
                color: #00e676;
                border: 1px solid rgba(0, 230, 118, 120);
                border-radius: 8px;
                padding: 10px 18px;
                font-family: 'Consolas', 'Segoe UI', monospace;
                font-size: 14px;
                font-weight: bold;
            }
            """
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hide()

        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide)

    def show_notification(self, message: str, duration_ms: int = 1400) -> None:
        """Display an on-screen notification centered at top-right of canvas."""
        self.setText(message)
        self.adjustSize()

        if self.parentWidget():
            pw = self.parentWidget().width()
            w = self.width()
            # Position at top-right of canvas
            self.move(pw - w - 24, 24)

        self.show()
        self.raise_()
        self.hide_timer.start(duration_ms)
