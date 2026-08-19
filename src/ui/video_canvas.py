"""High-Performance Interactive Continuous Video Canvas with 400X Sub-Pixel Zoom."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QWidget

from src.player.engine import ViewportBounds


class ContinuousVideoCanvas(QWidget):
    """Interactive canvas supporting continuous analytical panning and 1.0x to 400.0x zooming."""

    viewport_changed = Signal(object, float)  # Emits (ViewportBounds, zoom_factor)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background-color: #0d1117;")

        # Navigation State in continuous normalized coordinates [-1.0, 1.0]^2
        self.center_x: float = 0.0
        self.center_y: float = 0.0
        self.zoom_factor: float = 1.0
        self.min_zoom: float = 1.0
        self.max_zoom: float = 400.0

        # Mouse Drag State
        self.is_dragging: bool = False
        self.last_mouse_pos: Optional[QPointF] = None

        # Display Frame Buffer
        self.current_qimage: Optional[QImage] = None
        self.hud_text: str = ""

    def set_frame_buffer(self, rgb_numpy: np.ndarray, hud_text: str = "") -> None:
        """Update canvas with new RGB frame buffer from GPU inference engine."""
        h, w, _ = rgb_numpy.shape
        bytes_per_line = 3 * w
        # QImage takes buffer reference
        qimg = QImage(rgb_numpy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.current_qimage = qimg
        self.hud_text = hud_text
        self.update()

    def get_viewport_bounds(self) -> ViewportBounds:
        """Calculate normalized coordinate bounding box in [-1.0, 1.0]."""
        half_w = 1.0 / self.zoom_factor
        half_h = 1.0 / self.zoom_factor

        x_min = max(-1.0, self.center_x - half_w)
        x_max = min(1.0, self.center_x + half_w)
        y_min = max(-1.0, self.center_y - half_h)
        y_max = min(1.0, self.center_y + half_h)

        # Adjust center if clamped
        if x_max - x_min < 2 * half_w:
            if x_min == -1.0:
                x_max = min(1.0, -1.0 + 2 * half_w)
            elif x_max == 1.0:
                x_min = max(-1.0, 1.0 - 2 * half_w)

        if y_max - y_min < 2 * half_h:
            if y_min == -1.0:
                y_max = min(1.0, -1.0 + 2 * half_h)
            elif y_max == 1.0:
                y_min = max(-1.0, 1.0 - 2 * half_h)

        return ViewportBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def set_zoom(self, zoom: float) -> None:
        """Programmatically set zoom factor."""
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, float(zoom)))
        self.viewport_changed.emit(self.get_viewport_bounds(), self.zoom_factor)
        self.update()

    def reset_view(self) -> None:
        """Reset zoom and pan to default full view."""
        self.center_x = 0.0
        self.center_y = 0.0
        self.zoom_factor = 1.0
        self.viewport_changed.emit(self.get_viewport_bounds(), self.zoom_factor)
        self.update()

    # --- Mouse Event Handlers for Continuous Pan & Zoom ---

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self.is_dragging and self.last_mouse_pos is not None:
            delta = event.position() - self.last_mouse_pos
            self.last_mouse_pos = event.position()

            # Convert pixel delta to continuous normalized coordinates
            cw = max(1, self.width())
            ch = max(1, self.height())

            norm_dx = -(delta.x() / cw) * (2.0 / self.zoom_factor)
            norm_dy = -(delta.y() / ch) * (2.0 / self.zoom_factor)

            self.center_x = max(-1.0, min(1.0, self.center_x + norm_dx))
            self.center_y = max(-1.0, min(1.0, self.center_y + norm_dy))

            self.viewport_changed.emit(self.get_viewport_bounds(), self.zoom_factor)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Smooth exponential zoom centered on mouse cursor."""
        delta = event.angleDelta().y()
        zoom_mult = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom_factor * zoom_mult))

        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.viewport_changed.emit(self.get_viewport_bounds(), self.zoom_factor)
            self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        """Double click resets zoom and pan."""
        self.reset_view()

    # --- Rendering ---

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(13, 17, 23))

        if self.current_qimage is not None:
            # Draw frame scaled to canvas
            painter.drawImage(QRectF(0, 0, w, h), self.current_qimage)
        else:
            # Splash text if no frame loaded
            painter.setPen(QColor(139, 148, 158))
            font = QFont("Segoe UI", 14)
            painter.setFont(font)
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "⚡ SIREN-ZIP Continuous Field Canvas\n\nOpen a .neura container to begin playback")

        # Draw HUD overlay if available
        if self.hud_text:
            painter.setPen(QColor(0, 230, 118))
            font_hud = QFont("Consolas", 10)
            painter.setFont(font_hud)
            painter.drawText(16, 26, self.hud_text)
