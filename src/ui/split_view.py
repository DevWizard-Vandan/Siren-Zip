"""Interactive Side-by-Side Split View: Discrete H.264 vs Continuous SIREN-Zip."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from src.types.viewport import ViewportBounds


class SplitComparisonView(QWidget):
    """Split-screen widget with draggable divider comparing discrete bitmap vs continuous INR."""

    viewport_changed = Signal(object, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background-color: #0d1117;")

        # Split position ratio (0.0 to 1.0, default 0.5 center)
        self.split_ratio: float = 0.5
        self.is_dragging_split: bool = False
        self.is_panning: bool = False
        self.last_mouse_pos: Optional[QPointF] = None

        # Viewport Navigation
        self.center_x: float = 0.0
        self.center_y: float = 0.0
        self.zoom_factor: float = 1.0
        self.min_zoom: float = 1.0
        self.max_zoom: float = 400.0

        # Frame buffers
        self.siren_rgb: Optional[np.ndarray] = None
        self.discrete_rgb: Optional[np.ndarray] = None
        self.hud_text: str = ""

    def update_buffers(
        self,
        siren_rgb: np.ndarray,
        discrete_full_frame: Optional[np.ndarray],
        viewport: ViewportBounds,
        hud_text: str = "",
    ) -> None:
        """Update both SIREN prediction and discrete cropped baseline frame."""
        self.siren_rgb = siren_rgb
        self.hud_text = hud_text

        if discrete_full_frame is not None:
            # Crop discrete baseline to the exact same normalized viewport
            orig_h, orig_w, _ = discrete_full_frame.shape
            px_min = max(0, min(orig_w - 2, int(((viewport.x_min + 1.0) / 2.0) * orig_w)))
            px_max = max(px_min + 1, min(orig_w, int(((viewport.x_max + 1.0) / 2.0) * orig_w)))
            py_min = max(0, min(orig_h - 2, int(((viewport.y_min + 1.0) / 2.0) * orig_h)))
            py_max = max(py_min + 1, min(orig_h, int(((viewport.y_max + 1.0) / 2.0) * orig_h)))

            cropped = discrete_full_frame[py_min:py_max, px_min:px_max]
            # Nearest neighbor scaling to show discrete pixels at high zoom
            target_h, target_w, _ = siren_rgb.shape
            self.discrete_rgb = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        else:
            self.discrete_rgb = None

        self.update()

    def get_viewport_bounds(self) -> ViewportBounds:
        half_w = 1.0 / self.zoom_factor
        half_h = 1.0 / self.zoom_factor

        x_min = max(-1.0, self.center_x - half_w)
        x_max = min(1.0, self.center_x + half_w)
        y_min = max(-1.0, self.center_y - half_h)
        y_max = min(1.0, self.center_y + half_h)

        return ViewportBounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    def set_zoom(self, zoom: float) -> None:
        self.zoom_factor = max(self.min_zoom, min(self.max_zoom, float(zoom)))
        self.viewport_changed.emit(self.get_viewport_bounds(), self.zoom_factor)
        self.update()

    def reset_view(self) -> None:
        self.center_x = 0.0
        self.center_y = 0.0
        self.zoom_factor = 1.0
        self.split_ratio = 0.5
        self.viewport_changed.emit(self.get_viewport_bounds(), self.zoom_factor)
        self.update()

    # --- Mouse Handlers for Draggable Splitter & Viewport Pan/Zoom ---

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        split_x = self.width() * self.split_ratio
        if abs(pos.x() - split_x) <= 12:
            self.is_dragging_split = True
            self.setCursor(Qt.CursorShape.SplitHCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        split_x = self.width() * self.split_ratio

        if not self.is_dragging_split and not self.is_panning:
            if abs(pos.x() - split_x) <= 12:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.is_dragging_split:
            self.split_ratio = max(0.05, min(0.95, pos.x() / max(1, self.width())))
            self.update()
        elif self.is_panning and self.last_mouse_pos is not None:
            delta = pos - self.last_mouse_pos
            self.last_mouse_pos = pos

            cw = max(1, self.width())
            ch = max(1, self.height())
            norm_dx = -(delta.x() / cw) * (2.0 / self.zoom_factor)
            norm_dy = -(delta.y() / ch) * (2.0 / self.zoom_factor)

            self.center_x = max(-1.0, min(1.0, self.center_x + norm_dx))
            self.center_y = max(-1.0, min(1.0, self.center_y + norm_dy))

            self.viewport_changed.emit(self.get_viewport_bounds(), self.zoom_factor)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self.is_dragging_split = False
        self.is_panning = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        zoom_mult = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom_factor * zoom_mult))

        if new_zoom != self.zoom_factor:
            self.zoom_factor = new_zoom
            self.viewport_changed.emit(self.get_viewport_bounds(), self.zoom_factor)
            self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_view()

    # --- Paint Event ---

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = self.width()
        h = self.height()
        split_x = int(w * self.split_ratio)

        # 1. Draw Left Side (Discrete H.264 Baseline)
        if self.discrete_rgb is not None:
            dh, dw, _ = self.discrete_rgb.shape
            qimg_left = QImage(self.discrete_rgb.data, dw, dh, 3 * dw, QImage.Format.Format_RGB888)
            # Clip painter to left side
            painter.save()
            painter.setClipRect(0, 0, split_x, h)
            painter.drawImage(QRectF(0, 0, w, h), qimg_left)
            painter.restore()
        else:
            painter.fillRect(0, 0, split_x, h, QColor(22, 27, 34))
            painter.setPen(QColor(139, 148, 158))
            painter.drawText(QRectF(0, 0, split_x, h), Qt.AlignmentFlag.AlignCenter, "Load reference .mp4 to enable split comparison")

        # 2. Draw Right Side (SIREN Continuous INR)
        if self.siren_rgb is not None:
            sh, sw, _ = self.siren_rgb.shape
            qimg_right = QImage(self.siren_rgb.data, sw, sh, 3 * sw, QImage.Format.Format_RGB888)
            painter.save()
            painter.setClipRect(split_x, 0, w - split_x, h)
            painter.drawImage(QRectF(0, 0, w, h), qimg_right)
            painter.restore()

        # 3. Draw Splitter Divider Line & Handle
        pen_divider = QPen(QColor(0, 230, 118), 3)
        painter.setPen(pen_divider)
        painter.drawLine(split_x, 0, split_x, h)

        # Center Grip Handle
        handle_w = 28
        handle_h = 44
        handle_y = (h - handle_h) // 2
        painter.fillRect(split_x - handle_w // 2, handle_y, handle_w, handle_h, QColor(0, 230, 118))
        painter.setPen(QColor(13, 17, 23))
        font_arrows = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(font_arrows)
        painter.drawText(QRectF(split_x - handle_w // 2, handle_y, handle_w, handle_h), Qt.AlignmentFlag.AlignCenter, "◀ ▶")

        # 4. Top Header Badges
        font_badge = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font_badge)

        # Left Badge: Discrete
        painter.fillRect(16, 14, 250, 30, QColor(200, 40, 40, 210))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(26, 34, f"TRADITIONAL H.264 ({self.zoom_factor:.1f}X)")

        # Right Badge: SIREN Continuous
        badge_right_w = 260
        painter.fillRect(w - badge_right_w - 16, 14, badge_right_w, 30, QColor(20, 140, 60, 210))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(w - badge_right_w - 6, 34, f"SIREN-ZIP CONTINUOUS ({self.zoom_factor:.1f}X)")

        # 5. Diagnostic HUD Overlay
        if self.hud_text:
            painter.setPen(QColor(0, 230, 118))
            font_hud = QFont("Consolas", 10)
            painter.setFont(font_hud)
            painter.drawText(16, h - 16, self.hud_text)
