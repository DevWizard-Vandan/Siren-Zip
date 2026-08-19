"""SRT and WebVTT Subtitle Parser and Antialiased Vector Canvas Renderer."""

from __future__ import annotations

import os
import re
from typing import List, NamedTuple, Optional

import cv2
import numpy as np


class SubtitleItem(NamedTuple):
    start_sec: float
    end_sec: float
    text: str


class SubtitleEngine:
    """Parses SRT/VTT files and renders crisp, antialiased subtitle overlays."""

    def __init__(self, subtitle_path: Optional[str] = None) -> None:
        self.subtitles: List[SubtitleItem] = []
        self.is_loaded = False
        if subtitle_path and os.path.exists(subtitle_path):
            self.load_file(subtitle_path)

    @staticmethod
    def _parse_time(time_str: str) -> float:
        """Parse '00:01:23,456' or '00:01:23.456' into seconds."""
        time_str = time_str.strip().replace(",", ".")
        parts = time_str.split(":")
        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600.0 + m * 60.0 + s
        elif len(parts) == 2:
            m = float(parts[0])
            s = float(parts[1])
            return m * 60.0 + s
        return 0.0

    def load_file(self, file_path: str) -> bool:
        """Load SRT or VTT file into memory."""
        self.subtitles = []
        if not os.path.exists(file_path):
            self.is_loaded = False
            return False

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Normalize linebreaks
            blocks = re.split(r"\n\s*\n", content.strip())
            for block in blocks:
                lines = [line.strip() for line in block.split("\n") if line.strip()]
                if not lines:
                    continue

                # Locate timestamp line
                time_line_idx = -1
                for i, line in enumerate(lines):
                    if "-->" in line:
                        time_line_idx = i
                        break

                if time_line_idx == -1:
                    continue

                time_line = lines[time_line_idx]
                time_parts = time_line.split("-->")
                start_sec = self._parse_time(time_parts[0])
                end_sec = self._parse_time(time_parts[1].split()[0])

                # Remaining lines are subtitle text
                text_lines = lines[time_line_idx + 1 :]
                # Strip HTML/VTT tags e.g. <c>, <b>
                raw_text = " ".join(text_lines)
                clean_text = re.sub(r"<[^>]+>", "", raw_text).strip()

                if clean_text:
                    self.subtitles.append(
                        SubtitleItem(
                            start_sec=start_sec,
                            end_sec=end_sec,
                            text=clean_text,
                        )
                    )

            self.subtitles.sort(key=lambda s: s.start_sec)
            self.is_loaded = len(self.subtitles) > 0
            return self.is_loaded

        except Exception:
            self.is_loaded = False
            return False

    def get_active_text(self, current_sec: float) -> Optional[str]:
        """Find subtitle text active at timestamp current_sec."""
        if not self.is_loaded or not self.subtitles:
            return None

        # Binary search / Linear scan
        for item in self.subtitles:
            if item.start_sec <= current_sec <= item.end_sec:
                return item.text
            elif item.start_sec > current_sec:
                break
        return None

    def render_overlay(
        self,
        rgb_image: np.ndarray,
        current_sec: float,
        font_scale: float = 0.85,
        text_color: tuple[int, int, int] = (255, 255, 255),
        border_color: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """Render antialiased vector subtitle text with outline over canvas."""
        text = self.get_active_text(current_sec)
        if not text:
            return rgb_image

        h, w, _ = rgb_image.shape
        img = rgb_image.copy()

        font = cv2.FONT_HERSHEY_DUPLEX
        thickness_border = max(2, int(font_scale * 3.5))
        thickness_text = max(1, int(font_scale * 1.8))

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness_border)
        x = max(10, (w - text_w) // 2)
        y = int(h - text_h - 20)

        # Draw black drop-shadow / outline
        cv2.putText(img, text, (x, y), font, font_scale, border_color, thickness_border, cv2.LINE_AA)
        # Draw crisp white text
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness_text, cv2.LINE_AA)

        return img
