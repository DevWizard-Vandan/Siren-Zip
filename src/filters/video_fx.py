"""Real-Time Video Equalizer and SIREN Micro-Texture Detail Booster."""

from __future__ import annotations

import cv2
import numpy as np


class VideoFXFilter:
    """Real-time shader filter applying Brightness, Contrast, Saturation, Gamma, and SIREN Detail Boost."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        gamma: float = 1.0,
        detail_boost: float = 0.0,
    ) -> None:
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.gamma = float(gamma)
        self.detail_boost = float(detail_boost)

    def is_identity(self) -> bool:
        """Check if filter requires processing."""
        return (
            abs(self.brightness) < 1e-4
            and abs(self.contrast - 1.0) < 1e-4
            and abs(self.saturation - 1.0) < 1e-4
            and abs(self.gamma - 1.0) < 1e-4
            and abs(self.detail_boost) < 1e-4
        )

    def apply(self, img_rgb: np.ndarray) -> np.ndarray:
        """Apply video equalizer adjustments to uint8 (H, W, 3) RGB image."""
        if self.is_identity():
            return img_rgb

        x = img_rgb.astype(np.float32) / 255.0

        # 1. Contrast & Brightness: C * (x - 0.5) + 0.5 + B
        if abs(self.contrast - 1.0) > 1e-4 or abs(self.brightness) > 1e-4:
            x = self.contrast * (x - 0.5) + 0.5 + self.brightness

        # 2. Gamma: x^(1/gamma)
        if abs(self.gamma - 1.0) > 1e-4:
            g = max(0.1, self.gamma)
            x = np.power(np.maximum(x, 0.0), 1.0 / g)

        # 3. Saturation: lerp(gray, rgb, S)
        if abs(self.saturation - 1.0) > 1e-4:
            gray = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
            gray = np.expand_dims(gray, axis=-1)
            x = gray + self.saturation * (x - gray)

        # 4. SIREN Micro-Texture Detail Booster (High-Pass Unsharp Masking)
        if self.detail_boost > 1e-4:
            blurred = cv2.GaussianBlur(x, (0, 0), sigmaX=1.0, sigmaY=1.0)
            high_freq = x - blurred
            x = x + self.detail_boost * high_freq

        out_uint8 = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)
        return out_uint8
