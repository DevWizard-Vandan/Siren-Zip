"""Perceptual Color Transforms (Oklab and YCbCr) for Human Visual System (HVS) Optimization."""

from __future__ import annotations

import torch
import torch.nn as nn


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear/sRGB tensor (..., 3) in [0, 1] to Oklab color space (L, a, b).
    
    L: Lightness [0, 1] (80% perceptual weight)
    a: Green-Red opponent channel [-0.5, 0.5] (10% perceptual weight)
    b: Blue-Yellow opponent channel [-0.5, 0.5] (10% perceptual weight)
    """
    # 1. Linear sRGB to LMS
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    # 2. Non-linear cube root compression
    l_ = torch.pow(torch.clamp(l, min=1e-7), 1.0 / 3.0)
    m_ = torch.pow(torch.clamp(m, min=1e-7), 1.0 / 3.0)
    s_ = torch.pow(torch.clamp(s, min=1e-7), 1.0 / 3.0)

    # 3. LMS to Oklab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return torch.stack([L, a, b_out], dim=-1)


def oklab_to_rgb(oklab: torch.Tensor) -> torch.Tensor:
    """Convert Oklab tensor (..., 3) back to sRGB [0, 1]."""
    L = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_out = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    rgb = torch.stack([r, g, b_out], dim=-1)
    return torch.clamp(rgb, 0.0, 1.0)


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor (..., 3) to standard BT.709 YCbCr."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556 + 0.5
    cr = (r - y) / 1.5748 + 0.5

    return torch.stack([y, cb, cr], dim=-1)


def ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert BT.709 YCbCr tensor (..., 3) back to RGB."""
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1] - 0.5
    cr = ycbcr[..., 2] - 0.5

    r = y + 1.5748 * cr
    b = y + 1.8556 * cb
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152

    rgb = torch.stack([r, g, b], dim=-1)
    return torch.clamp(rgb, 0.0, 1.0)
