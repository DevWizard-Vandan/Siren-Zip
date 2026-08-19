"""SMPTE ST.2084 Perceptual Quantizer (PQ), HLG, and Rec.709/Rec.2020 Color Math."""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

# --- SMPTE ST.2084 (Perceptual Quantizer / PQ) Constants ---
M1 = 2610.0 / 16384.0  # 0.1593017578125
M2 = (2523.0 / 4096.0) * 128.0  # 78.84375
C1 = 3424.0 / 4096.0  # 0.8359375
C2 = (2413.0 / 4096.0) * 32.0  # 18.8515625
C3 = (2392.0 / 4096.0) * 32.0  # 18.6875


def pq_to_linear(
    n_pq: Union[torch.Tensor, np.ndarray],
    peak_nits: float = 10000.0,
) -> Union[torch.Tensor, np.ndarray]:
    """Inverse Perceptual Quantizer: maps non-linear PQ signal [0, 1] to linear luminance (nits / 10000).

    Formula:
        L = ( max(N^(1/m2) - c1, 0) / (c2 - c3 * N^(1/m2)) )^(1/m1)
    """
    if isinstance(n_pq, torch.Tensor):
        n_clamped = torch.clamp(n_pq, min=0.0, max=1.0)
        n_pow = torch.pow(n_clamped, 1.0 / M2)
        numerator = torch.clamp(n_pow - C1, min=0.0)
        denominator = torch.clamp(C2 - C3 * n_pow, min=1e-8)
        linear = torch.pow(numerator / denominator, 1.0 / M1)
        return linear
    else:
        n_clamped = np.clip(n_pq, 0.0, 1.0)
        n_pow = np.power(n_clamped, 1.0 / M2)
        numerator = np.maximum(n_pow - C1, 0.0)
        denominator = np.maximum(C2 - C3 * n_pow, 1e-8)
        linear = np.power(numerator / denominator, 1.0 / M1)
        return linear


def linear_to_pq(
    linear: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """Forward Perceptual Quantizer: maps linear luminance [0, 1] to non-linear PQ [0, 1].

    Formula:
        N = ( (c1 + c2 * Y^m1) / (1 + c3 * Y^m1) )^m2
    """
    if isinstance(linear, torch.Tensor):
        y_clamped = torch.clamp(linear, min=0.0, max=1.0)
        y_pow = torch.pow(y_clamped, M1)
        numerator = C1 + C2 * y_pow
        denominator = 1.0 + C3 * y_pow
        pq = torch.pow(numerator / denominator, M2)
        return pq
    else:
        y_clamped = np.clip(linear, 0.0, 1.0)
        y_pow = np.power(y_clamped, M1)
        numerator = C1 + C2 * y_pow
        denominator = 1.0 + C3 * y_pow
        pq = np.power(numerator / denominator, M2)
        return pq


def hlg_to_linear(
    hlg: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """Inverse Hybrid Log-Gamma (HLG / ARIB STD-B67) to linear luminance."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    if isinstance(hlg, torch.Tensor):
        linear = torch.where(
            hlg <= 0.5,
            (hlg ** 2) / 3.0,
            (torch.exp((hlg - c) / a) + b) / 12.0,
        )
        return linear
    else:
        return np.where(
            hlg <= 0.5,
            (hlg ** 2) / 3.0,
            (np.exp((hlg - c) / a) + b) / 12.0,
        )


def linear_to_hlg(
    linear: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """Forward linear luminance to Hybrid Log-Gamma (HLG)."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    if isinstance(linear, torch.Tensor):
        hlg = torch.where(
            linear <= 1.0 / 12.0,
            torch.sqrt(3.0 * linear),
            a * torch.log(torch.clamp(12.0 * linear - b, min=1e-8)) + c,
        )
        return hlg
    else:
        return np.where(
            linear <= 1.0 / 12.0,
            np.sqrt(3.0 * np.maximum(linear, 0.0)),
            a * np.log(np.maximum(12.0 * linear - b, 1e-8)) + c,
        )


# --- 3x3 Color Gamut Transformation Matrices ---

# Rec.2020 to Rec.709 Linear RGB Matrix
MAT_REC2020_TO_REC709 = np.array(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ],
    dtype=np.float32,
)

# Rec.709 to Rec.2020 Linear RGB Matrix
MAT_REC709_TO_REC2020 = np.array(
    [
        [0.6274, 0.3293, 0.0433],
        [0.0691, 0.9195, 0.0114],
        [0.0164, 0.0880, 0.8956],
    ],
    dtype=np.float32,
)


def rec2020_to_rec709(rgb: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert Wide Gamut Rec.2020 RGB to standard HDTV Rec.709 RGB."""
    if isinstance(rgb, torch.Tensor):
        mat = torch.from_numpy(MAT_REC2020_TO_REC709).to(device=rgb.device, dtype=rgb.dtype)
        orig_shape = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)
        converted = torch.matmul(rgb_flat, mat.T)
        return converted.reshape(orig_shape)
    else:
        orig_shape = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)
        converted = np.matmul(rgb_flat, MAT_REC2020_TO_REC709.T)
        return converted.reshape(orig_shape)


def rec709_to_rec2020(rgb: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Convert standard Rec.709 RGB to Wide Gamut Rec.2020 RGB."""
    if isinstance(rgb, torch.Tensor):
        mat = torch.from_numpy(MAT_REC709_TO_REC2020).to(device=rgb.device, dtype=rgb.dtype)
        orig_shape = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)
        converted = torch.matmul(rgb_flat, mat.T)
        return converted.reshape(orig_shape)
    else:
        orig_shape = rgb.shape
        rgb_flat = rgb.reshape(-1, 3)
        converted = np.matmul(rgb_flat, MAT_REC709_TO_REC2020.T)
        return converted.reshape(orig_shape)
