"""Vectorized PSNR, SSIM, and compression metrics for SIREN image codec."""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F


def calculate_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Calculate Mean Squared Error between prediction and target tensors."""
    mse_tensor = F.mse_loss(pred, target)
    return float(mse_tensor.item())


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB on GPU or CPU.

    Formula: PSNR = 10 * log10(max_val^2 / (MSE + eps))

    Args:
        pred: Predicted RGB tensor in [0.0, 1.0].
        target: Target RGB tensor in [0.0, 1.0].
        max_val: Maximum dynamic range (default: 1.0).
        eps: Small epsilon for numerical stability.

    Returns:
        psnr: PSNR value in decibels (dB).
    """
    mse = F.mse_loss(pred, target)
    if mse.item() < eps:
        return 100.0
    psnr = 10.0 * torch.log10((max_val ** 2) / (mse + eps))
    return float(psnr.item())


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian filter kernel for SSIM calculation."""
    gauss = torch.tensor(
        [math.exp(-((x - window_size // 2) ** 2) / (2.0 * sigma ** 2)) for x in range(window_size)],
        device=device,
        dtype=torch.float32,
    )
    gauss = gauss / gauss.sum()
    _1d_window = gauss.unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channels, 1, window_size, window_size).contiguous()
    return window


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> float:
    """Calculate Structural Similarity Index Measure (SSIM) on GPU/CPU.

    Args:
        pred: Predicted tensor of shape (H, W, C) or (B, C, H, W) in range [0, 1].
        target: Target tensor of shape (H, W, C) or (B, C, H, W) in range [0, 1].
        window_size: Gaussian kernel window size (default: 11).
        sigma: Gaussian kernel standard deviation (default: 1.5).
        max_val: Maximum dynamic range (default: 1.0).

    Returns:
        ssim: Scalar SSIM score in [0.0, 1.0].
    """
    # Standardize shape to (B, C, H, W)
    if pred.ndim == 3:  # (H, W, C)
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.ndim == 4 and pred.shape[-1] in (1, 3):  # (B, H, W, C)
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

    device = pred.device
    channels = pred.shape[1]

    window = _gaussian_window(window_size, sigma, channels, device)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())


def calculate_metrics_all(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute MSE, PSNR, and SSIM simultaneously."""
    mse = calculate_mse(pred, target)
    psnr = calculate_psnr(pred, target)
    ssim = calculate_ssim(pred, target)
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
    }
