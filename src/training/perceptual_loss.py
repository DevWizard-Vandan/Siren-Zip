"""Perceptual Cinema Loss (HVS-Optimized L1 + MS-SSIM + High-Frequency Edge Penalty)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_window_1d(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def create_window_2d(window_size: int = 11, channel: int = 3, sigma: float = 1.5) -> torch.Tensor:
    """Create 2D Gaussian filter kernel for SSIM."""
    _1d = create_window_1d(window_size, sigma).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_metric(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int = 11,
    channel: int = 3,
) -> torch.Tensor:
    """Compute Structural Similarity Index (SSIM) on 4D batch tensors (B, C, H, W)."""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7
    )
    return ssim_map.mean()


class PerceptualCinemaLoss(nn.Module):
    """Human Visual System (HVS) Perceptual Loss combining L1, MS-SSIM, and Edge Gradient supervision.
    
    Ignores sub-pixel camera sensor noise and focuses 100% capacity on visible contrast,
    facial details, architectural lines, and lighting.
    """

    def __init__(
        self,
        l1_weight: float = 0.50,
        ssim_weight: float = 0.30,
        edge_weight: float = 0.20,
        noise_deadband: float = 0.012,
        window_size: int = 11,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.noise_deadband = noise_deadband
        self.window_size = window_size

        # Pre-register Sobel filters for edge-gradient supervision
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1), persistent=False)
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1), persistent=False)
        self.register_buffer("ssim_window", create_window_2d(window_size, 3), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute perceptual loss between predicted and ground-truth frame tensors.
        
        Args:
            pred: (B, 3, H, W) or (N, 3) in [0.0, 1.0]
            target: (B, 3, H, W) or (N, 3) in [0.0, 1.0]
        """
        # If flat point batches, compute deadband L1 loss
        if pred.dim() == 2:
            diff = torch.abs(pred - target)
            # Apply deadband threshold: ignore sensor noise below threshold
            diff_active = F.relu(diff - self.noise_deadband)
            loss_l1 = diff_active.mean()
            return loss_l1, {"loss_l1": loss_l1.item(), "loss_ssim": 0.0, "loss_edge": 0.0}

        # 1. Deadband L1 Loss (Color & Luminance)
        diff = torch.abs(pred - target)
        diff_active = F.relu(diff - self.noise_deadband)
        loss_l1 = diff_active.mean()

        # Adaptive Multi-Scale Downsampling for 4K UHD VRAM optimization
        if pred.shape[-2] > 1080:
            pred_eval = F.avg_pool2d(pred, 2)
            target_eval = F.avg_pool2d(target, 2)
        else:
            pred_eval = pred
            target_eval = target

        # 2. Structural Similarity (MS-SSIM / SSIM)
        if pred_eval.shape[-1] >= self.window_size and pred_eval.shape[-2] >= self.window_size:
            ssim_val = ssim_metric(pred_eval, target_eval, self.ssim_window, self.window_size, channel=3)
            loss_ssim = 1.0 - ssim_val
        else:
            loss_ssim = torch.tensor(0.0, device=pred.device)

        # 3. High-Frequency Edge Gradient Loss (Sobel Edge Preservation)
        grad_pred_x = F.conv2d(pred_eval, self.sobel_x, padding=1, groups=3)
        grad_pred_y = F.conv2d(pred_eval, self.sobel_y, padding=1, groups=3)

        grad_target_x = F.conv2d(target_eval, self.sobel_x, padding=1, groups=3)
        grad_target_y = F.conv2d(target_eval, self.sobel_y, padding=1, groups=3)

        loss_edge = F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

        # Total Perceptual Weighted Loss
        total_loss = (
            self.l1_weight * loss_l1
            + self.ssim_weight * loss_ssim
            + self.edge_weight * loss_edge
        )

        metrics = {
            "loss_l1": loss_l1.item(),
            "loss_ssim": loss_ssim.item() if isinstance(loss_ssim, torch.Tensor) else 0.0,
            "loss_edge": loss_edge.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics
