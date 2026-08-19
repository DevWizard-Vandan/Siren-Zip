"""ConvSIREN: Sinusoidal Fourier Convolutional Video Generator for Crystal Sharp Neural Cinema."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SineConv2d(nn.Module):
    """2D Convolution with SIREN Sinusoidal Activation and Principal Frequency Scaling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, omega: float = 30.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.omega = omega

        # SIREN weight initialization: w ~ Uniform(-sqrt(6/n)/omega, sqrt(6/n)/omega)
        with torch.no_grad():
            limit = math.sqrt(6.0 / (in_channels * kernel_size * kernel_size)) / omega
            self.conv.weight.uniform_(-limit, limit)
            self.conv.bias.uniform_(-1.0 / math.sqrt(in_channels), 1.0 / math.sqrt(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.conv(x))


class ConvSIRENVideo(nn.Module):
    """Full-Frame Sinusoidal Convolutional Neural Video Generator.
    
    Maps frame timestamps / indices directly to pristine (3, H, W) RGB frames in a single GPU pass (<15ms).
    """

    def __init__(
        self,
        num_frames: int,
        latent_dim: int = 32,
        stem_dim: int = 192,
        target_height: int = 720,
        target_width: int = 1280,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.stem_dim = stem_dim
        self.target_height = target_height
        self.target_width = target_width

        # 1. Temporal Latent Embeddings (one compact vector per frame)
        self.temporal_embed = nn.Embedding(num_frames, latent_dim)

        # 2. Base Grid: 5 x 9 (16:9 aspect ratio)
        self.base_h = 5
        self.base_w = 9
        self.stem = nn.Linear(latent_dim, stem_dim * self.base_h * self.base_w)

        # 3. 6 Progressive Sinusoidal Convolutional Upscaling Stages
        # 5x9 -> 10x18 -> 20x36 -> 40x72 -> 80x144 -> 160x288 -> 320x576 -> (bilinear 720x1280 or 1080x1920)
        self.conv1 = SineConv2d(stem_dim, 144, kernel_size=3, padding=1, omega=30.0)
        self.up1 = nn.PixelShuffle(2)  # 36 x 10 x 18

        self.conv2 = SineConv2d(36, 128, kernel_size=3, padding=1, omega=30.0)
        self.up2 = nn.PixelShuffle(2)  # 32 x 20 x 36

        self.conv3 = SineConv2d(32, 96, kernel_size=3, padding=1, omega=30.0)
        self.up3 = nn.PixelShuffle(2)  # 24 x 40 x 72

        self.conv4 = SineConv2d(24, 64, kernel_size=3, padding=1, omega=30.0)
        self.up4 = nn.PixelShuffle(2)  # 16 x 80 x 144

        self.conv5 = SineConv2d(16, 48, kernel_size=3, padding=1, omega=30.0)
        self.up5 = nn.PixelShuffle(2)  # 12 x 160 x 288

        self.conv6 = SineConv2d(12, 32, kernel_size=3, padding=1, omega=30.0)
        self.up6 = nn.PixelShuffle(2)  # 8 x 320 x 576

        # 4. Final Direct RGB Head
        self.head = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass: maps 1D frame index tensor (B,) to RGB frames (B, 3, target_h, target_w)."""
        # Ensure 1D indices
        if frame_indices.dim() > 1:
            frame_indices = frame_indices.squeeze(-1)
        if frame_indices.dtype != torch.long:
            frame_indices = torch.clamp(frame_indices, 0, self.num_frames - 1).long()

        B = frame_indices.shape[0]
        latents = self.temporal_embed(frame_indices)
        stem = self.stem(latents).view(B, self.stem_dim, self.base_h, self.base_w)

        h = self.up1(self.conv1(stem))
        h = self.up2(self.conv2(h))
        h = self.up3(self.conv3(h))
        h = self.up4(self.conv4(h))
        h = self.up5(self.conv5(h))
        h = self.up6(self.conv6(h))
        rgb = self.head(h)

        if rgb.shape[-2:] != (self.target_height, self.target_width):
            rgb = F.interpolate(
                rgb,
                size=(self.target_height, self.target_width),
                mode="bilinear",
                align_corners=False,
            )

        return rgb
