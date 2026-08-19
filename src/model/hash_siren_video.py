"""Compact Hash-Grid Powered SIREN for 10X Accelerated Video Training & Inference."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.kernels.hash_encoder import SpatioTemporalHashGrid


class SineLayer(nn.Module):
    """Linear layer with periodic Sine activation and Sitzmann uniform initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / self.in_features, 1.0 / self.in_features)
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-1e-4, 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class HashSirenVideo(nn.Module):
    """Compact 3-layer Micro-SIREN accelerated by Spatio-Temporal Multi-Resolution Hash Grid."""

    def __init__(
        self,
        n_levels: int = 12,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 16,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        out_features: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.hash_grid = SpatioTemporalHashGrid(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
        )

        # Input dimension = Hash Features + Raw Coordinates (x, y, t)
        in_dim = self.hash_grid.out_dim + 3

        layers = []
        # First Sine Layer
        layers.append(
            SineLayer(
                in_features=in_dim,
                out_features=hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        # Hidden Sine Layers
        for _ in range(hidden_layers - 1):
            layers.append(
                SineLayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*layers)

        # Final Linear Output Layer
        self.final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            self.final_linear.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate coordinates (N, 3) in [-1.0, 1.0]^3 into RGB output (N, 3)."""
        # 1. Multi-resolution spatial-temporal hash encoding
        hash_feats = self.hash_grid(coords)  # (N, L * F)

        # 2. Concatenate with continuous coordinates for global gradient continuity
        full_feats = torch.cat([hash_feats, coords], dim=-1)  # (N, L*F + 3)

        # 3. Fast Micro-SIREN forward pass
        hidden = self.net(full_feats)
        output = self.final_linear(hidden)

        # Output in [0.0, 1.0] (RGB)
        return torch.clamp(output, 0.0, 1.0)
