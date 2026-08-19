"""Multi-Resolution Spatio-Temporal Hash Grid Encoder (Instant-NGP Style for Continuous INR Video)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Large prime numbers for spatial-temporal hashing (Instant-NGP specification)
PRIMES = [1, 2654435761, 805453171]


class SpatioTemporalHashGrid(nn.Module):
    """Multi-resolution spatial-temporal hash grid encoder for (x, y, t) in [-1.0, 1.0]^3."""

    def __init__(
        self,
        n_levels: int = 12,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 16,
        base_resolution: int = 16,
        max_resolution: int = 1024,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2**log2_hashmap_size
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.out_dim = n_levels * n_features_per_level

        # Growth factor b per resolution level
        self.growth_factor = math.exp(
            (math.log(max_resolution) - math.log(base_resolution)) / max(1, n_levels - 1)
        )

        # Multi-resolution embedding tables
        self.embeddings = nn.ParameterList([
            nn.Parameter(
                torch.empty(self.hashmap_size, n_features_per_level).uniform_(-1e-4, 1e-4)
            )
            for _ in range(n_levels)
        ])

        # Register corner offsets {0, 1}^3 for 3D trilinear interpolation (8 corners)
        corners = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.long,
        )
        self.register_buffer("corners", corners, persistent=False)
        self.register_buffer(
            "primes", torch.tensor(PRIMES, dtype=torch.long), persistent=False
        )

    def _hash_corners(self, coords: torch.Tensor) -> torch.Tensor:
        """Hash 3D integer coordinates (N, 8, 3) into hash table indices (N, 8)."""
        # coords: (N, 8, 3)
        # Spatial-temporal XOR hashing
        p = self.primes
        h = (coords[..., 0] * p[0]) ^ (coords[..., 1] * p[1]) ^ (coords[..., 2] * p[2])
        # Modulo table size
        return torch.remainder(h, self.hashmap_size)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """Encode continuous coordinates (N, 3) in [-1.0, 1.0]^3 into multi-resolution features (N, L*F)."""
        # 1. Normalize from [-1.0, 1.0]^3 to [0.0, 1.0]^3
        p = (xyt + 1.0) * 0.5
        p = torch.clamp(p, 0.0, 1.0 - 1e-6)

        N = p.shape[0]
        level_features = []

        for l in range(self.n_levels):
            # 2. Compute grid resolution at level l
            res = math.floor(self.base_resolution * (self.growth_factor**l))
            scaled_p = p * res  # (N, 3)

            # Integer lower corner and fractional offset
            p0 = torch.floor(scaled_p).long()  # (N, 3)
            diff = scaled_p - p0.float()  # (N, 3)

            # 3. 8 corner integer coordinates (N, 8, 3)
            corner_coords = p0.unsqueeze(1) + self.corners.unsqueeze(0)  # (N, 8, 3)

            # 4. Hash indices for all 8 corners (N, 8)
            corner_indices = self._hash_corners(corner_coords)  # (N, 8)

            # 5. Gather feature vectors (N, 8, F)
            features = self.embeddings[l][corner_indices]  # (N, 8, F)

            # 6. Trilinear interpolation weights (N, 8, 1)
            # w = (1 - dx or dx) * (1 - dy or dy) * (1 - dt or dt)
            dx = diff[:, 0:1]  # (N, 1)
            dy = diff[:, 1:2]
            dt = diff[:, 2:3]

            w000 = (1.0 - dx) * (1.0 - dy) * (1.0 - dt)
            w001 = (1.0 - dx) * (1.0 - dy) * dt
            w010 = (1.0 - dx) * dy * (1.0 - dt)
            w011 = (1.0 - dx) * dy * dt
            w100 = dx * (1.0 - dy) * (1.0 - dt)
            w101 = dx * (1.0 - dy) * dt
            w110 = dx * dy * (1.0 - dt)
            w111 = dx * dy * dt

            weights = torch.stack(
                [w000, w001, w010, w011, w100, w101, w110, w111], dim=1
            )  # (N, 8, 1)

            # Interpolated feature vector for level l (N, F)
            interp_feat = torch.sum(features * weights, dim=1)  # (N, F)
            level_features.append(interp_feat)

        # Concatenate across all levels: (N, L * F)
        return torch.cat(level_features, dim=-1)
