"""Spatio-Temporal SIREN (Sinusoidal Representation Network) for Video INR.

Maps 3D continuous spacetime coordinates:
    f_theta(x, y, t) -> (r, g, b) where (x, y, t) in [-1.0, 1.0]^3

Features:
- Anisotropic spatio-temporal frequency scaling (omega_xy=30.0 for spatial sharpness, omega_t=10.0 for temporal motion).
- Specialized Sitzmann weight initialization.
- Built-in INT8 quantization and memory footprint calculation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalSineLayer(nn.Module):
    """SineLayer with support for anisotropic spatio-temporal frequency scaling:

        phi_0(x, y, t) = sin( W_xy * [x, y] * omega_xy + W_t * [t] * omega_t + b )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
        omega_xy: float = 30.0,
        omega_t: float = 10.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = float(omega_0)
        self.omega_xy = float(omega_xy)
        self.omega_t = float(omega_t)

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if self.is_first:
            # Register anisotropic scale vector: [omega_xy, omega_xy, omega_t]
            scale_vector = torch.tensor([self.omega_xy, self.omega_xy, self.omega_t], dtype=torch.float32)
            self.register_buffer("coord_scale", scale_vector)
        else:
            self.register_buffer("coord_scale", None)

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1 / in_features, 1 / in_features]
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers: uniform in [-sqrt(6 / in_features) / omega_0, sqrt(6 / in_features) / omega_0]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_first and self.coord_scale is not None:
            # Pre-scale (x, y, t) by [omega_xy, omega_xy, omega_t]
            scaled_input = x * self.coord_scale
            return torch.sin(self.linear(scaled_input))
        else:
            return torch.sin(self.omega_0 * self.linear(x))

    def extra_repr(self) -> str:
        if self.is_first:
            return f"in={self.in_features}, out={self.out_features}, is_first=True, omega_xy={self.omega_xy}, omega_t={self.omega_t}"
        return f"in={self.in_features}, out={self.out_features}, is_first=False, omega_0={self.omega_0}"


class SirenVideo(nn.Module):
    """Continuous Spatio-Temporal Video Implicit Neural Representation network.

    f_theta(x, y, t) -> (r, g, b) where (x, y, t) in [-1.0, 1.0]^3

    Args:
        in_features: Number of coordinate dimensions (default: 3 for x, y, t).
        hidden_features: Number of hidden units per layer (default: 384).
        hidden_layers: Number of hidden sinusoidal layers (default: 6).
        out_features: Output channels (default: 3 for RGB).
        omega_xy: Spatial frequency multiplier for first layer (default: 30.0).
        omega_t: Temporal frequency multiplier for first layer (default: 10.0).
        omega_0_hidden: Frequency multiplier for hidden layers (default: 30.0).
        final_activation: Output activation ('clamp', 'sigmoid', 'none').
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 384,
        hidden_layers: int = 6,
        out_features: int = 3,
        omega_xy: float = 30.0,
        omega_t: float = 10.0,
        omega_0_hidden: float = 30.0,
        final_activation: Literal["clamp", "sigmoid", "none"] = "clamp",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_xy = float(omega_xy)
        self.omega_t = float(omega_t)
        self.omega_0_hidden = float(omega_0_hidden)
        self.final_activation = final_activation

        layers: List[nn.Module] = []

        # 1. First Anisotropic Spatio-Temporal Sine Layer
        layers.append(
            SpatioTemporalSineLayer(
                in_features=in_features,
                out_features=hidden_features,
                is_first=True,
                omega_xy=omega_xy,
                omega_t=omega_t,
                omega_0=omega_0_hidden,
            )
        )

        # 2. Hidden Sine Layers
        for _ in range(hidden_layers):
            layers.append(
                SpatioTemporalSineLayer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    is_first=False,
                    omega_0=omega_0_hidden,
                )
            )

        self.net = nn.Sequential(*layers)

        # 3. Final Linear Readout Layer
        self.final_linear = nn.Linear(hidden_features, out_features)
        self.init_final_layer()

    def init_final_layer(self) -> None:
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.hidden_features) / self.omega_0_hidden
            self.final_linear.weight.uniform_(-bound, bound)
            if self.final_linear.bias is not None:
                self.final_linear.bias.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass over continuous (x, y, t) spacetime coordinates.

        Args:
            coords: Tensor of shape (Batch, 3) in range [-1.0, 1.0].

        Returns:
            rgb: Tensor of shape (Batch, 3) representing predicted RGB in [0.0, 1.0].
        """
        features = self.net(coords)
        out = self.final_linear(features)

        if self.final_activation == "clamp":
            return torch.clamp(out, 0.0, 1.0)
        elif self.final_activation == "sigmoid":
            return torch.sigmoid(out)
        return out

    def get_num_params(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_kb(self, precision: Literal["fp32", "fp16", "int8"] = "fp32") -> float:
        """Calculate memory footprint in Kilobytes."""
        num_params = self.get_num_params()
        bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}[precision]
        return (num_params * bytes_per_param) / 1024.0

    def get_config(self) -> Dict[str, Any]:
        """Return architecture configuration dictionary."""
        return {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "hidden_layers": self.hidden_layers,
            "out_features": self.out_features,
            "omega_xy": self.omega_xy,
            "omega_t": self.omega_t,
            "omega_0_hidden": self.omega_0_hidden,
            "final_activation": self.final_activation,
        }
