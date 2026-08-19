"""Pure SIREN (Sinusoidal Representation Networks) implementation in PyTorch.

Reference:
    Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions",
    NeurIPS 2020. https://arxiv.org/abs/2006.09661
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SineLayer(nn.Module):
    """A linear layer followed by a periodic sinusoidal activation function:

        phi(x) = sin(omega_0 * (W x + b))

    With specialized Sitzmann initialization preserving distribution across layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = float(omega_0)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
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
                # Initialize bias to uniform or zeros
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, is_first={self.is_first}, omega_0={self.omega_0}"


class SirenImage(nn.Module):
    """Implicit Neural Representation (INR) network mapping 2D continuous coordinates

    f_theta(x, y) -> (r, g, b) where (x, y) in [-1.0, 1.0]^2

    Args:
        in_features: Number of input coordinate dimensions (default: 2 for x, y).
        hidden_features: Number of hidden units per layer (default: 256).
        hidden_layers: Number of hidden sinusoidal layers (default: 5).
        out_features: Number of output channels (default: 3 for RGB).
        omega_0: Angular frequency multiplier for the first layer (default: 30.0).
        omega_0_hidden: Angular frequency multiplier for hidden layers (default: 30.0).
        final_activation: Activation applied to the output ('sigmoid', 'clamp', 'none').
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        out_features: int = 3,
        omega_0: float = 30.0,
        omega_0_hidden: float = 30.0,
        final_activation: Literal["sigmoid", "clamp", "none"] = "sigmoid",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_0 = omega_0
        self.omega_0_hidden = omega_0_hidden
        self.final_activation = final_activation

        layers: List[nn.Module] = []

        # 1. First Sine Layer
        layers.append(
            SineLayer(
                in_features=in_features,
                out_features=hidden_features,
                is_first=True,
                omega_0=omega_0,
            )
        )

        # 2. Hidden Sine Layers
        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
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
        """Forward evaluation of continuous coordinates.

        Args:
            coords: Tensor of shape (Batch, 2) in range [-1.0, 1.0].

        Returns:
            rgb: Tensor of shape (Batch, 3) representing predicted RGB in [0.0, 1.0].
        """
        features = self.net(coords)
        out = self.final_linear(features)

        if self.final_activation == "sigmoid":
            return torch.sigmoid(out)
        elif self.final_activation == "clamp":
            return torch.clamp(out, 0.0, 1.0)
        return out

    def get_num_params(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_kb(self, precision: Literal["fp32", "fp16", "int8"] = "fp32") -> float:
        """Calculate uncompressed memory footprint in Kilobytes."""
        num_params = self.get_num_params()
        bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1}[precision]
        return (num_params * bytes_per_param) / 1024.0

    def quantize_to_int8(self) -> Dict[str, Tuple[torch.Tensor, float, int]]:
        """Quantize model weights to INT8 using symmetric min-max quantization.

        Returns a dictionary of quantized integer tensors and scale factors.
        """
        quantized_state: Dict[str, Tuple[torch.Tensor, float, int]] = {}
        for name, param in self.named_parameters():
            tensor = param.data
            max_val = torch.max(torch.abs(tensor)).item()
            scale = max_val / 127.0 if max_val > 0 else 1.0
            q_tensor = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
            quantized_state[name] = (q_tensor, scale, 0)
        return quantized_state
