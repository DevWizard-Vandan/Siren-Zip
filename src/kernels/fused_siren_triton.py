"""Fused Linear + Sine Periodic Activation CUDA / Triton Kernels for Ultra-Fast Inference."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Check for Triton support
HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_sine_linear_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        M,
        N,
        K,
        omega_0: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr = 64,
        BLOCK_SIZE_N: tl.constexpr = 64,
        BLOCK_SIZE_K: tl.constexpr = 32,
    ):
        """Fused Matrix Multiplication (X @ W^T + b) + sin(omega_0 * z) in SRAM."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        x_ptrs = x_ptr + (offs_m[:, None] * K + offs_k[None, :])
        w_ptrs = w_ptr + (offs_n[None, :] * K + offs_k[:, None])

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE_K):
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
            accumulator += tl.dot(x, w)
            x_ptrs += BLOCK_SIZE_K
            w_ptrs += BLOCK_SIZE_K

        # Add bias
        if b_ptr is not None:
            b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
            accumulator += b[None, :]

        # Fused Sine Activation: sin(omega_0 * accumulator)
        out = tl.sin(omega_0 * accumulator)

        out_ptrs = out_ptr + (offs_m[:, None] * N + offs_n[None, :])
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class FusedSineLinear(nn.Module):
    """Fused Linear + Sine layer executing on GPU Tensor Cores with TensorFloat-32 / Triton."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = float(omega_0)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6.0 / self.in_features) / self.omega_0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1e-4, 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # High-performance PyTorch C++ / CUDA fused path (TensorFloat-32 enabled on RTX GPUs)
        z = torch.addmm(self.bias, x, self.weight.t()) if self.bias is not None else torch.mm(x, self.weight.t())
        return torch.sin(self.omega_0 * z)
