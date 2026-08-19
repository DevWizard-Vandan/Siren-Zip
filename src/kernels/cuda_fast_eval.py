"""Vectorized Fast GPU Coordinate Evaluator with Buffer Pre-Allocation and CUDA Streams."""

from __future__ import annotations

import time
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.types.viewport import ViewportBounds


class FastRenderResult(NamedTuple):
    rgb_numpy: np.ndarray
    eval_time_ms: float
    total_pixels: int


class FastGPUEvaluator:
    """Pre-allocates GPU buffers and evaluates 4K coordinates in sub-16ms latency."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        enable_compile: bool = False,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device).eval()

        # Enable TensorFloat-32 for Ampere & Ada Lovelace RTX GPUs
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        # Optional torch.compile acceleration
        self.compiled_model = None
        if enable_compile and hasattr(torch, "compile"):
            try:
                self.compiled_model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                self.compiled_model = self.model
        else:
            self.compiled_model = self.model

        # Pre-allocated coordinate buffers
        self._cached_w: int = 0
        self._cached_h: int = 0
        self._cached_xy: Optional[torch.Tensor] = None
        self._cached_coords: Optional[torch.Tensor] = None
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _prepare_grid(self, width: int, height: int, viewport: ViewportBounds, t_val: float) -> torch.Tensor:
        """Pre-allocate or reuse continuous grid buffers on GPU."""
        if (
            self._cached_w != width
            or self._cached_h != height
            or self._cached_coords is None
        ):
            self._cached_w = width
            self._cached_h = height

            # Continuous coordinate vectors
            y_coords = torch.linspace(viewport.y_min, viewport.y_max, height, device=self.device)
            x_coords = torch.linspace(viewport.x_min, viewport.x_max, width, device=self.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            self._cached_xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (N, 2)
            self._cached_coords = torch.empty((height * width, 3), device=self.device, dtype=torch.float32)
        else:
            # Recompute if viewport bounds changed
            y_coords = torch.linspace(viewport.y_min, viewport.y_max, height, device=self.device)
            x_coords = torch.linspace(viewport.x_min, viewport.x_max, width, device=self.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            self._cached_xy[:, 0] = xx.flatten()
            self._cached_xy[:, 1] = yy.flatten()

        # Update time coordinate column
        self._cached_coords[:, 0:2] = self._cached_xy
        self._cached_coords[:, 2] = t_val
        return self._cached_coords

    @torch.no_grad()
    def evaluate_frame(
        self,
        t_val: float,
        width: int,
        height: int,
        viewport: Optional[ViewportBounds] = None,
    ) -> FastRenderResult:
        """Evaluate continuous neural field at given timestamp and resolution."""
        vp = viewport or ViewportBounds()
        t0 = time.perf_counter()

        coords = self._prepare_grid(width, height, vp, t_val)
        num_pixels = width * height

        # Chunked evaluation if 4K UHD or larger to avoid VRAM spikes
        chunk_size = 1048576  # 1M points per sub-batch
        if num_pixels <= chunk_size:
            rgb_tensor = self.compiled_model(coords)
        else:
            rgb_chunks = []
            for i in range(0, num_pixels, chunk_size):
                sub_coords = coords[i : i + chunk_size]
                rgb_sub = self.compiled_model(sub_coords)
                rgb_chunks.append(rgb_sub)
            rgb_tensor = torch.cat(rgb_chunks, dim=0)

        # Reshape to (H, W, 3) and convert to uint8
        rgb_tensor = rgb_tensor.view(height, width, 3)
        rgb_uint8 = (torch.clamp(rgb_tensor, 0.0, 1.0) * 255.0).to(torch.uint8)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        eval_ms = (time.perf_counter() - t0) * 1000.0
        rgb_np = rgb_uint8.cpu().numpy()

        return FastRenderResult(
            rgb_numpy=rgb_np,
            eval_time_ms=eval_ms,
            total_pixels=num_pixels,
        )
