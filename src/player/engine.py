"""High-Speed GPU Inference Engine with Dynamic Spatio-Temporal Viewport Culling."""

from __future__ import annotations

import time
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.model.siren_video import SirenVideo


class ViewportBounds(NamedTuple):
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0


class RenderRequest(NamedTuple):
    t_val: float
    viewport: ViewportBounds
    canvas_width: int
    canvas_height: int
    lod_fast: bool = False


class RenderResult(NamedTuple):
    rgb_numpy: np.ndarray  # uint8 RGB array (H, W, 3)
    t_val: float
    zoom_factor: float
    compute_time_ms: float
    culling_ratio_pct: float
    evaluated_pixels: int
    width: int
    height: int


class PlayerEngine:
    """GPU-accelerated continuous inference engine with Dynamic Viewport Culling & LOD."""

    def __init__(
        self,
        model: SirenVideo,
        meta: Dict[str, Any],
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.meta = meta

        self.native_width = int(meta.get("width", 1280))
        self.native_height = int(meta.get("height", 720))
        self.native_fps = float(meta.get("fps", 24.0))
        self.frame_count = int(meta.get("frame_count", 96))

        # Enable TF32 for fast Tensor Core evaluation
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    @torch.no_grad()
    def render_viewport(
        self,
        t_val: float,
        viewport: ViewportBounds = ViewportBounds(),
        render_width: int = 640,
        render_height: int = 360,
        lod_fast: bool = False,
        chunk_size: int = 262144,
    ) -> RenderResult:
        """Evaluate continuous Spatio-Temporal SIREN strictly within the visible viewport bounds.

        Args:
            t_val: Continuous normalized timestamp in [-1.0, 1.0].
            viewport: ViewportBounds (x_min, x_max, y_min, y_max).
            render_width: Target canvas width.
            render_height: Target canvas height.
            lod_fast: If True, uses 0.5x resolution for interactive low-latency scrubbing.
            chunk_size: Sub-batch evaluation chunk size.

        Returns:
            RenderResult containing uint8 image and performance metrics.
        """
        start_time = time.perf_counter()

        # Dynamic Level of Detail (LOD)
        eff_w = max(64, render_width // 2) if lod_fast else max(64, render_width)
        eff_h = max(36, render_height // 2) if lod_fast else max(36, render_height)

        # 1. Generate 3D continuous coordinate grid within [x_min, x_max] x [y_min, y_max] x [t_val]
        y_coords = torch.linspace(viewport.y_min, viewport.y_max, steps=eff_h, device=self.device, dtype=torch.float32)
        x_coords = torch.linspace(viewport.x_min, viewport.x_max, steps=eff_w, device=self.device, dtype=torch.float32)

        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        t_grid = torch.full_like(x_grid, fill_value=t_val, device=self.device, dtype=torch.float32)

        coords_flat = torch.stack([x_grid, y_grid, t_grid], dim=-1).reshape(-1, 3)
        total_eval_points = coords_flat.shape[0]

        # 2. Batch forward inference through SIREN
        preds: list[torch.Tensor] = []
        for start in range(0, total_eval_points, chunk_size):
            end = min(start + chunk_size, total_eval_points)
            batch = coords_flat[start:end]
            rgb_pred = self.model(batch)
            preds.append(rgb_pred)

        rgb_full = torch.cat(preds, dim=0).reshape(eff_h, eff_w, 3)

        # 3. Transfer to CPU uint8 array
        rgb_np = (rgb_full.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)

        # If LOD was applied, scale back up to requested render size
        if lod_fast and (eff_w != render_width or eff_h != render_height):
            import cv2
            rgb_np = cv2.resize(rgb_np, (render_width, render_height), interpolation=cv2.INTER_LINEAR)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Calculate Zoom & Dynamic Viewport Culling Savings
        dx = max(1e-6, viewport.x_max - viewport.x_min)
        dy = max(1e-6, viewport.y_max - viewport.y_min)
        zoom_x = 2.0 / dx
        zoom_y = 2.0 / dy
        zoom_factor = float(max(zoom_x, zoom_y))

        viewport_area_fraction = min(1.0, (dx * dy) / 4.0)
        culling_ratio_pct = max(0.0, (1.0 - viewport_area_fraction) * 100.0)

        return RenderResult(
            rgb_numpy=rgb_np,
            t_val=t_val,
            zoom_factor=zoom_factor,
            compute_time_ms=elapsed_ms,
            culling_ratio_pct=culling_ratio_pct,
            evaluated_pixels=total_eval_points,
            width=render_width,
            height=render_height,
        )
