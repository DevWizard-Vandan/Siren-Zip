"""Continuous Multi-Chunk Streaming Engine with Sub-Millisecond Weight Paging."""

from __future__ import annotations

import time
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

from src.container.neura_v2_format import ChunkIndexRecord
from src.container.neura_v2_reader import NeuraV2Reader
from src.player.engine import ViewportBounds


class StreamRenderResult(NamedTuple):
    rgb_numpy: np.ndarray  # uint8 RGB array (H, W, 3)
    t_global: float
    t_local: float
    chunk_idx: int
    total_chunks: int
    paging_time_ms: float
    eval_time_ms: float
    total_latency_ms: float
    zoom_factor: float
    culling_saved_pct: float
    width: int
    height: int


class StreamEngine:
    """Runtime streaming engine for seamless multi-chunk continuous playback."""

    def __init__(
        self,
        neura_path: str,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.reader = NeuraV2Reader(neura_path)
        self.header = self.reader.header

        # Preallocate single GPU model shell
        self.model = self.reader.create_model_shell(device=self.device)
        self.active_chunk_idx: int = -1

        # Performance Cache
        self.chunk_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.max_cached_chunks = 8  # Keep up to 8 recent chunks in memory for instant bi-directional scrubbing

        # Enable TF32 for fast Tensor Core forward evaluation
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def page_chunk(self, chunk_idx: int) -> float:
        """Page weights for chunk_idx into GPU model shell in sub-millisecond time."""
        if chunk_idx == self.active_chunk_idx:
            return 0.0

        start_paging = time.perf_counter()

        if chunk_idx in self.chunk_cache:
            state_dict = self.chunk_cache[chunk_idx]
        else:
            state_dict = self.reader.load_chunk_state_dict(chunk_idx, device=self.device)
            if len(self.chunk_cache) >= self.max_cached_chunks:
                # Remove oldest cached chunk
                first_key = next(iter(self.chunk_cache))
                del self.chunk_cache[first_key]
            self.chunk_cache[chunk_idx] = state_dict

        self.model.load_state_dict(state_dict, strict=False)
        self.active_chunk_idx = chunk_idx

        paging_ms = (time.perf_counter() - start_paging) * 1000.0
        return paging_ms

    @torch.no_grad()
    def render_at_time(
        self,
        t_global: float,
        viewport: ViewportBounds = ViewportBounds(),
        render_width: int = 640,
        render_height: int = 360,
        lod_fast: bool = False,
        chunk_size: int = 262144,
    ) -> StreamRenderResult:
        """Evaluate continuous spatio-temporal field at global movie timestamp t_global.

        Args:
            t_global: Global timestamp in seconds [0.0, total_duration].
            viewport: ViewportBounds (x_min, x_max, y_min, y_max).
            render_width: Viewport width.
            render_height: Viewport height.
            lod_fast: If True, uses 0.5x resolution for ultra-low latency scrubbing.
            chunk_size: Coordinate batch size.

        Returns:
            StreamRenderResult with frame buffer and latency diagnostics.
        """
        start_total = time.perf_counter()

        # 1. Locate chunk and compute local coordinate t_local in [-1.0, 1.0]
        chunk_idx, record, t_local = self.reader.locate_chunk_and_local_time(t_global)

        # 2. Sub-millisecond weight paging
        paging_ms = self.page_chunk(chunk_idx)

        # 3. Dynamic Level of Detail (LOD)
        start_eval = time.perf_counter()
        eff_w = max(64, render_width // 2) if lod_fast else max(64, render_width)
        eff_h = max(36, render_height // 2) if lod_fast else max(36, render_height)

        # 4. Generate coordinate grid within [x_min, x_max] x [y_min, y_max] at local time t_local
        y_coords = torch.linspace(viewport.y_min, viewport.y_max, steps=eff_h, device=self.device, dtype=torch.float32)
        x_coords = torch.linspace(viewport.x_min, viewport.x_max, steps=eff_w, device=self.device, dtype=torch.float32)

        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        t_grid = torch.full_like(x_grid, fill_value=t_local, device=self.device, dtype=torch.float32)

        coords_flat = torch.stack([x_grid, y_grid, t_grid], dim=-1).reshape(-1, 3)
        total_eval_points = coords_flat.shape[0]

        # 5. Batch forward inference
        preds: list[torch.Tensor] = []
        for start in range(0, total_eval_points, chunk_size):
            end = min(start + chunk_size, total_eval_points)
            batch = coords_flat[start:end]
            preds.append(self.model(batch))

        rgb_full = torch.cat(preds, dim=0).reshape(eff_h, eff_w, 3)
        rgb_np = (rgb_full.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)

        if lod_fast and (eff_w != render_width or eff_h != render_height):
            import cv2
            rgb_np = cv2.resize(rgb_np, (render_width, render_height), interpolation=cv2.INTER_LINEAR)

        eval_ms = (time.perf_counter() - start_eval) * 1000.0
        total_latency_ms = (time.perf_counter() - start_total) * 1000.0

        # Viewport zoom & culling ratio
        dx = max(1e-6, viewport.x_max - viewport.x_min)
        dy = max(1e-6, viewport.y_max - viewport.y_min)
        zoom_factor = float(max(2.0 / dx, 2.0 / dy))
        viewport_area_fraction = min(1.0, (dx * dy) / 4.0)
        culling_saved_pct = max(0.0, (1.0 - viewport_area_fraction) * 100.0)

        return StreamRenderResult(
            rgb_numpy=rgb_np,
            t_global=t_global,
            t_local=t_local,
            chunk_idx=chunk_idx,
            total_chunks=self.header.total_chunks,
            paging_time_ms=paging_ms,
            eval_time_ms=eval_ms,
            total_latency_ms=total_latency_ms,
            zoom_factor=zoom_factor,
            culling_saved_pct=culling_saved_pct,
            width=render_width,
            height=render_height,
        )

    def close(self) -> None:
        self.reader.close()
