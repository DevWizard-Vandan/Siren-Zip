"""Continuous Multi-Chunk Streaming Engine with Audio Master Clock & Double-Buffered Prefetching."""

from __future__ import annotations

import time
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch

from src.codec.residual_codec import ResidualCodec
from src.color.hdr_transfer import pq_to_linear, rec2020_to_rec709
from src.color.tone_mapper import apply_tone_mapping
from src.container.neura_v2_format import ChunkIndexRecord
from src.container.neura_v2_reader import NeuraV2Reader
from src.player.engine import ViewportBounds
from src.streaming.prefetcher import ChunkPrefetcher


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
    tone_map_mode: str
    is_hdr_source: bool
    width: int
    height: int
    is_prefetched: bool


class StreamEngine:
    """Runtime streaming engine for seamless multi-chunk continuous cinema playback."""

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

        # Lossless Residual Codec & Chunk Cache
        self.residual_codec = ResidualCodec()
        self.active_residuals: Optional[np.ndarray] = None

        # Double-Buffered CUDA Prefetcher
        self.prefetcher = ChunkPrefetcher(self.reader, device=self.device)

        # Color & HDR properties
        self.is_hdr = bool(self.header.transfer_characteristics in (16, 18) or self.header.color_primaries == 9)

        # Enable TF32 for fast Tensor Core forward evaluation
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def page_chunk(self, chunk_idx: int) -> Tuple[float, bool]:
        """Page weights and residual stream for chunk_idx in sub-millisecond time."""
        if chunk_idx == self.active_chunk_idx:
            return 0.0, True

        start_paging = time.perf_counter()
        state_dict, residual_bytes, was_prefetched = self.prefetcher.get_chunk_state_dict(chunk_idx)
        self.model.load_state_dict(state_dict, strict=False)
        self.active_chunk_idx = chunk_idx

        if residual_bytes:
            self.active_residuals = self.residual_codec.decode_chunk_residuals(residual_bytes)
        else:
            self.active_residuals = None

        paging_ms = (time.perf_counter() - start_paging) * 1000.0
        return paging_ms, was_prefetched

    @torch.no_grad()
    def render_at_time(
        self,
        t_global: float,
        viewport: ViewportBounds = ViewportBounds(),
        render_width: int = 640,
        render_height: int = 360,
        tone_map_mode: str = "aces",
        exposure: float = 1.0,
        lod_fast: bool = False,
        chunk_size: int = 262144,
    ) -> StreamRenderResult:
        """Evaluate continuous field driven by Master Clock with prefetching and tone-mapping."""
        start_total = time.perf_counter()

        # 1. Locate chunk and compute local coordinate t_local in [-1.0, 1.0]
        chunk_idx, record, t_local = self.reader.locate_chunk_and_local_time(t_global)

        # 2. Check lookahead and asynchronously prefetch next chunk
        self.prefetcher.check_and_prefetch(chunk_idx, t_local)

        # 3. Sub-millisecond weight paging
        paging_ms, was_prefetched = self.page_chunk(chunk_idx)

        # 4. Dynamic Level of Detail (LOD) for Real-Time 30-60 FPS Playback
        start_eval = time.perf_counter()
        if lod_fast:
            eff_w = min(480, max(64, render_width // 2))
            eff_h = min(270, max(36, render_height // 2))
        else:
            eff_w = max(64, render_width)
            eff_h = max(36, render_height)

        from src.model.perceptual_nerv import PerceptualNeRVVideo

        if isinstance(self.model, PerceptualNeRVVideo):
            # Instantaneous Single-Pass Full-Frame GPU Generation (<15ms)
            t_norm = torch.tensor([[(t_local + 1.0) * 0.5]], device=self.device, dtype=torch.float32)
            frame_tensor = self.model(t_norm)  # (1, 3, H, W)
            
            # Crop to viewport if zoomed
            if viewport.x_min != -1.0 or viewport.x_max != 1.0 or viewport.y_min != -1.0 or viewport.y_max != 1.0:
                H_f, W_f = frame_tensor.shape[-2], frame_tensor.shape[-1]
                x0 = int(round(max(0.0, (viewport.x_min + 1.0) * 0.5 * W_f)))
                x1 = int(round(min(float(W_f), (viewport.x_max + 1.0) * 0.5 * W_f)))
                y0 = int(round(max(0.0, (viewport.y_min + 1.0) * 0.5 * H_f)))
                y1 = int(round(min(float(H_f), (viewport.y_max + 1.0) * 0.5 * H_f)))
                frame_tensor = frame_tensor[:, :, y0:max(y0+1, y1), x0:max(x0+1, x1)]

            if (frame_tensor.shape[-2], frame_tensor.shape[-1]) != (eff_h, eff_w):
                frame_tensor = torch.nn.functional.interpolate(
                    frame_tensor, size=(eff_h, eff_w), mode="bilinear", align_corners=False
                )
            rgb_full = frame_tensor[0].permute(1, 2, 0)
        else:
            # 5. Generate coordinate grid within visible viewport
            y_coords = torch.linspace(viewport.y_min, viewport.y_max, steps=eff_h, device=self.device, dtype=torch.float32)
            x_coords = torch.linspace(viewport.x_min, viewport.x_max, steps=eff_w, device=self.device, dtype=torch.float32)

            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
            t_grid = torch.full_like(x_grid, fill_value=t_local, device=self.device, dtype=torch.float32)

            coords_flat = torch.stack([x_grid, y_grid, t_grid], dim=-1).reshape(-1, 3)
            total_eval_points = coords_flat.shape[0]

            # 6. Batch forward inference
            preds: list[torch.Tensor] = []
            for start in range(0, total_eval_points, chunk_size):
                end = min(start + chunk_size, total_eval_points)
                batch = coords_flat[start:end]
                preds.append(self.model(batch))

            rgb_full = torch.cat(preds, dim=0).reshape(eff_h, eff_w, 3)

        # 7. Apply Color Science & Tone Mapping if needed
        if self.header.transfer_characteristics == 16:  # ST.2084 PQ
            rgb_linear = pq_to_linear(rgb_full)
            if self.header.color_primaries == 9:  # Rec.2020
                rgb_linear = rec2020_to_rec709(rgb_linear)
            rgb_mapped = apply_tone_mapping(rgb_linear, mode=tone_map_mode, exposure=exposure)
        elif tone_map_mode != "linear":
            rgb_mapped = apply_tone_mapping(rgb_full, mode=tone_map_mode, exposure=exposure)
        else:
            rgb_mapped = rgb_full.clamp(0.0, 1.0)

        rgb_np = (rgb_mapped.cpu().numpy() * 255.0).astype(np.uint8)

        if lod_fast and (eff_w != render_width or eff_h != render_height):
            import cv2
            rgb_np = cv2.resize(rgb_np, (render_width, render_height), interpolation=cv2.INTER_LINEAR)

        # 8. Apply Lossless High-Frequency Residual Stream if present
        if self.active_residuals is not None:
            import cv2
            frame_idx = min(
                self.active_residuals.shape[0] - 1,
                max(0, int(round((t_local + 1.0) * 0.5 * (self.active_residuals.shape[0] - 1)))),
            )
            diff_frame = self.active_residuals[frame_idx]
            if diff_frame.shape[:2] != (rgb_np.shape[0], rgb_np.shape[1]):
                diff_frame = cv2.resize(diff_frame, (rgb_np.shape[1], rgb_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            rgb_np = np.clip(rgb_np.astype(np.int16) + diff_frame, 0, 255).astype(np.uint8)

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
            tone_map_mode=tone_map_mode,
            is_hdr_source=self.is_hdr,
            width=render_width,
            height=render_height,
            is_prefetched=was_prefetched,
        )

    def close(self) -> None:
        self.prefetcher.clear()
        self.reader.close()
