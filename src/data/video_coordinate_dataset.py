"""GPU-Accelerated Video Coordinate Dataset and Sampler for Spatio-Temporal INR."""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_video_tensor(
    video_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, float, int, int, int]:
    """Load video frames into a normalized float32 RGB tensor in [0.0, 1.0].

    Args:
        video_path: Path to input video (.mp4, .avi, etc.).
        target_size: Optional (height, width) to resize frames.
        max_frames: Optional cap on total frames loaded.

    Returns:
        video_tensor: (T, H, W, 3) float32 tensor in range [0.0, 1.0].
        fps: Original frame rate.
        num_frames: Total number of frames.
        height: Frame height.
        width: Frame width.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 24.0

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if target_size is not None:
            target_h, target_w = target_size
            frame_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

        frames.append(frame_rgb)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames could be extracted from: {video_path}")

    frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
    video_tensor = torch.from_numpy(frames_np)  # (T, H, W, 3)
    num_frames, height, width, _ = video_tensor.shape

    return video_tensor, fps, num_frames, height, width


def make_frame_coordinate_grid(
    t_val: float,
    height: int,
    width: int,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Generate 3D coordinates (x, y, t) for a single frame at timestamp t_val.

    Returns:
        coords: Tensor of shape (height * width, 3) where [:, 0]=x, [:, 1]=y, [:, 2]=t.
    """
    y_coords = torch.linspace(y_range[0], y_range[1], steps=height, device=device, dtype=torch.float32)
    x_coords = torch.linspace(x_range[0], x_range[1], steps=width, device=device, dtype=torch.float32)

    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    t_grid = torch.full_like(x_grid, fill_value=t_val, device=device, dtype=torch.float32)

    coords = torch.stack([x_grid, y_grid, t_grid], dim=-1)
    return coords.reshape(-1, 3)


class VideoCoordinateData:
    """Ultra-fast GPU spatio-temporal coordinate container using contiguous 1D memory layout."""

    def __init__(
        self,
        video_path: str,
        target_size: Optional[Tuple[int, int]] = None,
        max_frames: Optional[int] = None,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        video_tensor, self.fps, self.num_frames, self.height, self.width = load_video_tensor(
            video_path=video_path,
            target_size=target_size,
            max_frames=max_frames,
        )

        self.total_voxels = self.num_frames * self.height * self.width
        self.video_tensor = video_tensor.to(self.device)  # (T, H, W, 3)
        self.flat_rgb = self.video_tensor.reshape(-1, 3).contiguous()  # (total_voxels, 3)

        # Precompute constants for instant GPU coordinate reconstruction
        self.hw = self.height * self.width
        self.w_denom = float(max(1, self.width - 1))
        self.h_denom = float(max(1, self.height - 1))
        self.t_denom = float(max(1, self.num_frames - 1))

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random spatio-temporal points (x, y, t) in single contiguous 1D gather (<0.2ms).

        Returns:
            coords: Tensor of shape (batch_size, 3) -> [x, y, t] in [-1.0, 1.0]^3.
            rgb: Tensor of shape (batch_size, 3) -> [r, g, b] in [0.0, 1.0].
        """
        # 1. Sample 1D flat voxel indices uniformly on GPU
        flat_idx = torch.randint(0, self.total_voxels, (batch_size,), device=self.device)

        # 2. Extract ground truth RGB via fast 1D contiguous indexing
        batch_rgb = self.flat_rgb[flat_idx]

        # 3. Vectorized arithmetic to compute (x, y, t) in [-1.0, 1.0]
        idx_t = flat_idx // self.hw
        rem_hw = flat_idx % self.hw
        idx_y = rem_hw // self.width
        idx_x = rem_hw % self.width

        x = -1.0 + 2.0 * (idx_x.float() / self.w_denom)
        y = -1.0 + 2.0 * (idx_y.float() / self.h_denom)
        t = -1.0 + 2.0 * (idx_t.float() / self.t_denom)

        batch_coords = torch.stack([x, y, t], dim=-1)

        return batch_coords, batch_rgb

    def get_frame_timestamp(self, frame_idx: int) -> float:
        """Return normalized timestamp t in [-1.0, 1.0] for a given frame index."""
        if self.num_frames <= 1:
            return 0.0
        return float(-1.0 + 2.0 * frame_idx / (self.num_frames - 1))
