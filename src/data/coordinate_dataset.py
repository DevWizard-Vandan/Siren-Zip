"""Continuous 2D coordinate dataset and GPU-accelerated coordinate sampler for SIREN."""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def make_coordinate_grid(
    height: int,
    width: int,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Generate a dense 2D coordinate grid in normalized space.

    Args:
        height: Number of vertical coordinate steps (rows).
        width: Number of horizontal coordinate steps (columns).
        x_range: Bounds for horizontal coordinate (x_min, x_max).
        y_range: Bounds for vertical coordinate (y_min, y_max).
        device: Target torch device.

    Returns:
        coords: Tensor of shape (height * width, 2) where coords[:, 0] = x, coords[:, 1] = y.
    """
    y_coords = torch.linspace(y_range[0], y_range[1], steps=height, device=device, dtype=torch.float32)
    x_coords = torch.linspace(x_range[0], x_range[1], steps=width, device=device, dtype=torch.float32)

    # meshgrid with indexing='ij' -> y_grid: (H, W), x_grid: (H, W)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")

    # Stack into (H, W, 2) as (x, y) coordinates
    coords = torch.stack([x_grid, y_grid], dim=-1)

    # Reshape to (H * W, 2)
    return coords.reshape(-1, 2)


def load_image_as_tensor(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, int, int]:
    """Load an image file and convert to normalized float32 RGB tensor in [0.0, 1.0].

    Args:
        image_path: Path to image file (.png, .jpg, etc.).
        target_size: Optional (height, width) to resize.

    Returns:
        rgb_tensor: Tensor of shape (H, W, 3) in [0.0, 1.0].
        height: Image height.
        width: Image width.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img = Image.open(image_path).convert("RGB")
    if target_size is not None:
        # PIL resize takes (width, height)
        img = img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)

    img_np = np.array(img, dtype=np.float32) / 255.0
    rgb_tensor = torch.from_numpy(img_np)
    height, width, _ = rgb_tensor.shape
    return rgb_tensor, height, width


class ImageCoordinateData:
    """High-throughput in-VRAM coordinate container for continuous image training."""

    def __init__(
        self,
        image_path: str,
        target_size: Optional[Tuple[int, int]] = None,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        rgb_tensor, self.height, self.width = load_image_as_tensor(image_path, target_size)

        self.num_pixels = self.height * self.width
        self.rgb_grid = rgb_tensor.to(self.device)  # (H, W, 3)
        self.rgb_flat = self.rgb_grid.reshape(-1, 3)  # (N, 3)

        self.coords_flat = make_coordinate_grid(
            height=self.height,
            width=self.width,
            device=self.device,
        )  # (N, 2)

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast GPU-resident random point batch sampling.

        Args:
            batch_size: Number of coordinates to sample.

        Returns:
            batch_coords: Tensor of shape (batch_size, 2).
            batch_rgb: Tensor of shape (batch_size, 3).
        """
        indices = torch.randint(0, self.num_pixels, (batch_size,), device=self.device)
        return self.coords_flat[indices], self.rgb_flat[indices]


class CoordinateDataset(Dataset):
    """Standard PyTorch Dataset for coordinate-based image training."""

    def __init__(
        self,
        image_path: str,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        rgb_tensor, self.height, self.width = load_image_as_tensor(image_path, target_size)
        self.num_pixels = self.height * self.width
        self.rgb_flat = rgb_tensor.reshape(-1, 3)
        self.coords_flat = make_coordinate_grid(
            height=self.height,
            width=self.width,
            device="cpu",
        )

    def __len__(self) -> int:
        return self.num_pixels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords_flat[idx], self.rgb_flat[idx]
