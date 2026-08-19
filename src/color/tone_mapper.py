"""Film-Industry Standard Tone-Mapping Algorithms (ACES Filmic, Reinhard, Reinhard-Jodie)."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


def tone_map_aces_filmic(
    color: Union[torch.Tensor, np.ndarray],
    exposure: float = 1.0,
) -> Union[torch.Tensor, np.ndarray]:
    """ACES Filmic tone-mapping curve (Narkowicz / Unreal Engine 4 formulation).

    Formula:
        ACES(x) = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)

    Preserves high-luminance specular highlights while gracefully retaining shadow contrast.
    """
    if isinstance(color, torch.Tensor):
        x = torch.clamp(color * exposure, min=0.0)
        numerator = x * (2.51 * x + 0.03)
        denominator = x * (2.43 * x + 0.59) + 0.14
        mapped = torch.clamp(numerator / denominator, 0.0, 1.0)
        return mapped
    else:
        x = np.maximum(color * exposure, 0.0)
        numerator = x * (2.51 * x + 0.03)
        denominator = x * (2.43 * x + 0.59) + 0.14
        mapped = np.clip(numerator / denominator, 0.0, 1.0)
        return mapped


def tone_map_reinhard(
    color: Union[torch.Tensor, np.ndarray],
    exposure: float = 1.0,
) -> Union[torch.Tensor, np.ndarray]:
    """Standard Reinhard Tone-Mapping: x / (1 + x)."""
    if isinstance(color, torch.Tensor):
        x = torch.clamp(color * exposure, min=0.0)
        return torch.clamp(x / (1.0 + x), 0.0, 1.0)
    else:
        x = np.maximum(color * exposure, 0.0)
        return np.clip(x / (1.0 + x), 0.0, 1.0)


def tone_map_reinhard_jodie(
    color: Union[torch.Tensor, np.ndarray],
    exposure: float = 1.0,
) -> Union[torch.Tensor, np.ndarray]:
    """Reinhard-Jodie: Color-preserving luminance tone mapper.

    Prevents hue shifts and saturation blowout in bright highlights.
    """
    if isinstance(color, torch.Tensor):
        x = torch.clamp(color * exposure, min=0.0)
        # Rec.709 relative luminance
        lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        lum = lum.unsqueeze(-1)
        reinhard_rgb = x / (1.0 + x)
        reinhard_lum = x / (1.0 + lum)
        mapped = torch.lerp(reinhard_rgb, reinhard_lum, x / torch.clamp(x + 1.0, min=1e-6))
        return torch.clamp(mapped, 0.0, 1.0)
    else:
        x = np.maximum(color * exposure, 0.0)
        lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        lum = np.expand_dims(lum, axis=-1)
        reinhard_rgb = x / (1.0 + x)
        reinhard_lum = x / (1.0 + lum)
        mapped = reinhard_rgb + (reinhard_lum - reinhard_rgb) * (x / np.maximum(x + 1.0, 1e-6))
        return np.clip(mapped, 0.0, 1.0)


def tone_map_linear_clamp(
    color: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """Standard SDR linear clamp in [0.0, 1.0]."""
    if isinstance(color, torch.Tensor):
        return torch.clamp(color, 0.0, 1.0)
    else:
        return np.clip(color, 0.0, 1.0)


def apply_tone_mapping(
    image: Union[torch.Tensor, np.ndarray],
    mode: str = "aces",
    exposure: float = 1.0,
) -> Union[torch.Tensor, np.ndarray]:
    """Universal dispatcher for display tone-mapping curves."""
    m = mode.lower()
    if m in ("aces", "aces_filmic", "filmic"):
        return tone_map_aces_filmic(image, exposure=exposure)
    elif m in ("reinhard",):
        return tone_map_reinhard(image, exposure=exposure)
    elif m in ("reinhard_jodie", "jodie"):
        return tone_map_reinhard_jodie(image, exposure=exposure)
    else:
        return tone_map_linear_clamp(image)
