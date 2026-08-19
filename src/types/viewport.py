"""Lightweight type definitions for Siren-Zip UI and Player Engines."""

from __future__ import annotations

from typing import NamedTuple


class ViewportBounds(NamedTuple):
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
