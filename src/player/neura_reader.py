"""High-speed .neura Container Reader and GPU Instantiator."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Union

import torch

from src.model.quantizer import unpack_neura_container
from src.model.siren_video import SirenVideo
from src.utils.neura_format import HEADER_SIZE, deserialize_header


class NeuraReader:
    """Loads and deserializes proprietary .neura binary containers."""

    @staticmethod
    def inspect_header(filepath: str) -> Dict[str, Any]:
        """Read only the 128-byte header to inspect metadata instantly without loading weights."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f".neura file not found: {filepath}")

        with open(filepath, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            meta = deserialize_header(header_bytes)

        file_size_kb = os.path.getsize(filepath) / 1024.0
        meta["file_size_kb"] = file_size_kb
        meta["file_path"] = filepath
        return meta

    @staticmethod
    def load(
        filepath: str,
        device: Union[str, torch.device] = "cuda",
    ) -> Tuple[SirenVideo, Dict[str, Any]]:
        """Load .neura container, dequantize INT8 weights, and return executable SirenVideo model on GPU."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f".neura container not found: {filepath}")

        target_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        model, meta = unpack_neura_container(filepath, device=target_device)

        file_size_kb = os.path.getsize(filepath) / 1024.0
        meta["file_size_kb"] = file_size_kb
        meta["file_path"] = filepath
        meta["device"] = str(target_device)

        return model, meta
