"""Universal .neura Reader Supporting Both Version 1.0 and Version 2.0 Containers."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Union

import torch

from src.container.neura_v2_format import HEADER_V2_SIZE, deserialize_v2_header
from src.container.neura_v2_reader import NeuraV2Reader
from src.model.quantizer import unpack_neura_container
from src.model.siren_video import SirenVideo
from src.utils.neura_format import HEADER_SIZE, deserialize_header


class NeuraReader:
    """Universal reader for .neura 1.0 (single-chunk) and .neura 2.0 (multi-chunk) containers."""

    @staticmethod
    def detect_version(filepath: str) -> int:
        """Detect container version from magic bytes."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f".neura file not found: {filepath}")

        with open(filepath, "rb") as f:
            magic = f.read(4)

        if magic == b"NEU2":
            return 2
        elif magic == b"NEUR":
            return 1
        else:
            raise ValueError(f"Unknown .neura magic bytes: {magic!r}")

    @staticmethod
    def inspect_header(filepath: str) -> Dict[str, Any]:
        """Read header metadata without loading payload."""
        ver = NeuraReader.detect_version(filepath)
        file_size_kb = os.path.getsize(filepath) / 1024.0

        if ver == 2:
            with open(filepath, "rb") as f:
                header_bytes = f.read(HEADER_V2_SIZE)
            h2 = deserialize_v2_header(header_bytes)
            return {
                "version": 2,
                "total_chunks": h2.total_chunks,
                "total_duration": h2.total_duration,
                "fps": h2.base_fps,
                "width": h2.native_width,
                "height": h2.native_height,
                "chunk_duration": h2.chunk_duration,
                "hidden_layers": h2.hidden_layers,
                "hidden_features": h2.hidden_features,
                "omega_xy": h2.omega_xy,
                "omega_t": h2.omega_t,
                "omega_0_hidden": h2.omega_0_hidden,
                "final_activation": h2.final_activation,
                "file_size_kb": file_size_kb,
                "file_path": filepath,
            }
        else:
            with open(filepath, "rb") as f:
                header_bytes = f.read(HEADER_SIZE)
            h1 = deserialize_header(header_bytes)
            h1["file_size_kb"] = file_size_kb
            h1["file_path"] = filepath
            h1["version"] = 1
            return h1

    @staticmethod
    def load(
        filepath: str,
        device: Union[str, torch.device] = "cuda",
    ) -> Tuple[SirenVideo, Dict[str, Any]]:
        """Load container into model and metadata dictionary."""
        ver = NeuraReader.detect_version(filepath)
        target_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        file_size_kb = os.path.getsize(filepath) / 1024.0

        if ver == 2:
            reader = NeuraV2Reader(filepath)
            model = reader.create_model_shell(device=target_device)
            # Load initial chunk 0 state dict
            state_dict = reader.load_chunk_state_dict(0, device=target_device)
            model.load_state_dict(state_dict, strict=False)

            meta = {
                "version": 2,
                "total_chunks": reader.header.total_chunks,
                "total_duration": reader.header.total_duration,
                "fps": reader.header.base_fps,
                "width": reader.header.native_width,
                "height": reader.header.native_height,
                "chunk_duration": reader.header.chunk_duration,
                "frame_count": int(round(reader.header.total_duration * reader.header.base_fps)),
                "file_size_kb": file_size_kb,
                "file_path": filepath,
                "device": str(target_device),
                "reader_v2": reader,
            }
            return model, meta
        else:
            model, meta = unpack_neura_container(filepath, device=target_device)
            meta["file_size_kb"] = file_size_kb
            meta["file_path"] = filepath
            meta["device"] = str(target_device)
            meta["version"] = 1
            return model, meta
