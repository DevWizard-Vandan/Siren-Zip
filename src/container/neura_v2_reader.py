"""High-Speed Memory-Mapped .neura 2.0 Streaming Reader with Audio Extraction."""

from __future__ import annotations

import math
import mmap
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from src.container.neura_v2_format import (
    HEADER_V2_SIZE,
    ChunkIndexRecord,
    NeuraV2Header,
    deserialize_index_table,
    deserialize_v2_header,
)
from src.model.quantizer import dequantize_state_dict
from src.model.siren_video import SirenVideo
from src.utils.neura_format import deserialize_payload


class NeuraV2Reader:
    """Memory-mapped streaming reader for .neura 2.0 multi-chunk cinema containers."""

    def __init__(self, neura_path: str) -> None:
        if not os.path.exists(neura_path):
            raise FileNotFoundError(f".neura 2.0 container not found: {neura_path}")

        self.neura_path = neura_path
        self.file_size = os.path.getsize(neura_path)

        # Open file and memory map for sub-millisecond zero-copy slicing
        self.file_handle = open(neura_path, "rb")
        self.mm = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)

        # 1. Parse 128-byte Header
        header_bytes = self.mm[:HEADER_V2_SIZE]
        self.header: NeuraV2Header = deserialize_v2_header(header_bytes)

        # 2. Parse Seek Index Table
        index_start = self.header.index_table_offset
        index_end = index_start + self.header.index_table_size
        table_bytes = self.mm[index_start:index_end]
        self.index_records: List[ChunkIndexRecord] = deserialize_index_table(table_bytes, self.header.total_chunks)

        if len(self.index_records) == 0:
            raise ValueError(f"Corrupt .neura 2.0 container: zero chunk index records found in {neura_path}")

    def close(self) -> None:
        """Close memory map and file handle."""
        if hasattr(self, "mm") and self.mm is not None:
            self.mm.close()
        if hasattr(self, "file_handle") and self.file_handle is not None:
            self.file_handle.close()

    def __enter__(self) -> NeuraV2Reader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_audio_payload(self) -> Tuple[bytes, int, int, int]:
        """Extract multiplexed audio payload bytes and format descriptors.

        Returns:
            audio_bytes: Raw binary compressed audio stream (AAC/Opus/MP3).
            audio_codec_type: 0=None, 1=AAC, 2=Opus, 3=MP3, 4=PCM.
            sample_rate: Sample rate (e.g. 48000 Hz).
            channels: Channel count (e.g. 2, 6, 8).
        """
        if self.header.audio_payload_size == 0 or self.header.audio_payload_offset == 0:
            return b"", 0, 48000, 2

        offset = self.header.audio_payload_offset
        size = self.header.audio_payload_size
        audio_bytes = bytes(self.mm[offset : offset + size])
        return audio_bytes, self.header.audio_codec_type, self.header.audio_sample_rate, self.header.audio_channels

    def get_chunk_info(self, chunk_idx: int) -> ChunkIndexRecord:
        """Retrieve metadata record for a specific chunk index."""
        if chunk_idx < 0 or chunk_idx >= len(self.index_records):
            raise IndexError(f"Chunk index {chunk_idx} out of range [0, {len(self.index_records) - 1}]")
        return self.index_records[chunk_idx]

    def locate_chunk_and_local_time(self, t_global: float) -> Tuple[int, ChunkIndexRecord, float]:
        """Locate active chunk index and compute local normalized coordinate t_local in [-1.0, 1.0]."""
        t_clamped = max(0.0, min(float(self.header.total_duration), float(t_global)))
        tau = max(0.001, float(self.header.chunk_duration))
        candidate_idx = min(len(self.index_records) - 1, max(0, int(t_clamped // tau)))

        record = self.index_records[candidate_idx]
        if not (record.start_time <= t_clamped <= record.end_time):
            low, high = 0, len(self.index_records) - 1
            while low <= high:
                mid = (low + high) // 2
                rec = self.index_records[mid]
                if rec.start_time <= t_clamped <= rec.end_time:
                    candidate_idx = mid
                    record = rec
                    break
                elif t_clamped < rec.start_time:
                    high = mid - 1
                else:
                    low = mid + 1

        dt = max(1e-6, record.end_time - record.start_time)
        norm_factor = (t_clamped - record.start_time) / dt
        t_local = float(max(-1.0, min(1.0, -1.0 + 2.0 * norm_factor)))

        return candidate_idx, record, t_local

    def load_chunk_state_dict(
        self,
        chunk_idx: int,
        device: Union[str, torch.device] = "cuda",
    ) -> Dict[str, torch.Tensor]:
        """Zero-copy slice payload from mmap and dequantize INT8 weights into GPU state_dict in <1ms."""
        rec = self.get_chunk_info(chunk_idx)
        payload_bytes = self.mm[rec.byte_offset : rec.byte_offset + rec.byte_size]

        quantized_tensors = deserialize_payload(payload_bytes, self.header.num_tensors_per_chunk)
        torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        state_dict = dequantize_state_dict(quantized_tensors, torch_device)
        return state_dict

    def create_model_shell(self, device: Union[str, torch.device] = "cuda") -> nn.Module:
        """Create single preallocated GPU model shell ready for instantaneous weight swapping."""
        torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if self.header.hidden_layers <= 3:
            from src.model.hash_siren_video import HashSirenVideo
            # Dynamic probe of n_levels and log2_hashmap_size from first chunk if available
            n_levels = 12
            log2_hashmap_size = 16
            if len(self.index_records) > 0:
                rec = self.index_records[0]
                payload_bytes = self.mm[rec.byte_offset : rec.byte_offset + rec.byte_size]
                quantized_tensors = deserialize_payload(payload_bytes, self.header.num_tensors_per_chunk)
                # Find embedding tensors
                embed_tensors = [q for q in quantized_tensors if "hash_grid.embeddings" in q.name]
                if embed_tensors:
                    n_levels = len(embed_tensors)
                    table_len = embed_tensors[0].shape[0]
                    log2_hashmap_size = int(round(math.log2(table_len)))

            model = HashSirenVideo(
                n_levels=n_levels,
                n_features_per_level=2,
                log2_hashmap_size=log2_hashmap_size,
                hidden_features=self.header.hidden_features,
                hidden_layers=self.header.hidden_layers,
                out_features=3,
            )
        else:
            model = SirenVideo(
                in_features=3,
                hidden_features=self.header.hidden_features,
                hidden_layers=self.header.hidden_layers,
                out_features=3,
                omega_xy=self.header.omega_xy,
                omega_t=self.header.omega_t,
                omega_0_hidden=self.header.omega_0_hidden,
                final_activation=self.header.final_activation,
            )
        model.to(torch_device)
        model.eval()
        return model
