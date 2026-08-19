"""Streaming .neura 2.0 Binary Container Writer with Seek Index Table."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.container.neura_v2_format import (
    HEADER_V2_SIZE,
    INDEX_RECORD_SIZE,
    ChunkIndexRecord,
    NeuraV2Header,
    serialize_index_table,
    serialize_v2_header,
)
from src.model.quantizer import quantize_model
from src.model.siren_video import SirenVideo
from src.utils.neura_format import QuantizedTensor, serialize_payload


class NeuraV2Writer:
    """Incremental writer for .neura 2.0 multi-chunk cinema containers."""

    def __init__(
        self,
        output_path: str,
        video_meta: Dict[str, Any],
        model_config: Dict[str, Any],
        total_chunks: int,
        chunk_duration: float = 3.0,
    ) -> None:
        self.output_path = output_path
        self.video_meta = video_meta
        self.model_config = model_config
        self.total_chunks = total_chunks
        self.chunk_duration = chunk_duration

        self.native_width = int(video_meta.get("width", 1920))
        self.native_height = int(video_meta.get("height", 1080))
        self.base_fps = float(video_meta.get("fps", 24.0))
        self.total_duration = float(video_meta.get("total_duration", 0.0))

        self.hidden_layers = int(model_config.get("hidden_layers", 6))
        self.hidden_features = int(model_config.get("hidden_features", 384))
        self.omega_xy = float(model_config.get("omega_xy", 30.0))
        self.omega_t = float(model_config.get("omega_t", 10.0))
        self.omega_0_hidden = float(model_config.get("omega_0_hidden", 30.0))
        self.final_activation = str(model_config.get("final_activation", "clamp"))

        self.index_records: List[ChunkIndexRecord] = []
        self.num_tensors_per_chunk: int = 0
        self.is_finalized: bool = False

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        self.file_handle = open(output_path, "wb")

        # Reserve 128 bytes for provisional header
        provisional_header = b"\x00" * HEADER_V2_SIZE
        self.file_handle.write(provisional_header)

        # Index table will be placed immediately after 128-byte header
        self.index_table_offset = HEADER_V2_SIZE
        self.index_table_size = self.total_chunks * INDEX_RECORD_SIZE

        # Reserve space for Seek Index Table
        provisional_index = b"\x00" * self.index_table_size
        self.file_handle.write(provisional_index)

        # Payload begins after the Index Table
        self.current_payload_offset = HEADER_V2_SIZE + self.index_table_size

    def append_chunk(
        self,
        chunk_idx: int,
        start_time: float,
        end_time: float,
        num_frames: int,
        model_or_tensors: Union[SirenVideo, List[QuantizedTensor]],
    ) -> int:
        """Serialize and append a trained chunk's INT8 payload to the container."""
        if self.is_finalized:
            raise RuntimeError("Cannot append chunks to a finalized NeuraV2Writer.")

        if isinstance(model_or_tensors, nn.Module):
            quantized_tensors = quantize_model(model_or_tensors)
        else:
            quantized_tensors = model_or_tensors

        if self.num_tensors_per_chunk == 0:
            self.num_tensors_per_chunk = len(quantized_tensors)

        payload_bytes = serialize_payload(quantized_tensors)
        payload_size = len(payload_bytes)

        # Write chunk payload at current file position
        byte_offset = self.file_handle.tell()
        self.file_handle.write(payload_bytes)

        # Track in index table
        record = ChunkIndexRecord(
            chunk_idx=chunk_idx,
            start_time=start_time,
            end_time=end_time,
            num_frames=num_frames,
            byte_offset=byte_offset,
            byte_size=payload_size,
        )
        self.index_records.append(record)
        return payload_size

    def finalize(self) -> int:
        """Write finalized Seek Index Table and 128-byte header."""
        if self.is_finalized:
            return os.path.getsize(self.output_path)

        # 1. Seek to Index Table Offset and write real index table
        self.file_handle.seek(self.index_table_offset)
        index_table_bytes = serialize_index_table(self.index_records)
        self.file_handle.write(index_table_bytes)

        # 2. Seek to Byte 0 and write finalized 128-byte header
        self.file_handle.seek(0)
        actual_total_duration = self.index_records[-1].end_time if self.index_records else self.total_duration

        final_header = serialize_v2_header(
            total_chunks=len(self.index_records),
            total_duration=actual_total_duration,
            base_fps=self.base_fps,
            native_width=self.native_width,
            native_height=self.native_height,
            chunk_duration=self.chunk_duration,
            hidden_layers=self.hidden_layers,
            hidden_features=self.hidden_features,
            omega_xy=self.omega_xy,
            omega_t=self.omega_t,
            omega_0_hidden=self.omega_0_hidden,
            final_activation=self.final_activation,
            num_tensors_per_chunk=self.num_tensors_per_chunk,
            index_table_offset=self.index_table_offset,
            index_table_size=len(index_table_bytes),
            audio_offset=0,
        )
        self.file_handle.write(final_header)

        self.file_handle.flush()
        self.file_handle.close()
        self.is_finalized = True

        total_bytes = os.path.getsize(self.output_path)
        return total_bytes
