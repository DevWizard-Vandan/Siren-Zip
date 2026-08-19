"""Neura 2.0 Container binary format specifications with Audio & HDR Multiplexing."""

from __future__ import annotations

import io
import struct
from typing import Any, Dict, List, NamedTuple, Tuple

HEADER_V2_SIZE = 128
MAGIC_V2_BYTES = b"NEU2"
CURRENT_V2_VERSION = 2

# Exact 128-byte layout:
# 1. 4s: magic (4)
# 2. I: version (4)
# 3. I: total_chunks (4)
# 4. d: total_duration (8)
# 5. f: base_fps (4)
# 6. I: native_width (4)
# 7. I: native_height (4)
# 8. f: chunk_duration (4)
# 9. I: hidden_layers (4)
# 10. I: hidden_features (4)
# 11. f: omega_xy (4)
# 12. f: omega_t (4)
# 13. f: omega_0_hidden (4)
# 14. I: final_activation_id (4)
# 15. I: num_tensors_per_chunk (4)
# 16. Q: index_table_offset (8)
# 17. Q: index_table_size (8)
# 18. I: audio_codec_type (4)
# 19. I: audio_sample_rate (4)
# 20. I: audio_channels (4)
# 21. I: color_primaries (4)
# 22. I: transfer_characteristics (4)
# 23. Q: audio_payload_offset (8)
# 24. Q: audio_payload_size (8)
# 25. 12s: reserved_padding (12)
# Sum = 4+4+4+8+4+4+4+4+4+4+4+4+4+4+4+8+8+4+4+4+4+4+8+8+12 = 128 bytes.

HEADER_V2_FORMAT = "<4s II d f II f II fff II QQ IIIII QQ 12s"

INDEX_RECORD_FORMAT = "<I dd I QQ"
INDEX_RECORD_SIZE = struct.calcsize(INDEX_RECORD_FORMAT)  # 40 bytes


class ChunkIndexRecord(NamedTuple):
    chunk_idx: int
    start_time: float
    end_time: float
    num_frames: int
    byte_offset: int
    byte_size: int


class NeuraV2Header(NamedTuple):
    version: int
    total_chunks: int
    total_duration: float
    base_fps: float
    native_width: int
    native_height: int
    chunk_duration: float
    hidden_layers: int
    hidden_features: int
    omega_xy: float
    omega_t: float
    omega_0_hidden: float
    final_activation: str
    num_tensors_per_chunk: int
    index_table_offset: int
    index_table_size: int
    audio_codec_type: int  # 0=None, 1=AAC, 2=Opus, 3=MP3, 4=PCM
    audio_sample_rate: int
    audio_channels: int
    color_primaries: int  # 1=Rec.709, 9=Rec.2020
    transfer_characteristics: int  # 1=BT.709, 16=ST.2084 PQ, 18=HLG
    audio_payload_offset: int
    audio_payload_size: int


ACTIVATION_MAP = {"clamp": 0, "sigmoid": 1, "none": 2}
ACTIVATION_MAP_REV = {0: "clamp", 1: "sigmoid", 2: "none"}


def serialize_v2_header(
    total_chunks: int,
    total_duration: float,
    base_fps: float,
    native_width: int,
    native_height: int,
    chunk_duration: float,
    hidden_layers: int,
    hidden_features: int,
    omega_xy: float,
    omega_t: float,
    omega_0_hidden: float,
    final_activation: str,
    num_tensors_per_chunk: int,
    index_table_offset: int,
    index_table_size: int,
    audio_codec_type: int = 0,
    audio_sample_rate: int = 48000,
    audio_channels: int = 2,
    color_primaries: int = 1,
    transfer_characteristics: int = 1,
    audio_payload_offset: int = 0,
    audio_payload_size: int = 0,
) -> bytes:
    """Serialize exact 128-byte aligned .neura 2.0 container header."""
    act_id = ACTIVATION_MAP.get(final_activation, 0)
    header_bytes = struct.pack(
        HEADER_V2_FORMAT,
        MAGIC_V2_BYTES,
        CURRENT_V2_VERSION,
        int(total_chunks),
        float(total_duration),
        float(base_fps),
        int(native_width),
        int(native_height),
        float(chunk_duration),
        int(hidden_layers),
        int(hidden_features),
        float(omega_xy),
        float(omega_t),
        float(omega_0_hidden),
        int(act_id),
        int(num_tensors_per_chunk),
        int(index_table_offset),
        int(index_table_size),
        int(audio_codec_type),
        int(audio_sample_rate),
        int(audio_channels),
        int(color_primaries),
        int(transfer_characteristics),
        int(audio_payload_offset),
        int(audio_payload_size),
        b"\x00" * 12,
    )
    assert len(header_bytes) == HEADER_V2_SIZE, f"Header size must be 128 bytes, got {len(header_bytes)}"
    return header_bytes


def deserialize_v2_header(header_bytes: bytes) -> NeuraV2Header:
    """Deserialize 128-byte container header."""
    if len(header_bytes) < HEADER_V2_SIZE:
        raise ValueError(f"Header too short: expected {HEADER_V2_SIZE}, got {len(header_bytes)}")

    unpacked = struct.unpack(HEADER_V2_FORMAT, header_bytes[:HEADER_V2_SIZE])
    magic = unpacked[0]
    if magic != MAGIC_V2_BYTES:
        raise ValueError(f"Invalid .neura 2.0 magic bytes: {magic!r}, expected {MAGIC_V2_BYTES!r}")

    return NeuraV2Header(
        version=unpacked[1],
        total_chunks=unpacked[2],
        total_duration=unpacked[3],
        base_fps=unpacked[4],
        native_width=unpacked[5],
        native_height=unpacked[6],
        chunk_duration=unpacked[7],
        hidden_layers=unpacked[8],
        hidden_features=unpacked[9],
        omega_xy=unpacked[10],
        omega_t=unpacked[11],
        omega_0_hidden=unpacked[12],
        final_activation=ACTIVATION_MAP_REV.get(unpacked[13], "clamp"),
        num_tensors_per_chunk=unpacked[14],
        index_table_offset=unpacked[15],
        index_table_size=unpacked[16],
        audio_codec_type=unpacked[17],
        audio_sample_rate=unpacked[18],
        audio_channels=unpacked[19],
        color_primaries=unpacked[20],
        transfer_characteristics=unpacked[21],
        audio_payload_offset=unpacked[22],
        audio_payload_size=unpacked[23],
    )


def serialize_index_table(records: List[ChunkIndexRecord]) -> bytes:
    """Serialize list of ChunkIndexRecords into binary index table."""
    buffer = io.BytesIO()
    for rec in records:
        record_bytes = struct.pack(
            INDEX_RECORD_FORMAT,
            int(rec.chunk_idx),
            float(rec.start_time),
            float(rec.end_time),
            int(rec.num_frames),
            int(rec.byte_offset),
            int(rec.byte_size),
        )
        buffer.write(record_bytes)
    return buffer.getvalue()


def deserialize_index_table(table_bytes: bytes, total_chunks: int) -> List[ChunkIndexRecord]:
    """Deserialize binary index table into list of ChunkIndexRecords."""
    buffer = io.BytesIO(table_bytes)
    records: List[ChunkIndexRecord] = []
    for _ in range(total_chunks):
        raw = buffer.read(INDEX_RECORD_SIZE)
        if len(raw) < INDEX_RECORD_SIZE:
            break
        unpacked = struct.unpack(INDEX_RECORD_FORMAT, raw)
        records.append(
            ChunkIndexRecord(
                chunk_idx=unpacked[0],
                start_time=unpacked[1],
                end_time=unpacked[2],
                num_frames=unpacked[3],
                byte_offset=unpacked[4],
                byte_size=unpacked[5],
            )
        )
    return records
