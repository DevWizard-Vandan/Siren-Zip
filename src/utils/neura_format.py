"""Proprietary .neura 128-Byte Aligned Binary Container Format for Siren-Zip.

Container Architecture:
--------------------------------------------------------------------------------
Offset (Bytes) | Field Name            | Type     | Description
--------------------------------------------------------------------------------
0 - 3          | magic                 | char[4]  | Magic bytes: b'NEUR'
4 - 7          | version               | uint32   | Container version (e.g. 1)
8 - 11         | frame_count           | uint32   | Total temporal frames
12 - 15        | fps                   | float32  | Native video frame rate
16 - 19        | width                 | uint32   | Native spatial width
20 - 23        | height                | uint32   | Native spatial height
24 - 27        | hidden_layers         | uint32   | Model hidden layer count
28 - 31        | hidden_features       | uint32   | Model hidden units per layer
32 - 35        | omega_xy              | float32  | Spatial angular frequency
36 - 39        | omega_t               | float32  | Temporal angular frequency
40 - 43        | omega_0_hidden        | float32  | Hidden layer frequency
44 - 47        | final_activation_id   | uint32   | 0=clamp, 1=sigmoid, 2=none
48 - 51        | num_tensors           | uint32   | Number of quantized parameter arrays
52 - 59        | payload_size_bytes    | uint64   | Total byte length of weights payload
60 - 127       | reserved_padding      | byte[68] | Reserved zero bytes for 128-byte alignment
--------------------------------------------------------------------------------
128 - EOF      | Payload               | Binary   | Quantized INT8 matrices & scale factors
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import struct
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import torch

HEADER_SIZE = 128
MAGIC_BYTES = b"NEUR"
CURRENT_VERSION = 1

ACTIVATION_MAP = {
    "clamp": 0,
    "sigmoid": 1,
    "none": 2,
}

ACTIVATION_MAP_REV = {
    0: "clamp",
    1: "sigmoid",
    2: "none",
}

HEADER_FORMAT = "<4s II f IIII fff II Q 68s"


class QuantizedTensor(NamedTuple):
    name: str
    shape: Tuple[int, ...]
    scale: float
    zero_point: int
    data: bytes  # int8 bytes


def serialize_header(
    frame_count: int,
    fps: float,
    width: int,
    height: int,
    hidden_layers: int,
    hidden_features: int,
    omega_xy: float,
    omega_t: float,
    omega_0_hidden: float,
    final_activation: str,
    num_tensors: int,
    payload_size: int,
) -> bytes:
    """Pack 128-byte aligned container header using struct."""
    act_id = ACTIVATION_MAP.get(final_activation, 0)

    header_bytes = struct.pack(
        HEADER_FORMAT,
        MAGIC_BYTES,
        CURRENT_VERSION,
        int(frame_count),
        float(fps),
        int(width),
        int(height),
        int(hidden_layers),
        int(hidden_features),
        float(omega_xy),
        float(omega_t),
        float(omega_0_hidden),
        int(act_id),
        int(num_tensors),
        int(payload_size),
        b"\x00" * 68,
    )
    assert len(header_bytes) == HEADER_SIZE, f"Header size must be 128 bytes, got {len(header_bytes)}"
    return header_bytes


def deserialize_header(header_bytes: bytes) -> Dict[str, Any]:
    """Unpack 128-byte container header."""
    if len(header_bytes) < HEADER_SIZE:
        raise ValueError(f"Invalid header size: expected {HEADER_SIZE}, got {len(header_bytes)}")

    unpacked = struct.unpack(HEADER_FORMAT, header_bytes[:HEADER_SIZE])
    magic = unpacked[0]
    if magic != MAGIC_BYTES:
        raise ValueError(f"Invalid .neura magic bytes: {magic!r}, expected {MAGIC_BYTES!r}")

    version = unpacked[1]
    frame_count = unpacked[2]
    fps = unpacked[3]
    width = unpacked[4]
    height = unpacked[5]
    hidden_layers = unpacked[6]
    hidden_features = unpacked[7]
    omega_xy = unpacked[8]
    omega_t = unpacked[9]
    omega_0_hidden = unpacked[10]
    act_id = unpacked[11]
    num_tensors = unpacked[12]
    payload_size = unpacked[13]

    return {
        "version": version,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "hidden_layers": hidden_layers,
        "hidden_features": hidden_features,
        "omega_xy": omega_xy,
        "omega_t": omega_t,
        "omega_0_hidden": omega_0_hidden,
        "final_activation": ACTIVATION_MAP_REV.get(act_id, "clamp"),
        "num_tensors": num_tensors,
        "payload_size": payload_size,
    }


def serialize_payload(tensors: List[QuantizedTensor]) -> bytes:
    """Serialize quantized tensor list into binary payload."""
    buffer = io.BytesIO()
    for t in tensors:
        name_bytes = t.name.encode("utf-8")
        buffer.write(struct.pack("<H", len(name_bytes)))
        buffer.write(name_bytes)

        rank = len(t.shape)
        buffer.write(struct.pack("<H", rank))
        for dim in t.shape:
            buffer.write(struct.pack("<I", dim))

        buffer.write(struct.pack("<fi", float(t.scale), int(t.zero_point)))

        buffer.write(struct.pack("<I", len(t.data)))
        buffer.write(t.data)

    return buffer.getvalue()


def deserialize_payload(payload_bytes: bytes, num_tensors: int) -> List[QuantizedTensor]:
    """Deserialize binary payload back to list of QuantizedTensors."""
    tensors, _ = deserialize_payload_with_residual(payload_bytes, num_tensors)
    return tensors


def deserialize_payload_with_residual(payload_bytes: bytes, num_tensors: int) -> Tuple[List[QuantizedTensor], bytes]:
    """Deserialize binary payload and extract any trailing compressed residual stream."""
    buffer = io.BytesIO(payload_bytes)
    tensors: List[QuantizedTensor] = []

    for _ in range(num_tensors):
        name_len = struct.unpack("<H", buffer.read(2))[0]
        name = buffer.read(name_len).decode("utf-8")

        rank = struct.unpack("<H", buffer.read(2))[0]
        shape: list[int] = []
        for _ in range(rank):
            shape.append(struct.unpack("<I", buffer.read(4))[0])

        scale, zero_point = struct.unpack("<fi", buffer.read(8))
        data_len = struct.unpack("<I", buffer.read(4))[0]
        data = buffer.read(data_len)

        tensors.append(
            QuantizedTensor(
                name=name,
                shape=tuple(shape),
                scale=scale,
                zero_point=zero_point,
                data=data,
            )
        )

    residual_bytes = b""
    res_len_bytes = buffer.read(4)
    if len(res_len_bytes) == 4:
        res_len = struct.unpack("<I", res_len_bytes)[0]
        if res_len > 0:
            residual_bytes = buffer.read(res_len)

    return tensors, residual_bytes

