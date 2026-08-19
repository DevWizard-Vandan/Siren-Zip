"""Optimized Differential Weight Compression (Δθ) for Sub-Millisecond Live Neural Streaming."""

from __future__ import annotations

import json
import struct
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import lz4.frame as lz4_frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

CODEC_RAW = 0
CODEC_ZLIB = 1
CODEC_LZ4 = 2
CODEC_ZSTD = 3


class DeltaCompressor:
    """High-throughput differential weight compressor and sub-millisecond dequantizer."""

    def __init__(self, preferred_codec: str = "auto") -> None:
        if preferred_codec == "zstd" and HAS_ZSTD:
            self.codec_id = CODEC_ZSTD
        elif preferred_codec == "lz4" and HAS_LZ4:
            self.codec_id = CODEC_LZ4
        elif preferred_codec == "zlib":
            self.codec_id = CODEC_ZLIB
        else:
            if HAS_LZ4:
                self.codec_id = CODEC_LZ4  # LZ4 is ultra-fast for sub-ms decompression
            elif HAS_ZSTD:
                self.codec_id = CODEC_ZSTD
            else:
                self.codec_id = CODEC_ZLIB

        if HAS_ZSTD:
            self.cctx = zstd.ZstdCompressor(level=1)  # Level 1 for real-time live speed
            self.dctx = zstd.ZstdDecompressor()
        else:
            self.cctx = None
            self.dctx = None

    def _compress_bytes(self, raw_bytes: bytes) -> bytes:
        if self.codec_id == CODEC_LZ4 and HAS_LZ4:
            return struct.pack("!B", CODEC_LZ4) + lz4_frame.compress(raw_bytes)
        elif self.codec_id == CODEC_ZSTD and self.cctx is not None:
            return struct.pack("!B", CODEC_ZSTD) + self.cctx.compress(raw_bytes)
        elif self.codec_id == CODEC_ZLIB:
            return struct.pack("!B", CODEC_ZLIB) + zlib.compress(raw_bytes, level=1)
        return struct.pack("!B", CODEC_RAW) + raw_bytes

    def _decompress_bytes(self, payload: bytes) -> bytes:
        if len(payload) < 1:
            raise ValueError("Empty compression payload")
        codec = payload[0]
        data = payload[1:]
        if codec == CODEC_LZ4 and HAS_LZ4:
            return lz4_frame.decompress(data)
        elif codec == CODEC_ZSTD and HAS_ZSTD and self.dctx is not None:
            return self.dctx.decompress(data)
        elif codec == CODEC_ZLIB:
            return zlib.decompress(data)
        elif codec == CODEC_RAW:
            return data
        raise ValueError(f"Unknown codec: {codec}")

    def compress_state_dict(
        self,
        current_state_dict: Dict[str, torch.Tensor],
        prev_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        is_keyframe: bool = False,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Contiguously serialize and compress model weight differentials."""
        t0 = time.perf_counter()

        sorted_keys = sorted(current_state_dict.keys())
        meta_list = []
        int8_arrays: List[np.ndarray] = []

        raw_param_bytes = 0
        total_num_params = 0
        byte_offset = 0

        for key in sorted_keys:
            curr_tensor = current_state_dict[key].detach()
            numel = curr_tensor.numel()
            total_num_params += numel
            raw_param_bytes += numel * 4

            if not is_keyframe and prev_state_dict is not None and key in prev_state_dict:
                diff_tensor = curr_tensor - prev_state_dict[key].to(curr_tensor.device)
            else:
                diff_tensor = curr_tensor

            diff_cpu = diff_tensor.float().cpu()
            max_val = torch.max(torch.abs(diff_cpu)).item()
            scale = float(max_val / 127.0) if max_val > 1e-9 else 1.0

            q_int8 = torch.clamp(torch.round(diff_cpu / scale), -128, 127).to(torch.int8).numpy()
            int8_arrays.append(q_int8.ravel())

            meta_list.append({
                "n": key,
                "s": list(curr_tensor.shape),
                "c": scale,
                "o": byte_offset,
                "l": numel,
            })
            byte_offset += numel

        # Concat all INT8 buffers into single contiguous array
        contiguous_int8 = np.concatenate(int8_arrays).tobytes()
        meta_bytes = json.dumps(meta_list).encode("utf-8")

        # Format: [MetaLen (4B uint32)] [MetaJSON] [Contiguous INT8 Data]
        uncompressed_payload = struct.pack("!I", len(meta_bytes)) + meta_bytes + contiguous_int8
        compressed_payload = self._compress_bytes(uncompressed_payload)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        payload_kb = len(compressed_payload) / 1024.0
        ratio = (raw_param_bytes / len(compressed_payload)) if len(compressed_payload) > 0 else 1.0

        stats = {
            "is_keyframe": is_keyframe,
            "num_tensors": len(sorted_keys),
            "num_params": total_num_params,
            "raw_bytes": raw_param_bytes,
            "compressed_bytes": len(compressed_payload),
            "payload_kb": payload_kb,
            "compression_ratio": ratio,
            "compress_time_ms": elapsed_ms,
            "codec": "lz4" if self.codec_id == CODEC_LZ4 else ("zstd" if self.codec_id == CODEC_ZSTD else "zlib"),
        }
        return compressed_payload, stats

    def decompress_state_dict(
        self,
        payload_bytes: bytes,
        prev_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        is_keyframe: bool = False,
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Decompress and reconstruct full state_dict in sub-millisecond latency."""
        t0 = time.perf_counter()
        target_device = torch.device(device)

        raw_bytes = self._decompress_bytes(payload_bytes)

        (meta_len,) = struct.unpack("!I", raw_bytes[:4])
        meta_bytes = raw_bytes[4 : 4 + meta_len]
        meta_list = json.loads(meta_bytes.decode("utf-8"))

        data_offset = 4 + meta_len
        int8_buffer = np.frombuffer(raw_bytes, dtype=np.int8, offset=data_offset).copy()

        state_dict: Dict[str, torch.Tensor] = {}

        for item in meta_list:
            key = item["n"]
            shape = tuple(item["s"])
            scale = float(item["c"])
            offset = item["o"]
            length = item["l"]

            # Sub-array slice without copying
            q_slice = int8_buffer[offset : offset + length].reshape(shape)

            diff_tensor = torch.from_numpy(q_slice).to(target_device, dtype=torch.float32, non_blocking=True)
            if scale != 1.0:
                diff_tensor = diff_tensor * scale

            if not is_keyframe and prev_state_dict is not None and key in prev_state_dict:
                reconstructed = prev_state_dict[key].to(target_device) + diff_tensor
            else:
                reconstructed = diff_tensor

            state_dict[key] = reconstructed

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return state_dict, elapsed_ms
