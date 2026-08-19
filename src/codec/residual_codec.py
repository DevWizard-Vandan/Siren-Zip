"""High-Performance Residual Codec for Mathematically Lossless Visual Reconstruction."""

from __future__ import annotations

import struct
from typing import List, Optional, Tuple

import cv2
import numpy as np
import zstandard as zstd


class ResidualCodec:
    """Encodes and decodes high-frequency residual difference streams between original frames and neural predictions."""

    def __init__(self, compression_level: int = 1) -> None:
        self.cctx = zstd.ZstdCompressor(level=compression_level)
        self.dctx = zstd.ZstdDecompressor()

    def encode_frame_residual(
        self,
        original_rgb: np.ndarray,
        predicted_rgb: np.ndarray,
    ) -> bytes:
        """Compute signed 16-bit residual and compress with Zstandard."""
        # Residual = (Original - Predicted) in [-255, 255]
        diff_16 = original_rgb.astype(np.int16) - predicted_rgb.astype(np.int16)
        raw_bytes = diff_16.tobytes()
        compressed = self.cctx.compress(raw_bytes)
        return compressed

    def decode_frame_residual(
        self,
        predicted_rgb: np.ndarray,
        residual_bytes: bytes,
    ) -> np.ndarray:
        """Decompress residual stream and add to predicted frame for 100% bit-exact reconstruction."""
        if not residual_bytes:
            return predicted_rgb

        decompressed_raw = self.dctx.decompress(residual_bytes)
        diff_16 = np.frombuffer(decompressed_raw, dtype=np.int16).reshape(predicted_rgb.shape)
        reconstructed = np.clip(predicted_rgb.astype(np.int16) + diff_16, 0, 255).astype(np.uint8)
        return reconstructed

    def encode_chunk_residuals(
        self,
        original_frames: np.ndarray,
        predicted_frames: np.ndarray,
    ) -> bytes:
        """Pack an entire chunk of temporal frame residuals into a single contiguous binary block."""
        # original_frames: (T, H, W, 3)
        # predicted_frames: (T, H, W, 3)
        T, H, W, C = original_frames.shape
        diff_16 = original_frames.astype(np.int16) - predicted_frames.astype(np.int16)
        raw_bytes = diff_16.tobytes()

        # Header: (T, H, W, C)
        header = struct.pack("<IIII", T, H, W, C)
        compressed_payload = self.cctx.compress(raw_bytes)
        return header + compressed_payload

    def decode_chunk_residuals(self, chunk_residual_block: bytes) -> Optional[np.ndarray]:
        """Decompress an entire GOP chunk's residuals into a (T, H, W, 3) array."""
        if len(chunk_residual_block) < 16:
            return None

        T, H, W, C = struct.unpack("<IIII", chunk_residual_block[:16])
        compressed_payload = chunk_residual_block[16:]

        decompressed_raw = self.dctx.decompress(compressed_payload)
        diff_16 = np.frombuffer(decompressed_raw, dtype=np.int16).reshape((T, H, W, C))
        return diff_16
