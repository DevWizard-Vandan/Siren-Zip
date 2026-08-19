"""Asynchronous Double-Buffered CUDA Chunk Prefetcher for Zero-Stutter Playback."""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Tuple, Union

import torch

from src.container.neura_v2_reader import NeuraV2Reader


class ChunkPrefetcher:
    """Lookahead prefetcher that pre-loads adjacent Neural GOP weights into GPU VRAM."""

    def __init__(
        self,
        reader: NeuraV2Reader,
        device: Union[str, torch.device] = "cuda",
        lookahead_threshold: float = 0.75,
    ) -> None:
        self.reader = reader
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.lookahead_threshold = lookahead_threshold

        # Double-buffer cache: chunk_idx -> state_dict on GPU
        self.cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.lock = threading.Lock()
        self.prefetch_thread: Optional[threading.Thread] = None
        self.is_running = True

        # Pre-cache chunk 0
        if len(self.reader.index_records) > 0:
            self._preload_chunk(0)

    def _preload_chunk(self, chunk_idx: int) -> None:
        """Internal helper to load a chunk into cache."""
        if chunk_idx < 0 or chunk_idx >= len(self.reader.index_records):
            return

        with self.lock:
            if chunk_idx in self.cache:
                return

        state_dict = self.reader.load_chunk_state_dict(chunk_idx, device=self.device)

        with self.lock:
            # Maintain maximum 4 active preloaded chunks
            if len(self.cache) >= 4:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            self.cache[chunk_idx] = state_dict

    def check_and_prefetch(self, current_chunk_idx: int, t_local: float) -> None:
        """Check if current chunk is nearing completion and prefetch chunk_idx + 1."""
        # When local time t_local > 0.5 (representing >75% chunk progress)
        if t_local >= (2.0 * self.lookahead_threshold - 1.0):
            next_idx = current_chunk_idx + 1
            if next_idx < len(self.reader.index_records):
                with self.lock:
                    in_cache = next_idx in self.cache
                if not in_cache:
                    # Spawn light background thread to prefetch next chunk without blocking render
                    t = threading.Thread(target=self._preload_chunk, args=(next_idx,), daemon=True)
                    t.start()

    def get_chunk_state_dict(self, chunk_idx: int) -> Tuple[Dict[str, torch.Tensor], bool]:
        """Fetch chunk state dict from prefetch cache (0ms hit) or load immediately on miss."""
        with self.lock:
            if chunk_idx in self.cache:
                return self.cache[chunk_idx], True

        # Cache miss: load synchronously
        state_dict = self.reader.load_chunk_state_dict(chunk_idx, device=self.device)
        with self.lock:
            self.cache[chunk_idx] = state_dict
        return state_dict, False

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
