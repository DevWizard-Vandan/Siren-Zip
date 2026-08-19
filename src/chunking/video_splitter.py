"""Zero-Disk-Bloat Temporal Video Slicer and GPU Chunk Extractor."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Generator, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
import torch


class ChunkMetadata(NamedTuple):
    chunk_idx: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    num_frames: int
    fps: float
    width: int
    height: int


class VideoSplitter:
    """Extracts temporal video slices directly into GPU tensors with zero disk footprint."""

    def __init__(
        self,
        video_path: str,
        chunk_duration: float = 3.0,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.chunk_duration = float(chunk_duration)
        self.target_size = target_size

        # Inspect video stream properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or math.isnan(self.fps):
            self.fps = 24.0

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.total_duration = float(self.total_frames / self.fps) if self.total_frames > 0 else 0.0

        # Plan temporal chunks
        self.chunks: List[ChunkMetadata] = self._plan_chunks()

    def _plan_chunks(self) -> List[ChunkMetadata]:
        """Compute boundary timestamps and frame ranges for all temporal chunks."""
        chunks: List[ChunkMetadata] = []
        frames_per_chunk = max(1, int(round(self.chunk_duration * self.fps)))
        total_chunks = max(1, math.ceil(self.total_frames / frames_per_chunk))

        eff_w = self.target_size[1] if self.target_size else self.native_width
        eff_h = self.target_size[0] if self.target_size else self.native_height

        for k in range(total_chunks):
            start_f = k * frames_per_chunk
            end_f = min(self.total_frames, (k + 1) * frames_per_chunk)
            num_f = end_f - start_f

            if num_f <= 0:
                continue

            start_t = float(start_f / self.fps)
            end_t = float(end_f / self.fps)

            chunks.append(
                ChunkMetadata(
                    chunk_idx=k,
                    start_time=start_t,
                    end_time=end_t,
                    start_frame=start_f,
                    end_frame=end_f,
                    num_frames=num_f,
                    fps=self.fps,
                    width=eff_w,
                    height=eff_h,
                )
            )

        return chunks

    def get_video_info(self) -> Dict[str, Any]:
        """Return global video stream metadata dictionary."""
        eff_w = self.target_size[1] if self.target_size else self.native_width
        eff_h = self.target_size[0] if self.target_size else self.native_height
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "total_duration": self.total_duration,
            "fps": self.fps,
            "width": eff_w,
            "height": eff_h,
            "native_width": self.native_width,
            "native_height": self.native_height,
            "total_chunks": len(self.chunks),
            "chunk_duration": self.chunk_duration,
        }

    def extract_chunk_tensor(
        self,
        chunk_info: ChunkMetadata,
        device: Union[str, torch.device] = "cuda",
    ) -> torch.Tensor:
        """Extract a single temporal slice directly into normalized GPU float32 tensor (T, H, W, 3)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_info.start_frame)

        frames: list[np.ndarray] = []
        for _ in range(chunk_info.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.target_size is not None:
                th, tw = self.target_size
                frame_rgb = cv2.resize(frame_rgb, (tw, th), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Failed to read frames for chunk {chunk_info.chunk_idx}")

        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        chunk_tensor = torch.from_numpy(frames_np).to(torch_device)  # (T, H, W, 3)

        return chunk_tensor
