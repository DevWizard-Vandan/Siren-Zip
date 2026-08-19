"""Multi-Chunk GPU Training Orchestrator with Aggressive Memory Cleanup & Real-Time ETA."""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.chunking.video_splitter import ChunkMetadata, VideoSplitter
from src.container.neura_v2_writer import NeuraV2Writer
from src.model.siren_video import SirenVideo
from src.utils.metrics import compute_psnr_gpu


class ChunkTrainingResult(NamedTuple):
    chunk_idx: int
    num_frames: int
    duration_sec: float
    final_loss: float
    psnr_db: float
    training_time_sec: float
    payload_bytes: int


class ChunkOrchestrator:
    """Orchestrates temporal chunking, GPU training, memory management, and .neura 2.0 streaming."""

    def __init__(
        self,
        video_splitter: VideoSplitter,
        output_neura_path: str,
        epochs_per_chunk: int = 800,
        batch_size: int = 65536,
        lr: float = 2e-4,
        hidden_features: int = 384,
        hidden_layers: int = 6,
        omega_xy: float = 30.0,
        omega_t: float = 10.0,
        omega_0_hidden: float = 30.0,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.splitter = video_splitter
        self.output_neura_path = output_neura_path
        self.epochs_per_chunk = epochs_per_chunk
        self.batch_size = batch_size
        self.lr = lr

        self.model_config = {
            "hidden_features": hidden_features,
            "hidden_layers": hidden_layers,
            "omega_xy": omega_xy,
            "omega_t": omega_t,
            "omega_0_hidden": omega_0_hidden,
            "final_activation": "clamp",
        }

        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # Enable TF32 for matrix multiplication speedup on Ampere/Ada GPUs
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def train_single_chunk(
        self,
        chunk_info: ChunkMetadata,
        chunk_tensor: torch.Tensor,
    ) -> Tuple[SirenVideo, float, float]:
        """Train a micro-SIREN network on a single temporal chunk tensor (T, H, W, 3)."""
        num_frames, height, width, _ = chunk_tensor.shape
        total_voxels = num_frames * height * width
        flat_rgb = chunk_tensor.reshape(-1, 3).contiguous()

        # Precompute coordinate scaling constants
        hw = height * width
        w_denom = float(max(1, width - 1))
        h_denom = float(max(1, height - 1))
        t_denom = float(max(1, num_frames - 1))

        # Instantiate fresh micro-SIREN model
        model = SirenVideo(
            in_features=3,
            hidden_features=self.model_config["hidden_features"],
            hidden_layers=self.model_config["hidden_layers"],
            out_features=3,
            omega_xy=self.model_config["omega_xy"],
            omega_t=self.model_config["omega_t"],
            omega_0_hidden=self.model_config["omega_0_hidden"],
            final_activation="clamp",
        ).to(self.device)

        optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs_per_chunk, eta_min=1e-6)
        criterion = nn.MSELoss()

        model.train()
        final_loss = 0.0

        for _ in range(self.epochs_per_chunk):
            optimizer.zero_grad(set_to_none=True)

            # High-throughput 1D GPU coordinate sampling
            flat_idx = torch.randint(0, total_voxels, (self.batch_size,), device=self.device)
            batch_rgb = flat_rgb[flat_idx]

            idx_t = flat_idx // hw
            rem_hw = flat_idx % hw
            idx_y = rem_hw // width
            idx_x = rem_hw % width

            x = -1.0 + 2.0 * (idx_x.float() / w_denom)
            y = -1.0 + 2.0 * (idx_y.float() / h_denom)
            t = -1.0 + 2.0 * (idx_t.float() / t_denom)

            batch_coords = torch.stack([x, y, t], dim=-1)

            # Forward pass
            pred_rgb = model(batch_coords)
            loss = criterion(pred_rgb, batch_rgb)

            loss.backward()
            optimizer.step()
            scheduler.step()
            final_loss = float(loss.item())

        # Evaluate final reconstruction PSNR on a representative keyframe (middle frame)
        model.eval()
        with torch.no_grad():
            mid_idx = num_frames // 2
            mid_t = float(-1.0 + 2.0 * mid_idx / t_denom) if num_frames > 1 else 0.0

            y_c = torch.linspace(-1.0, 1.0, steps=height, device=self.device)
            x_c = torch.linspace(-1.0, 1.0, steps=width, device=self.device)
            y_g, x_g = torch.meshgrid(y_c, x_c, indexing="ij")
            t_g = torch.full_like(x_g, fill_value=mid_t)

            grid_coords = torch.stack([x_g, y_g, t_g], dim=-1).reshape(-1, 3)

            preds = []
            chunk_size = 262144
            for s in range(0, grid_coords.shape[0], chunk_size):
                preds.append(model(grid_coords[s : s + chunk_size]))
            pred_frame = torch.cat(preds, dim=0).reshape(height, width, 3).clamp(0.0, 1.0)
            gt_frame = chunk_tensor[mid_idx]

            psnr_val = float(compute_psnr_gpu(pred_frame, gt_frame))

        return model, final_loss, psnr_val

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute full multi-chunk compression pipeline and write .neura 2.0 container."""
        video_info = self.splitter.get_video_info()
        total_chunks = len(self.splitter.chunks)

        print(f"\n=======================================================", flush=True)
        print(f"🎬 SIREN-ZIP 2.0: Neural GOP Auto-Chunking & Compression", flush=True)
        print(f"   Input Video       : {video_info['video_path']}", flush=True)
        print(f"   Native Resolution : {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f} FPS", flush=True)
        print(f"   Total Duration    : {video_info['total_duration']:.2f}s ({video_info['total_frames']} frames)", flush=True)
        print(f"   Chunk Duration    : {self.splitter.chunk_duration:.1f}s/chunk -> {total_chunks} Neural GOP Chunks", flush=True)
        print(f"   Epochs Per Chunk  : {self.epochs_per_chunk} (Batch Size: {self.batch_size:,})", flush=True)
        print(f"   Destination File  : {self.output_neura_path}", flush=True)
        print(f"=======================================================\n", flush=True)

        writer = NeuraV2Writer(
            output_path=self.output_neura_path,
            video_meta=video_info,
            model_config=self.model_config,
            total_chunks=total_chunks,
            chunk_duration=self.splitter.chunk_duration,
        )

        results: List[ChunkTrainingResult] = []
        global_start_time = time.perf_counter()

        for idx, chunk_info in enumerate(self.splitter.chunks):
            chunk_start_time = time.perf_counter()

            # 1. Extract temporal slice into GPU memory
            chunk_tensor = self.splitter.extract_chunk_tensor(chunk_info, device=self.device)

            # 2. Train micro-SIREN representation
            model, final_loss, psnr_val = self.train_single_chunk(chunk_info, chunk_tensor)

            # 3. Append quantized INT8 payload to .neura 2.0 container
            payload_size = writer.append_chunk(
                chunk_idx=chunk_info.chunk_idx,
                start_time=chunk_info.start_time,
                end_time=chunk_info.end_time,
                num_frames=chunk_info.num_frames,
                model_or_tensors=model,
            )

            chunk_elapsed = time.perf_counter() - chunk_start_time
            results.append(
                ChunkTrainingResult(
                    chunk_idx=chunk_info.chunk_idx,
                    num_frames=chunk_info.num_frames,
                    duration_sec=chunk_info.end_time - chunk_info.start_time,
                    final_loss=final_loss,
                    psnr_db=psnr_val,
                    training_time_sec=chunk_elapsed,
                    payload_bytes=payload_size,
                )
            )

            # Calculate Global Progress & ETA
            completed = idx + 1
            remaining = total_chunks - completed
            avg_time_per_chunk = (time.perf_counter() - global_start_time) / completed
            eta_sec = remaining * avg_time_per_chunk
            eta_min = int(eta_sec // 60)
            eta_rem_sec = int(eta_sec % 60)

            print(
                f"[{completed:02d}/{total_chunks:02d}] "
                f"GOP [{chunk_info.start_time:5.1f}s - {chunk_info.end_time:5.1f}s] "
                f"| Frames: {chunk_info.num_frames:2d} "
                f"| Loss: {final_loss:.5f} "
                f"| PSNR: {psnr_val:5.2f} dB "
                f"| Time: {chunk_elapsed:4.1f}s "
                f"| ETA: {eta_min:02d}:{eta_rem_sec:02d}",
                flush=True,
            )

            # Aggressive GPU Memory Cleanup
            del model, chunk_tensor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # Finalize .neura 2.0 container and Seek Index Table
        total_neura_bytes = writer.finalize()
        total_time_sec = time.perf_counter() - global_start_time

        # Calculate Compression Statistics
        orig_file_bytes = os.path.getsize(video_info["video_path"]) if os.path.exists(video_info["video_path"]) else 1
        raw_rgb_bytes = video_info["total_frames"] * video_info["width"] * video_info["height"] * 3
        compression_ratio_raw = raw_rgb_bytes / total_neura_bytes if total_neura_bytes > 0 else 0.0
        compression_ratio_mp4 = orig_file_bytes / total_neura_bytes if total_neura_bytes > 0 else 0.0

        mean_psnr = float(np.mean([r.psnr_db for r in results]))

        print(f"\n=========================================================================================", flush=True)
        print(f"🎉 SIREN-ZIP 2.0: MULTI-CHUNK CINEMA COMPRESSION COMPLETE", flush=True)
        print(f"-----------------------------------------------------------------------------------------", flush=True)
        print(f"   Total Chunks Compressed   : {total_chunks} Neural GOP Chunks", flush=True)
        print(f"   Mean Reconstruction PSNR  : {mean_psnr:.2f} dB", flush=True)
        print(f"   Total Compression Time    : {total_time_sec:.1f}s ({total_time_sec/60.0:.2f} min)", flush=True)
        print(f"   Raw Uncompressed RGB Size : {raw_rgb_bytes / (1024*1024):.1f} MB", flush=True)
        print(f"   Baseline Compressed MP4   : {orig_file_bytes / 1024:.1f} KB ({orig_file_bytes / (1024*1024):.2f} MB)", flush=True)
        print(f"   SIREN-ZIP (.neura 2.0)    : {total_neura_bytes / 1024:.1f} KB ({total_neura_bytes / (1024*1024):.2f} MB)", flush=True)
        print(f"   Compression Ratio vs Raw  : {compression_ratio_raw:.1f}x", flush=True)
        print(f"   Compression Ratio vs MP4  : {compression_ratio_mp4:.2f}x", flush=True)
        print(f"=========================================================================================\n", flush=True)

        return {
            "total_chunks": total_chunks,
            "mean_psnr": mean_psnr,
            "total_time_sec": total_time_sec,
            "neura_file_size_bytes": total_neura_bytes,
            "raw_rgb_bytes": raw_rgb_bytes,
            "compression_ratio_raw": compression_ratio_raw,
            "compression_ratio_mp4": compression_ratio_mp4,
            "results": results,
        }
