"""High-throughput GPU training engine for Spatio-Temporal Video SIREN."""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.video_coordinate_dataset import (
    VideoCoordinateData,
    make_frame_coordinate_grid,
)
from src.model.siren_video import SirenVideo
from src.utils.metrics import calculate_psnr, calculate_ssim


@torch.no_grad()
def reconstruct_video_frame(
    model: nn.Module,
    t_val: float,
    height: int,
    width: int,
    device: torch.device,
    chunk_size: int = 262144,
) -> torch.Tensor:
    """Evaluate continuous Spatio-Temporal SIREN for a single frame at timestamp t_val.

    Args:
        model: Trained SirenVideo.
        t_val: Normalized timestamp in [-1.0, 1.0].
        height: Spatial height.
        width: Spatial width.
        device: Torch device.
        chunk_size: Coordinate batch size for inference.

    Returns:
        frame_rgb: (H, W, 3) tensor in [0.0, 1.0].
    """
    model.eval()
    coords_flat = make_frame_coordinate_grid(t_val=t_val, height=height, width=width, device=device)
    num_coords = coords_flat.shape[0]

    preds: list[torch.Tensor] = []
    for start in range(0, num_coords, chunk_size):
        end = min(start + chunk_size, num_coords)
        batch_coords = coords_flat[start:end]
        batch_rgb = model(batch_coords)
        preds.append(batch_rgb)

    full_rgb = torch.cat(preds, dim=0).reshape(height, width, 3)
    return full_rgb


def save_video_sample_frame(
    target_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    save_path: str,
    epoch: int,
    frame_idx: int,
    psnr: float,
    ssim: float,
) -> None:
    """Save visual frame comparison grid: [Ground Truth | SIREN Reconstruction | 5x Error]."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    target_np = (target_rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    pred_np = (pred_rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)

    error_np = np.abs(target_np.astype(np.float32) - pred_np.astype(np.float32)) * 5.0
    error_np = np.clip(error_np, 0.0, 255.0).astype(np.uint8)

    h, w, _ = target_np.shape
    banner_height = 40
    combined_width = w * 3

    combined_canvas = np.zeros((h + banner_height, combined_width, 3), dtype=np.uint8)
    combined_canvas[banner_height:, 0:w] = target_np
    combined_canvas[banner_height:, w:2*w] = pred_np
    combined_canvas[banner_height:, 2*w:3*w] = error_np

    img = Image.fromarray(combined_canvas)
    img.save(save_path)


class VideoTrainer:
    """Orchestrates high-speed GPU spatio-temporal video INR training."""

    def __init__(
        self,
        model: SirenVideo,
        data: VideoCoordinateData,
        lr: float = 2e-4,
        batch_size: int = 262144,
        epochs: int = 4000,
        steps_per_epoch: int = 1,
        save_freq: int = 200,
        eval_frames_count: int = 5,
        checkpoint_dir: str = "checkpoints",
        sample_dir: str = "runs/video_samples",
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.save_freq = save_freq
        self.eval_frames_count = eval_frames_count
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        # Enable TF32 for matrix multiplications on Ampere/Ada/Hopper GPUs
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        total_steps = self.epochs * self.steps_per_epoch

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.best_psnr = -1.0
        self.best_epoch = 0

    def train(self) -> Dict[str, Any]:
        """Execute full spatio-temporal video training loop."""
        raw_size_mb = (self.data.total_voxels * 3 * 4) / (1024 * 1024)
        print(f"\n=======================================================", flush=True)
        print(f"[START] Launching Spatio-Temporal Video SIREN on {self.device}", flush=True)
        print(f"   Frames x H x W     : {self.data.num_frames} frames @ {self.data.width}x{self.data.height} ({self.data.fps:.1f} FPS)", flush=True)
        print(f"   Total Voxels       : {self.data.total_voxels:,} ({raw_size_mb:.1f} MB raw float32)", flush=True)
        print(f"   Model Parameters   : {self.model.get_num_params():,} ({self.model.get_model_size_kb('fp32'):.1f} KB FP32 / {self.model.get_model_size_kb('int8'):.1f} KB INT8)", flush=True)
        print(f"   Batch Size         : {self.batch_size:,} coordinates/step", flush=True)
        print(f"   Frequencies        : Omega_XY={self.model.omega_xy:.1f}, Omega_T={self.model.omega_t:.1f}", flush=True)
        print(f"   Total Epochs       : {self.epochs}", flush=True)
        print(f"=======================================================\n", flush=True)

        pbar = tqdm(range(1, self.epochs + 1), desc="Training Video SIREN", unit="epoch", file=sys.stdout)
        start_time = time.time()

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0

            for _ in range(self.steps_per_epoch):
                batch_coords, batch_rgb = self.data.sample_batch(self.batch_size)

                self.optimizer.zero_grad(set_to_none=True)
                pred_rgb = self.model(batch_coords)
                loss = self.criterion(pred_rgb, batch_rgb)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / self.steps_per_epoch
            train_psnr = 10.0 * math.log10(1.0 / max(avg_loss, 1e-10))

            pbar.set_postfix({
                "Loss": f"{avg_loss:.6f}",
                "PSNR": f"{train_psnr:.2f}dB",
                "Best": f"{self.best_psnr:.2f}dB",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            # Periodic keyframe evaluation & visual sample export
            if epoch % self.save_freq == 0 or epoch == self.epochs or epoch == 1:
                avg_eval_psnr, avg_eval_ssim = self.evaluate_keyframes(epoch=epoch)

                # Save best checkpoint
                if avg_eval_psnr > self.best_psnr:
                    self.best_psnr = avg_eval_psnr
                    self.best_epoch = epoch
                    self.save_checkpoint(
                        save_path=os.path.join(self.checkpoint_dir, "best_video_siren.pth"),
                        epoch=epoch,
                        psnr=avg_eval_psnr,
                        ssim=avg_eval_ssim,
                    )

                # Save latest checkpoint
                self.save_checkpoint(
                    save_path=os.path.join(self.checkpoint_dir, "latest_video_siren.pth"),
                    epoch=epoch,
                    psnr=avg_eval_psnr,
                    ssim=avg_eval_ssim,
                )

        total_time = time.time() - start_time
        print(f"\n[DONE] Video Training Complete in {total_time:.2f}s!", flush=True)
        print(f"[METRIC] Best Video PSNR: {self.best_psnr:.2f} dB (Epoch {self.best_epoch})", flush=True)
        print(f"[SAVED] Checkpoint saved: '{os.path.join(self.checkpoint_dir, 'best_video_siren.pth')}'", flush=True)
        print(f"[SAVED] Visual keyframe samples saved in '{self.sample_dir}'\n", flush=True)

        return {
            "best_psnr": self.best_psnr,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
        }

    @torch.no_grad()
    def evaluate_keyframes(self, epoch: int) -> Tuple[float, float]:
        """Evaluate PSNR and SSIM across sampled keyframes."""
        num_eval = min(self.eval_frames_count, self.data.num_frames)
        frame_indices = np.linspace(0, self.data.num_frames - 1, num_eval, dtype=int)

        psnr_list: list[float] = []
        ssim_list: list[float] = []

        for f_idx in frame_indices:
            t_val = self.data.get_frame_timestamp(int(f_idx))
            target_rgb = self.data.video_tensor[f_idx]

            pred_rgb = reconstruct_video_frame(
                model=self.model,
                t_val=t_val,
                height=self.data.height,
                width=self.data.width,
                device=self.device,
            )

            psnr = calculate_psnr(pred_rgb, target_rgb)
            ssim = calculate_ssim(pred_rgb, target_rgb)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            # Save first keyframe visual comparison
            if f_idx == frame_indices[0]:
                sample_path = os.path.join(self.sample_dir, f"epoch_{epoch:04d}_frame_{f_idx:03d}.png")
                save_video_sample_frame(
                    target_rgb=target_rgb,
                    pred_rgb=pred_rgb,
                    save_path=sample_path,
                    epoch=epoch,
                    frame_idx=int(f_idx),
                    psnr=psnr,
                    ssim=ssim,
                )

        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))
        return avg_psnr, avg_ssim

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        psnr: float,
        ssim: float,
    ) -> None:
        """Serialize video model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.get_config(),
            "video_meta": {
                "frame_count": self.data.num_frames,
                "fps": self.data.fps,
                "height": self.data.height,
                "width": self.data.width,
            },
            "num_params": self.model.get_num_params(),
            "epoch": epoch,
            "psnr": psnr,
            "ssim": ssim,
        }
        torch.save(checkpoint, save_path)
