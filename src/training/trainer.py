"""High-throughput GPU training engine for SIREN Image INR."""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union

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

from src.data.coordinate_dataset import ImageCoordinateData, make_coordinate_grid
from src.model.siren import SirenImage
from src.utils.metrics import calculate_psnr, calculate_ssim


@torch.no_grad()
def reconstruct_full_image(
    model: nn.Module,
    height: int,
    width: int,
    device: torch.device,
    chunk_size: int = 262144,
) -> torch.Tensor:
    """Evaluate SIREN over the full (H, W) coordinate grid in memory-efficient chunks."""
    model.eval()
    coords_flat = make_coordinate_grid(height=height, width=width, device=device)
    num_coords = coords_flat.shape[0]

    rgb_preds: list[torch.Tensor] = []
    for start in range(0, num_coords, chunk_size):
        end = min(start + chunk_size, num_coords)
        batch_coords = coords_flat[start:end]
        batch_rgb = model(batch_coords)
        rgb_preds.append(batch_rgb)

    full_rgb = torch.cat(rgb_preds, dim=0).reshape(height, width, 3)
    return full_rgb


def save_reconstruction_sample(
    target_rgb: torch.Tensor,
    pred_rgb: torch.Tensor,
    save_path: str,
    epoch: int,
    psnr: float,
    ssim: float,
) -> None:
    """Save a side-by-side comparison: [Target | Prediction | 5x Error Map]."""
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


class ImageTrainer:
    """Orchestrates high-throughput SIREN training on GPU."""

    def __init__(
        self,
        model: SirenImage,
        data: ImageCoordinateData,
        lr: float = 1e-4,
        batch_size: int = 131072,
        epochs: int = 2000,
        steps_per_epoch: Optional[int] = None,
        save_freq: int = 100,
        checkpoint_dir: str = "checkpoints",
        sample_dir: str = "runs/samples",
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        # Enable TensorFloat32 on NVIDIA Ampere/Ada/Hopper RTX GPUs
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # 1 step per epoch by default for fast iteration (~2-3 minutes for 2000 epochs)
        self.steps_per_epoch = 1 if steps_per_epoch is None else steps_per_epoch
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
        """Execute full training loop and return history."""
        print(f"\n=======================================================")
        print(f"[START] Launching SIREN Image Compressor on {self.device}")
        print(f"   Resolution        : {self.data.width}x{self.data.height} ({self.data.num_pixels:,} pixels)")
        print(f"   Parameters        : {self.model.get_num_params():,} ({self.model.get_model_size_kb('fp32'):.1f} KB FP32 / {self.model.get_model_size_kb('int8'):.1f} KB INT8)")
        print(f"   Batch Size        : {self.batch_size:,} coordinates/step")
        print(f"   Steps per Epoch   : {self.steps_per_epoch}")
        print(f"   Total Epochs      : {self.epochs}")
        print(f"=======================================================\n")

        pbar = tqdm(range(1, self.epochs + 1), desc="Training SIREN", unit="epoch")
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

            # Periodic full evaluation & visual sampling
            if epoch % self.save_freq == 0 or epoch == self.epochs or epoch == 1:
                eval_psnr, eval_ssim, recon_img = self.evaluate()

                # Visual sample save
                sample_path = os.path.join(self.sample_dir, f"epoch_{epoch:04d}.png")
                save_reconstruction_sample(
                    target_rgb=self.data.rgb_grid,
                    pred_rgb=recon_img,
                    save_path=sample_path,
                    epoch=epoch,
                    psnr=eval_psnr,
                    ssim=eval_ssim,
                )

                # Save best checkpoint
                if eval_psnr > self.best_psnr:
                    self.best_psnr = eval_psnr
                    self.best_epoch = epoch
                    self.save_checkpoint(
                        save_path=os.path.join(self.checkpoint_dir, "best_siren.pth"),
                        epoch=epoch,
                        psnr=eval_psnr,
                        ssim=eval_ssim,
                    )

                # Save latest checkpoint
                self.save_checkpoint(
                    save_path=os.path.join(self.checkpoint_dir, "latest_siren.pth"),
                    epoch=epoch,
                    psnr=eval_psnr,
                    ssim=eval_ssim,
                )

        total_time = time.time() - start_time
        print(f"\n[DONE] Training Complete in {total_time:.2f}s!")
        print(f"[METRIC] Best PSNR: {self.best_psnr:.2f} dB (Epoch {self.best_epoch})")
        print(f"[SAVED] Checkpoints saved in '{self.checkpoint_dir}'")
        print(f"[SAVED] Visual samples saved in '{self.sample_dir}'\n")

        return {
            "best_psnr": self.best_psnr,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
        }

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, torch.Tensor]:
        """Perform full-resolution evaluation."""
        recon_img = reconstruct_full_image(
            model=self.model,
            height=self.data.height,
            width=self.data.width,
            device=self.device,
        )
        psnr = calculate_psnr(recon_img, self.data.rgb_grid)
        ssim = calculate_ssim(recon_img, self.data.rgb_grid)
        return psnr, ssim, recon_img

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        psnr: float,
        ssim: float,
    ) -> None:
        """Serialize model weights and metadata."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "in_features": self.model.in_features,
                "hidden_features": self.model.hidden_features,
                "hidden_layers": self.model.hidden_layers,
                "out_features": self.model.out_features,
                "omega_0": self.model.omega_0,
                "omega_0_hidden": self.model.omega_0_hidden,
                "final_activation": self.model.final_activation,
            },
            "image_size": (self.data.height, self.data.width),
            "num_params": self.model.get_num_params(),
            "epoch": epoch,
            "psnr": psnr,
            "ssim": ssim,
        }
        torch.save(checkpoint, save_path)
