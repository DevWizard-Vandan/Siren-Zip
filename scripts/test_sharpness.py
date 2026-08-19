"""High-Frequency Sharpness Diagnostic & Edge-Supervision Test on 4K Cinema Frames."""

from __future__ import annotations

import os
import sys
import time

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.kernels.hash_encoder import SpatioTemporalHashGrid


def compute_spatial_gradient_loss(pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge gradient difference to enforce razor-sharp high-frequency details.
    
    Args:
        pred_img: (B, 3, H, W)
        target_img: (B, 3, H, W)
    """
    # Sobel kernels for horizontal & vertical spatial derivatives
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred_img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred_img.device).view(1, 1, 3, 3)

    # Apply across all 3 RGB channels
    sobel_x = sobel_x.repeat(3, 1, 1, 1)
    sobel_y = sobel_y.repeat(3, 1, 1, 1)

    grad_pred_x = F.conv2d(pred_img, sobel_x, padding=1, groups=3)
    grad_pred_y = F.conv2d(pred_img, sobel_y, padding=1, groups=3)

    grad_target_x = F.conv2d(target_img, sobel_x, padding=1, groups=3)
    grad_target_y = F.conv2d(target_img, sobel_y, padding=1, groups=3)

    loss_gx = F.l1_loss(grad_pred_x, grad_target_x)
    loss_gy = F.l1_loss(grad_pred_y, grad_target_y)

    return loss_gx + loss_gy


class HighFidelityCinemaINR(nn.Module):
    """4K Cinema INR combining High-Resolution Hash Grids with a Fast LeakyReLU/SiLU Decoder."""

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 3840,
        hidden_features: int = 64,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hash_grid = SpatioTemporalHashGrid(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
        )

        in_dim = self.hash_grid.out_dim + 3

        layers = []
        curr = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(curr, hidden_features))
            layers.append(nn.SiLU())
            curr = hidden_features

        layers.append(nn.Linear(curr, 3))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.hash_grid(coords)
        full = torch.cat([feats, coords], dim=-1)
        return self.decoder(full)


def test_sharpness_optimization() -> None:
    print("=" * 80, flush=True)
    print("[*] SIREN-ZIP: HIGH-FREQUENCY SHARPNESS & EDGE CONVERGENCE BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract 72 frames (3.0s) from movie_trailer_4k.mkv around the ruined city (t=45s)
    cap = cv2.VideoCapture("movie_trailer_4k.mkv")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1080)

    frames = []
    for _ in range(72):
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()

    if not frames:
        print("[ERROR] Could not load frames from movie_trailer_4k.mkv", flush=True)
        return

    frames_np = np.stack(frames, axis=0)  # (72, 2160, 3840, 3)
    T, H, W, _ = frames_np.shape
    print(f"* Ingested Ruined City 4K GOP: {T} frames @ {W}x{H}", flush=True)

    # Subsample spatial resolution to 1920x1080 for fast benchmarking
    if W > 1920:
        frames_down = [cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA) for f in frames_np]
        frames_np = np.stack(frames_down, axis=0)
        T, H, W, _ = frames_np.shape
        print(f"* Scaled for Benchmark: {T} frames @ {W}x{H} 1080p Full HD", flush=True)

    target_tensor = torch.from_numpy(frames_np).float().to(device) / 255.0

    model = HighFidelityCinemaINR(
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        max_resolution=W,
        hidden_features=64,
        hidden_layers=2,
    ).to(device)

    # Multi-group optimizer: 1e-2 for Hash Table, 1e-3 for Decoder
    optimizer = optim.AdamW([
        {"params": model.hash_grid.parameters(), "lr": 1e-2, "weight_decay": 1e-6},
        {"params": model.decoder.parameters(), "lr": 1e-3, "weight_decay": 0.0},
    ])

    t_coords_1d = torch.linspace(-1.0, 1.0, T, device=device)
    y_coords_1d = torch.linspace(-1.0, 1.0, H, device=device)
    x_coords_1d = torch.linspace(-1.0, 1.0, W, device=device)

    patch_size = 16
    num_patches = 256  # 256 * 256 = 65,536 points per batch

    dy = torch.arange(patch_size, device=device)
    dx = torch.arange(patch_size, device=device)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")

    t0 = time.perf_counter()
    print("\n--- Training with High-Frequency Edge-Supervision ---", flush=True)

    for epoch in range(1, 601):
        t_idx = torch.randint(0, T, (num_patches,), device=device)
        y_start = torch.randint(0, H - patch_size + 1, (num_patches,), device=device)
        x_start = torch.randint(0, W - patch_size + 1, (num_patches,), device=device)

        py = y_start[:, None, None] + grid_y[None, :, :]
        px = x_start[:, None, None] + grid_x[None, :, :]
        pt = t_idx[:, None, None].expand(-1, patch_size, patch_size)

        coords = torch.stack(
            [x_coords_1d[px], y_coords_1d[py], t_coords_1d[pt]], dim=-1
        ).view(-1, 3)

        targets = target_tensor[pt, py, px].view(-1, 3)

        optimizer.zero_grad()
        preds = model(coords)

        # 1. Pixel Color Loss
        l1_loss = F.l1_loss(preds, targets)

        # 2. High-Frequency Edge Gradient Loss (Sobel)
        preds_patches = preds.view(num_patches, patch_size, patch_size, 3).permute(0, 3, 1, 2)
        targets_patches = targets.view(num_patches, patch_size, patch_size, 3).permute(0, 3, 1, 2)
        grad_loss = compute_spatial_gradient_loss(preds_patches, targets_patches)

        total_loss = l1_loss + 0.3 * grad_loss
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                val_mse = F.mse_loss(preds, targets).item()
                psnr = -10.0 * np.log10(max(1e-9, val_mse))
                print(f"  [Epoch {epoch:04d}/600] L1: {l1_loss.item():.5f} | Edge Grad: {grad_loss.item():.5f} | PSNR: {psnr:.2f} dB", flush=True)

    total_t = time.perf_counter() - t0
    print(f"\n* 600 Epochs completed in {total_t:.2f} seconds ({total_t/600.0*1000.0:.2f} ms/step)", flush=True)

    # Render a full test frame to verify sharpness
    with torch.no_grad():
        test_y = torch.linspace(-1.0, 1.0, 720, device=device)
        test_x = torch.linspace(-1.0, 1.0, 1280, device=device)
        yy, xx = torch.meshgrid(test_y, test_x, indexing="ij")
        tt = torch.zeros_like(xx)
        full_coords = torch.stack([xx.flatten(), yy.flatten(), tt.flatten()], dim=-1)

        # Evaluate in chunks
        out_chunks = []
        for i in range(0, full_coords.shape[0], 262144):
            out_chunks.append(model(full_coords[i : i + 262144]))
        rendered = torch.cat(out_chunks, dim=0).view(720, 1280, 3)
        rendered_np = (torch.clamp(rendered, 0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)

        # Save side-by-side comparison image
        ground_truth = cv2.resize(frames_np[0], (1280, 720))
        comparison = np.hstack([cv2.cvtColor(ground_truth, cv2.COLOR_RGB2BGR), cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)])
        cv2.imwrite("sharpness_verification.png", comparison)
        print("✅ Rendered sharp test frame saved to 'sharpness_verification.png'!", flush=True)


if __name__ == "__main__":
    test_sharpness_optimization()
