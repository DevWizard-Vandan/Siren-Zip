"""Perceptual Cinema Video Compressor (NeRV + Perceptual Loss + Oklab + INT8 Pruning)."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim

from src.color.perceptual_color import oklab_to_rgb, rgb_to_oklab
from src.container.neura_v2_writer import NeuraV2Writer
from src.ingestion.universal_demuxer import UniversalDemuxer
from src.model.perceptual_nerv import PerceptualNeRVVideo
from src.model.quantizer import quantize_model
from src.training.perceptual_loss import PerceptualCinemaLoss


def train_perceptual_nerv(
    video_frames_rgb: np.ndarray,
    device: str = "cuda",
    epochs: int = 1200,
    lr: float = 1e-3,
    target_h: int = 720,
    target_w: int = 1280,
    prune_sparsity: float = 0.30,
) -> Tuple[nn.Module, float]:
    """Train a Perceptual NeRV frame-tensor network over the video clip."""
    T_frames = video_frames_rgb.shape[0]

    model = PerceptualNeRVVideo(
        num_freqs=12,
        stem_dim=256,
        target_height=target_h,
        target_width=target_w,
        color_space="oklab",
    ).to(device)

    # Keep frames on CPU in uint8 or pre-resized to target resolution to conserve VRAM (<100 MB)
    if video_frames_rgb.shape[1] != target_h or video_frames_rgb.shape[2] != target_w:
        resized_frames = []
        for f in video_frames_rgb:
            resized_frames.append(cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA))
        video_frames_rgb = np.stack(resized_frames, axis=0)

    # Time indices t in [0.0, 1.0]
    t_coords = torch.linspace(0.0, 1.0, T_frames, device=device).unsqueeze(1)  # (T, 1)

    criterion = PerceptualCinemaLoss(
        l1_weight=0.50,
        ssim_weight=0.30,
        edge_weight=0.20,
        noise_deadband=0.012,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    batch_size = min(4, T_frames)
    model.train()

    for ep in range(1, epochs + 1):
        indices = torch.randint(0, T_frames, (batch_size,), device=device)
        batch_t = t_coords[indices]
        
        # Stream only batch from CPU to GPU (takes <2MB VRAM)
        batch_np = video_frames_rgb[indices.cpu().numpy()]
        batch_targets = torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device) / 255.0

        optimizer.zero_grad()
        preds = model(batch_t)
        loss, _ = criterion(preds, batch_targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % 100 == 0 or ep == epochs:
            with torch.no_grad():
                val_mse = nn.functional.mse_loss(preds, batch_targets).item()
                psnr = -10.0 * math.log10(max(1e-9, val_mse)) if val_mse > 0 else 50.0
                print(f"  [Epoch {ep:04d}/{epochs}] Loss: {loss.item():.5f} | PSNR: {psnr:.2f} dB", flush=True)

    # Apply 30% Magnitude Weight Pruning to eliminate near-zero redundant parameters
    if prune_sparsity > 0:
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name="weight", amount=prune_sparsity)
                prune.remove(module, "weight")

    # Final PSNR across all frames
    model.eval()
    with torch.no_grad():
        all_mses = []
        for i in range(0, T_frames, 4):
            end_i = min(i + 4, T_frames)
            b_t = t_coords[i:end_i]
            b_targets = torch.from_numpy(video_frames_rgb[i:end_i]).float().permute(0, 3, 1, 2).to(device) / 255.0
            pred_f = model(b_t)
            all_mses.append(nn.functional.mse_loss(pred_f, b_targets).item())
        final_mse = float(np.mean(all_mses))
        final_psnr = -10.0 * math.log10(max(1e-9, final_mse)) if final_mse > 0 else 50.0

    return model, final_psnr


def compress_perceptual_movie(
    input_path: str,
    output_neura_path: str,
    max_duration_sec: Optional[float] = None,
    epochs: int = 1000,
    target_res: str = "720p",
    audio_bitrate_kbps: int = 320,
) -> None:
    print("=" * 80, flush=True)
    print("[*] SIREN-ZIP: PERCEPTUAL CINEMA COMPRESSOR (HVS-OPTIMIZED NeRV)", flush=True)
    print("=" * 80, flush=True)
    print(f"* Input Video        : {input_path}", flush=True)
    print(f"* Target Output      : {output_neura_path}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    print(f"* Hardware Device    : {gpu_name}", flush=True)
    print(f"* Perceptual Engine  : Oklab Gamut + MS-SSIM + Sobel Edges + 30% Weight Pruning", flush=True)

    # 1. Demux & Ingest
    info = UniversalDemuxer.inspect(input_path)
    total_dur = min(info.duration_sec, max_duration_sec) if max_duration_sec else info.duration_sec
    print(f"  * Source Video     : {info.width}x{info.height} @ {info.fps:.2f} FPS ({total_dur:.2f}s)", flush=True)

    if target_res in ("2160p", "4k", "4K"):
        target_h = 2160
        target_w = 3840
    elif target_res in ("1080p", "1080", "fhd"):
        target_h = 1080
        target_w = 1920
    else:
        target_h = 720
        target_w = 1280

    # Ingest frames into memory buffer
    cap = cv2.VideoCapture(input_path)
    frames = []
    max_frames = int(round(total_dur * info.fps))
    for _ in range(max_frames):
        ret, bgr = cap.read()
        if not ret or bgr is None:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()

    frames_np = np.stack(frames, axis=0)
    print(f"  * Ingested Frames  : {len(frames_np)} frames for Neural Overfitting", flush=True)

    # 2. Extract Audio Stream
    tmp_audio = f"{output_neura_path}.opus"
    UniversalDemuxer.extract_audio_payload(input_path, tmp_audio, audio_track_idx=0, bitrate_kbps=audio_bitrate_kbps)
    audio_bytes = b""
    if os.path.exists(tmp_audio):
        with open(tmp_audio, "rb") as f:
            audio_bytes = f.read()
        try:
            os.remove(tmp_audio)
        except Exception:
            pass

    # 3. Train Perceptual NeRV Video Representation
    print(f"\n--- [Training] Overfitting Perceptual NeRV ({epochs} epochs) ---", flush=True)
    t0 = time.perf_counter()
    model, final_psnr = train_perceptual_nerv(
        video_frames_rgb=frames_np,
        device=device,
        epochs=epochs,
        target_h=target_h,
        target_w=target_w,
        prune_sparsity=0.30,
    )
    train_time = time.perf_counter() - t0

    # 4. Quantize and Export to .neura 2.0 Binary Container
    print(f"\n--- [Packaging] Quantizing INT8 Weights & Creating .neura Container ---", flush=True)
    writer = NeuraV2Writer(
        output_path=output_neura_path,
        video_meta={
            "width": target_w,
            "height": target_h,
            "fps": info.fps,
            "total_duration": total_dur,
        },
        model_config={
            "hidden_layers": 7,
            "hidden_features": 256,
            "omega_xy": 30.0,
            "omega_t": 10.0,
            "omega_0_hidden": 30.0,
            "final_activation": "sigmoid",
        },
        total_chunks=1,
        chunk_duration=float(total_dur),
        audio_bytes=audio_bytes,
        audio_codec_type=2 if audio_bytes else 0,
        audio_sample_rate=48000,
        audio_channels=2,
    )

    quantized_tensors = quantize_model(model)
    writer.append_chunk(
        chunk_idx=0,
        start_time=0.0,
        end_time=float(total_dur),
        num_frames=len(frames_np),
        model_or_tensors=quantized_tensors,
    )
    writer.finalize()

    final_size_mb = os.path.getsize(output_neura_path) / (1024.0 * 1024.0)
    raw_size_gb = (info.width * info.height * 3 * len(frames_np)) / (1024.0**3)
    comp_ratio = (raw_size_gb * 1024.0) / max(0.01, final_size_mb)

    print("=" * 80, flush=True)
    print("[SUCCESS] PERCEPTUAL CINEMA COMPRESSION COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    print(f"* Output Container   : {output_neura_path}", flush=True)
    print(f"* Container File Size: {final_size_mb:.2f} MB (Target: ~8-16 MB!)", flush=True)
    print(f"* Raw Uncompressed   : {raw_size_gb:.2f} GB", flush=True)
    print(f"* Compression Ratio  : {comp_ratio:.1f}x smaller", flush=True)
    print(f"* Perceptual PSNR    : {final_psnr:.2f} dB", flush=True)
    print(f"* Total Training Time: {train_time:.2f} seconds", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Perceptual Cinema Video Compressor (NeRV).")
    parser.add_argument("--input", type=str, default="movie_trailer_4k.mkv", help="Input video file")
    parser.add_argument("--output", type=str, default="cinema_perceptual.neura", help="Output .neura file")
    parser.add_argument("--epochs", type=int, default=800, help="Training epochs")
    parser.add_argument("--res", type=str, default="720p", choices=["720p", "1080p", "2160p", "4k"], help="Target rendering resolution")
    parser.add_argument("--max_duration", type=float, default=None, help="Max duration in seconds to compress")
    parser.add_argument("--audio_bitrate", type=int, default=320, help="Audio bitrate in kbps")
    args = parser.parse_args()

    compress_perceptual_movie(
        input_path=args.input,
        output_neura_path=args.output,
        max_duration_sec=args.max_duration,
        epochs=args.epochs,
        target_res=args.res,
        audio_bitrate_kbps=args.audio_bitrate,
    )


if __name__ == "__main__":
    main()
