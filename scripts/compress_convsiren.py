"""ConvSIREN Cinema Video Compressor (Sinusoidal Convolutions + Direct RGB + Tiny File Size)."""

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
import torch.nn.functional as F
import torch.optim as optim

from src.container.neura_v2_writer import NeuraV2Writer
from src.ingestion.universal_demuxer import UniversalDemuxer
from src.model.conv_siren_video import ConvSIRENVideo
from src.model.quantizer import quantize_model


def train_conv_siren(
    video_frames_rgb: np.ndarray,
    device: str = "cuda",
    epochs: int = 300,
    lr: float = 1e-3,
    stem_dim: int = 192,
    target_h: int = 720,
    target_w: int = 1280,
) -> Tuple[nn.Module, float]:
    """Train a ConvSIREN video generator across all frames."""
    T_frames = video_frames_rgb.shape[0]

    model = ConvSIRENVideo(
        num_frames=T_frames,
        latent_dim=32,
        stem_dim=stem_dim,
        target_height=target_h,
        target_width=target_w,
    ).to(device)

    # Convert frames to float tensor (T, 3, H, W)
    targets = torch.from_numpy(video_frames_rgb).float().permute(0, 3, 1, 2).to(device) / 255.0

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    batch_size = min(16, T_frames)
    model.train()

    for ep in range(1, epochs + 1):
        indices = torch.randint(0, T_frames, (batch_size,), device=device)
        batch_targets = targets[indices]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            preds = model(indices)
            loss_l1 = F.l1_loss(preds, batch_targets)
            loss = loss_l1

        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        if scale_before <= scale_after:
            scheduler.step()

        if ep % 50 == 0 or ep == epochs:
            with torch.no_grad():
                val_mse = F.mse_loss(preds.float(), batch_targets).item()
                psnr = -10.0 * math.log10(max(1e-9, val_mse)) if val_mse > 0 else 50.0
                print(f"  [Epoch {ep:04d}/{epochs}] Loss: {loss.item():.5f} | PSNR: {psnr:.2f} dB", flush=True)

    # Evaluate full video PSNR
    model.eval()
    with torch.no_grad():
        all_mses = []
        for i in range(0, T_frames, 8):
            end_i = min(i + 8, T_frames)
            idx = torch.arange(i, end_i, device=device)
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                pred_f = model(idx)
            all_mses.append(F.mse_loss(pred_f.float(), targets[i:end_i]).item())
        final_mse = float(np.mean(all_mses))
        final_psnr = -10.0 * math.log10(max(1e-9, final_mse)) if final_mse > 0 else 50.0

    return model, final_psnr


def compress_convsiren_movie(
    input_path: str,
    output_neura_path: str,
    epochs: int = 300,
    target_res: str = "720p",
    audio_track_idx: int = 0,
    audio_bitrate_kbps: int = 320,
    max_duration_sec: Optional[float] = None,
) -> None:
    print("=" * 80, flush=True)
    print("[*] SIREN-ZIP: CONV-SIREN CINEMA COMPRESSOR (TRUE COLOR & ULTRA-COMPACT)", flush=True)
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

    # 1. Demux & Ingest
    info = UniversalDemuxer.inspect(input_path)
    total_dur = min(info.duration_sec, max_duration_sec) if max_duration_sec else info.duration_sec
    print(f"  * Source Video     : {info.width}x{info.height} @ {info.fps:.2f} FPS ({total_dur:.2f}s)", flush=True)

    if target_res in ("1080p", "1080", "fhd"):
        target_h = 1080
        target_w = 1920
    else:
        target_h = 720
        target_w = 1280

    print(f"  * Target Resolution: {target_w}x{target_h} ({target_res})", flush=True)

    # 2. Extract Audio Stream
    print(f"\n--- [Step 1/3] Audio Extraction & High-Fidelity Transcoding ---", flush=True)
    tmp_audio = f"{output_neura_path}.opus"
    UniversalDemuxer.extract_audio_payload(input_path, tmp_audio, audio_track_idx=audio_track_idx, bitrate_kbps=audio_bitrate_kbps)
    audio_bytes = b""
    if os.path.exists(tmp_audio):
        with open(tmp_audio, "rb") as f:
            audio_bytes = f.read()
        try:
            os.remove(tmp_audio)
        except Exception:
            pass
    print(f"  * Audio Payload    : {len(audio_bytes)/1024.0:.1f} KB (Studio Fidelity Opus)", flush=True)

    # 3. Read and resize frames
    print(f"\n--- [Step 2/3] Reading Frames into Memory Buffer ---", flush=True)
    cap = cv2.VideoCapture(input_path)
    frames = []
    max_frames = int(round(total_dur * info.fps))
    for _ in range(max_frames):
        ret, bgr = cap.read()
        if not ret or bgr is None:
            break
        if bgr.shape[0] != target_h or bgr.shape[1] != target_w:
            bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()

    frames_np = np.stack(frames, axis=0)
    print(f"  * Ingested Frames  : {len(frames_np)} frames for Neural Overfitting", flush=True)

    # 4. Train Single Unified ConvSIREN Video Generator
    print(f"\n--- [Step 3/3] Training ConvSIREN Neural Video Generator ({epochs} epochs) ---", flush=True)
    t_start = time.perf_counter()
    model, final_psnr = train_conv_siren(
        video_frames_rgb=frames_np,
        device=device,
        epochs=epochs,
        stem_dim=192,
        target_h=target_h,
        target_w=target_w,
    )
    train_time = time.perf_counter() - t_start

    # 5. Quantize INT8 & Package into Single .neura Container
    writer = NeuraV2Writer(
        output_path=output_neura_path,
        video_meta={
            "width": target_w,
            "height": target_h,
            "fps": info.fps,
            "total_duration": total_dur,
        },
        model_config={
            "hidden_layers": 4,
            "hidden_features": 128,
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
    print("[SUCCESS] CONV-SIREN CINEMA COMPRESSION COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    print(f"* Output Container   : {output_neura_path}", flush=True)
    print(f"* Container File Size: {final_size_mb:.2f} MB (Target: ~8-12 MB!)", flush=True)
    print(f"* Raw Uncompressed   : {raw_size_gb:.2f} GB", flush=True)
    print(f"* Compression Ratio  : {comp_ratio:.1f}x smaller", flush=True)
    print(f"* Final PSNR         : {final_psnr:.2f} dB", flush=True)
    print(f"* Total Training Time: {train_time:.2f} seconds", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ConvSIREN Cinema Video Compressor.")
    parser.add_argument("--input", type=str, default="movie_trailer_4k.mkv", help="Input video file")
    parser.add_argument("--output", type=str, default="cinema_convsiren.neura", help="Output .neura file")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--res", type=str, default="720p", choices=["720p", "1080p"], help="Target rendering resolution")
    parser.add_argument("--audio_track", type=int, default=0, help="Audio track index")
    parser.add_argument("--audio_bitrate", type=int, default=320, help="Audio bitrate in kbps")
    parser.add_argument("--max_duration", type=float, default=None, help="Max duration in seconds to compress")
    args = parser.parse_args()

    compress_convsiren_movie(
        input_path=args.input,
        output_neura_path=args.output,
        epochs=args.epochs,
        target_res=args.res,
        audio_track_idx=args.audio_track,
        audio_bitrate_kbps=args.audio_bitrate,
        max_duration_sec=args.max_duration,
    )


if __name__ == "__main__":
    main()
