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

    # Dynamic batch size & gradient accumulation for 4K VRAM efficiency
    if target_h > 1080:
        micro_batch = 1
        accum_steps = 4
    elif target_h > 720:
        micro_batch = 2
        accum_steps = 2
    else:
        micro_batch = 4
        accum_steps = 1

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    model.train()

    for ep in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum_steps):
            indices = torch.randint(0, T_frames, (micro_batch,), device=device)
            batch_t = t_coords[indices]
            batch_np = video_frames_rgb[indices.cpu().numpy()]
            batch_targets = torch.from_numpy(batch_np).float().permute(0, 3, 1, 2).to(device) / 255.0

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                preds = model(batch_t)
                loss, _ = criterion(preds, batch_targets)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        if scale_before <= scale_after:
            scheduler.step()

        if ep % 100 == 0 or ep == epochs:
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    preds = model(batch_t)
                val_mse = nn.functional.mse_loss(preds.float(), batch_targets).item()
                psnr = -10.0 * math.log10(max(1e-9, val_mse)) if val_mse > 0 else 50.0
                print(f"  [Epoch {ep:04d}/{epochs}] Loss: {(loss.item() * accum_steps):.5f} | PSNR: {psnr:.2f} dB", flush=True)

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
        eval_batch = 1 if target_h > 1080 else (2 if target_h > 720 else 4)
        for i in range(0, T_frames, eval_batch):
            end_i = min(i + eval_batch, T_frames)
            b_t = t_coords[i:end_i]
            b_targets = torch.from_numpy(video_frames_rgb[i:end_i]).float().permute(0, 3, 1, 2).to(device) / 255.0
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                pred_f = model(b_t)
            all_mses.append(nn.functional.mse_loss(pred_f.float(), b_targets).item())
        final_mse = float(np.mean(all_mses))
        final_psnr = -10.0 * math.log10(max(1e-9, final_mse)) if final_mse > 0 else 50.0

    return model, final_psnr


def compress_perceptual_movie(
    input_path: str,
    output_neura_path: str,
    chunk_duration_sec: float = 3.0,
    epochs_per_chunk: int = 500,
    target_res: str = "1080p",
    audio_track_idx: int = 0,
    audio_bitrate_kbps: int = 320,
    max_duration_sec: Optional[float] = None,
    resume: bool = False,
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
    print(f"* Crash-Resume Mode  : {'ENABLED' if resume else 'DISABLED'}", flush=True)

    # 1. Demux & Probe Input
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

    print(f"  * Target Resolution: {target_w}x{target_h} ({target_res})", flush=True)

    # 2. Compute Neural GOP Chunks
    frames_per_chunk = int(round(chunk_duration_sec * info.fps))
    total_frames = int(round(total_dur * info.fps))
    total_chunks = math.ceil(total_frames / frames_per_chunk)
    print(f"\n--- [Step 1/4] Neural GOP Partitioning ({total_chunks} chunks, {chunk_duration_sec:.1f}s each) ---", flush=True)

    # Setup Checkpointing & Crash Recovery
    chunks_dir = f"{output_neura_path}.chunks"
    progress_file = f"{output_neura_path}.progress.json"
    os.makedirs(chunks_dir, exist_ok=True)

    completed_chunks: Dict[int, float] = {}
    if resume and os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                completed_chunks = {int(k): float(v) for k, v in saved.get("completed_chunks", {}).items()}
                print(f"  [RESUME] Found {len(completed_chunks)} previously completed chunks. Resuming...", flush=True)
        except Exception:
            completed_chunks = {}

    # 3. Audio Transcoding to Studio Opus
    print(f"\n--- [Step 2/4] Audio Extraction & High-Fidelity Transcoding ---", flush=True)
    tmp_audio = os.path.join(chunks_dir, f"audio_track_{audio_track_idx}.opus")
    UniversalDemuxer.extract_audio_payload(input_path, tmp_audio, audio_track_idx=audio_track_idx, bitrate_kbps=audio_bitrate_kbps)
    audio_bytes = b""
    if os.path.exists(tmp_audio):
        with open(tmp_audio, "rb") as f:
            audio_bytes = f.read()
    print(f"  * Audio Payload    : {len(audio_bytes)/1024.0:.1f} KB (Studio Fidelity Opus)", flush=True)

    # 4. Neural Chunk Training Loop (Low RAM Footprint < 500 MB)
    print(f"\n--- [Step 3/4] Multi-Resolution Perceptual Training ({epochs_per_chunk} epochs/chunk) ---", flush=True)
    cap = cv2.VideoCapture(input_path)
    t_start = time.perf_counter()

    for k in range(total_chunks):
        chunk_file = os.path.join(chunks_dir, f"chunk_{k:05d}.pt")

        if k in completed_chunks and os.path.exists(chunk_file):
            print(f"  [Chunk {k+1:02d}/{total_chunks:02d}] Skipped (Already trained, PSNR: {completed_chunks[k]:.2f} dB)", flush=True)
            for _ in range(frames_per_chunk):
                cap.read()
            continue

        chunk_frames = []
        for _ in range(frames_per_chunk):
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            if bgr.shape[0] != target_h or bgr.shape[1] != target_w:
                bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            chunk_frames.append(rgb)

        if len(chunk_frames) == 0:
            break

        frames_np = np.stack(chunk_frames, axis=0)
        t_chunk_start = time.perf_counter()

        model, psnr = train_perceptual_nerv(
            video_frames_rgb=frames_np,
            device=device,
            epochs=epochs_per_chunk,
            target_h=target_h,
            target_w=target_w,
            prune_sparsity=0.30,
        )

        chunk_time = time.perf_counter() - t_chunk_start
        completed_chunks[k] = float(psnr)

        # Save quantized checkpoint
        quantized_tensors = quantize_model(model)
        torch.save({"tensors": quantized_tensors, "psnr": psnr}, chunk_file)

        # Save progress manifest
        manifest = {
            "version": 2,
            "input_path": input_path,
            "output_path": output_neura_path,
            "total_chunks": total_chunks,
            "chunk_duration": chunk_duration_sec,
            "target_res": target_res,
            "completed_chunks": completed_chunks,
        }
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        del model, frames_np, chunk_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        print(f"  [Chunk {k+1:02d}/{total_chunks:02d}] Trained in {chunk_time:.2f}s | PSNR: {psnr:.2f} dB | VRAM: {vram_mb:.1f} MB", flush=True)

    cap.release()

    # 5. Assemble and Finalize .neura 2.0 Binary Container
    print("\n--- [Step 4/4] Assembling Final .neura 2.0 Binary Container & Seek Index ---", flush=True)
    writer = NeuraV2Writer(
        output_path=output_neura_path,
        video_meta={
            "width": target_w,
            "height": target_h,
            "fps": info.fps,
            "total_duration": total_dur,
        },
        model_config={
            "hidden_layers": 8 if target_h > 1080 else 7,
            "hidden_features": 256,
            "omega_xy": 30.0,
            "omega_t": 10.0,
            "omega_0_hidden": 30.0,
            "final_activation": "sigmoid",
        },
        total_chunks=len(completed_chunks),
        chunk_duration=float(chunk_duration_sec),
        audio_bytes=audio_bytes,
        audio_codec_type=2 if audio_bytes else 0,
        audio_sample_rate=48000,
        audio_channels=2,
    )

    for k in sorted(completed_chunks.keys()):
        chunk_file = os.path.join(chunks_dir, f"chunk_{k:05d}.pt")
        chunk_data = torch.load(chunk_file, map_location="cpu", weights_only=False)
        quantized_tensors = chunk_data["tensors"]

        writer.append_chunk(
            chunk_idx=k,
            start_time=float(k * chunk_duration_sec),
            end_time=float((k + 1) * chunk_duration_sec),
            num_frames=frames_per_chunk,
            model_or_tensors=quantized_tensors,
        )

    writer.finalize()

    # Clean up temporary chunks directory and progress file
    try:
        shutil.rmtree(chunks_dir, ignore_errors=True)
        if os.path.exists(progress_file):
            os.remove(progress_file)
    except Exception:
        pass

    total_time = time.perf_counter() - t_start
    final_size_mb = os.path.getsize(output_neura_path) / (1024.0 * 1024.0)
    raw_size_gb = (info.width * info.height * 3 * int(total_dur * info.fps)) / (1024.0**3)
    comp_ratio = (raw_size_gb * 1024.0) / max(0.01, final_size_mb)
    avg_psnr = sum(completed_chunks.values()) / max(1, len(completed_chunks))

    print("=" * 80, flush=True)
    print("[SUCCESS] PERCEPTUAL CINEMA COMPRESSION COMPLETE!", flush=True)
    print("=" * 80, flush=True)
    print(f"* Output Container   : {output_neura_path}", flush=True)
    print(f"* Container File Size: {final_size_mb:.2f} MB", flush=True)
    print(f"* Raw Uncompressed   : {raw_size_gb:.2f} GB", flush=True)
    print(f"* Compression Ratio  : {comp_ratio:.1f}x smaller", flush=True)
    print(f"* Average PSNR       : {avg_psnr:.2f} dB", flush=True)
    print(f"* Total Pipeline Time: {total_time:.2f} seconds", flush=True)
    print("=" * 80, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Perceptual Cinema Video Compressor (NeRV).")
    parser.add_argument("--input", type=str, default="movie_trailer_4k.mkv", help="Input video file")
    parser.add_argument("--output", type=str, default="cinema_perceptual.neura", help="Output .neura file")
    parser.add_argument("--chunk_duration", type=float, default=3.0, help="GOP chunk duration in seconds")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs per GOP chunk")
    parser.add_argument("--res", type=str, default="1080p", choices=["720p", "1080p", "2160p", "4k"], help="Target rendering resolution")
    parser.add_argument("--audio_track", type=int, default=0, help="Audio track index")
    parser.add_argument("--audio_bitrate", type=int, default=320, help="Audio bitrate in kbps")
    parser.add_argument("--max_duration", type=float, default=None, help="Max duration in seconds to compress")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    compress_perceptual_movie(
        input_path=args.input,
        output_neura_path=args.output,
        chunk_duration_sec=args.chunk_duration,
        epochs_per_chunk=args.epochs,
        target_res=args.res,
        audio_track_idx=args.audio_track,
        audio_bitrate_kbps=args.audio_bitrate,
        max_duration_sec=args.max_duration,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

