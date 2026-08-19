"""Universal Cinema Compressor: Compresses MKV/MP4/WebM into Siren-Zip .neura 2.0 Containers."""

from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.container.neura_v2_writer import NeuraV2Writer
from src.ingestion.universal_demuxer import UniversalDemuxer
from src.model.hash_siren_video import HashSirenVideo
from src.model.siren_video import SirenVideo


def train_single_chunk(
    frames_rgb: np.ndarray,
    chunk_idx: int,
    device: str = "cuda",
    epochs: int = 300,
    lr: float = 2e-4,
    use_hash_grid: bool = True,
) -> Tuple[Dict[str, Any], float]:
    """Train a single Neural GOP chunk on GPU."""
    T_frames, H, W, _ = frames_rgb.shape

    # 1. Instantiate Model
    if use_hash_grid:
        model = HashSirenVideo(
            n_levels=12,
            n_features_per_level=2,
            log2_hashmap_size=16,
            hidden_features=64,
            hidden_layers=2,
            out_features=3,
        ).to(device)
    else:
        model = SirenVideo(
            in_features=3,
            hidden_features=256,
            hidden_layers=5,
            out_features=3,
            omega_xy=30.0,
            omega_t=10.0,
        ).to(device)

    # 2. Prepare Coordinate 1D Tensors & Target Tensor
    target_tensor = torch.from_numpy(frames_rgb).float().to(device) / 255.0  # (T, H, W, 3)

    t_coords_1d = torch.linspace(-1.0, 1.0, T_frames, device=device)
    y_coords_1d = torch.linspace(-1.0, 1.0, H, device=device)
    x_coords_1d = torch.linspace(-1.0, 1.0, W, device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    batch_size = 65536
    t0 = time.perf_counter()
    model.train()

    for epoch in range(epochs):
        t_idx = torch.randint(0, T_frames, (batch_size,), device=device)
        y_idx = torch.randint(0, H, (batch_size,), device=device)
        x_idx = torch.randint(0, W, (batch_size,), device=device)

        b_coords = torch.stack(
            [x_coords_1d[x_idx], y_coords_1d[y_idx], t_coords_1d[t_idx]], dim=-1
        )
        b_target = target_tensor[t_idx, y_idx, x_idx]

        optimizer.zero_grad()
        pred = model(b_coords)
        loss = criterion(pred, b_target)
        loss.backward()
        optimizer.step()

    # Calculate PSNR
    model.eval()
    with torch.no_grad():
        val_t = torch.randint(0, T_frames, (131072,), device=device)
        val_y = torch.randint(0, H, (131072,), device=device)
        val_x = torch.randint(0, W, (131072,), device=device)
        val_coords = torch.stack([x_coords_1d[val_x], y_coords_1d[val_y], t_coords_1d[val_t]], dim=-1)
        val_target = target_tensor[val_t, val_y, val_x]
        val_pred = model(val_coords)
        val_mse = criterion(val_pred, val_target).item()
        psnr = -10.0 * math.log10(max(1e-9, val_mse)) if val_mse > 0 else 50.0

    return model, psnr


def compress_cinema_movie(
    input_path: str,
    output_neura_path: str,
    chunk_duration_sec: float = 3.0,
    epochs_per_chunk: int = 250,
    fast_hash: bool = True,
    max_duration_sec: Optional[float] = None,
) -> None:
    print("=" * 80)
    print("[*] SIREN-ZIP 2.0: UNIVERSAL CINEMA COMPRESSION PIPELINE")
    print("=" * 80)

    t_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"* Input Video       : {input_path}")
    print(f"* Target Output     : {output_neura_path}")
    print(f"* Hardware Device   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"* Fast Hash-Grid    : {fast_hash} (Instant-NGP Multi-Resolution Acceleration)")

    # 1. Demux and Inspect Media
    print("\n--- [Step 1/5] Universal Demuxing & HDR Extraction ---")
    info = UniversalDemuxer.inspect(input_path)
    total_dur = info.duration_sec if max_duration_sec is None else min(info.duration_sec, max_duration_sec)

    print(f"  * Resolution      : {info.width} x {info.height}")
    print(f"  * Frame Rate      : {info.fps:.2f} FPS")
    print(f"  * Target Duration : {total_dur:.2f}s ({total_dur/60.0:.1f} mins)")
    print(f"  * HDR10+ Detected : {info.hdr_info.is_hdr} (Primaries: {info.hdr_info.color_primaries})")
    print(f"  * Audio Tracks    : {len(info.audio_tracks)} track(s) found")

    # 2. Extract Audio Stream
    print("\n--- [Step 2/5] Spatial Audio Extraction & Transcoding ---")
    tmp_audio = os.path.join(tempfile.gettempdir(), f"audio_track_{int(time.time())}.opus")
    has_audio = UniversalDemuxer.extract_audio_payload(input_path, tmp_audio, codec="opus", bitrate_kbps=192)
    audio_bytes = b""
    if has_audio and os.path.exists(tmp_audio):
        with open(tmp_audio, "rb") as f:
            audio_bytes = f.read()
        print(f"  * Audio Payload   : {len(audio_bytes)/1024.0:.1f} KB (Opus High-Fidelity)")
        try:
            os.remove(tmp_audio)
        except Exception:
            pass
    else:
        print("  * Audio Payload   : None (Muted or extraction bypassed)")

    # 3. Read Frames and Partition into Neural GOPs
    print("\n--- [Step 3/5] Temporal GOP Chunk Partitioning ---")
    cap = cv2.VideoCapture(input_path)
    frames_per_chunk = max(12, int(chunk_duration_sec * info.fps))
    total_chunks = max(1, int(math.ceil(total_dur / chunk_duration_sec)))
    print(f"  * Neural GOPs     : {total_chunks} chunks ({chunk_duration_sec:.1f}s each / {frames_per_chunk} frames/chunk)")

    # 4. Initialize Neura 2.0 Streaming Writer
    print("\n--- [Step 4/5] Multi-Resolution Neural Training & Incremental Writing ---")
    from src.utils.neura_format import QuantizedTensor

    writer = NeuraV2Writer(
        output_path=output_neura_path,
        video_meta={
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "total_duration": total_dur,
        },
        model_config={
            "hidden_layers": 2 if fast_hash else 6,
            "hidden_features": 64 if fast_hash else 256,
            "omega_xy": 30.0,
            "omega_t": 10.0,
            "omega_0_hidden": 30.0,
            "final_activation": "clamp",
        },
        total_chunks=total_chunks,
        chunk_duration=float(chunk_duration_sec),
        audio_bytes=audio_bytes,
        audio_codec_type=2 if audio_bytes else 0,
        audio_sample_rate=48000,
        audio_channels=2,
        color_primaries=9 if info.hdr_info.is_hdr else 1,
        transfer_characteristics=16 if info.hdr_info.is_hdr else 1,
    )

    chunk_psnrs: List[float] = []

    for k in range(total_chunks):
        chunk_frames = []
        for _ in range(frames_per_chunk):
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            # Scale down for micro-chunk training if 4K
            if bgr.shape[1] > 1280:
                bgr = cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            chunk_frames.append(rgb)

        if len(chunk_frames) == 0:
            break

        frames_np = np.stack(chunk_frames, axis=0)
        t_chunk_start = time.perf_counter()

        model, psnr = train_single_chunk(
            frames_rgb=frames_np,
            chunk_idx=k,
            device=device,
            epochs=epochs_per_chunk,
            use_hash_grid=fast_hash,
        )

        chunk_time = time.perf_counter() - t_chunk_start
        chunk_psnrs.append(psnr)

        payload_size = writer.append_chunk(
            chunk_idx=k,
            start_time=float(k * chunk_duration_sec),
            end_time=float((k + 1) * chunk_duration_sec),
            num_frames=len(chunk_frames),
            model_or_tensors=model,
        )

        print(f"  [Chunk {k+1:02d}/{total_chunks:02d}] Trained in {chunk_time:.2f}s | PSNR: {psnr:.2f} dB | INT8 Size: {payload_size/1024.0:.1f} KB")

    cap.release()

    # 5. Finalize Container and Flush Seek Table
    print("\n--- [Step 5/5] Finalizing .neura 2.0 Container & Seek Table ---")
    writer.finalize()

    total_time = time.perf_counter() - t_start
    final_size_mb = os.path.getsize(output_neura_path) / (1024.0 * 1024.0)
    raw_size_gb = (info.width * info.height * 3 * int(total_dur * info.fps)) / (1024.0**3)
    comp_ratio = (raw_size_gb * 1024.0) / final_size_mb

    print("=" * 80)
    print("[SUCCESS] UNIVERSAL CINEMA COMPRESSION COMPLETE!")
    print("=" * 80)
    print(f"* Output File        : {output_neura_path}")
    print(f"* Final Container    : {final_size_mb:.2f} MB")
    print(f"* Raw Uncompressed   : {raw_size_gb:.2f} GB")
    print(f"* Compression Ratio  : {comp_ratio:.1f}x smaller")
    print(f"* Average PSNR       : {sum(chunk_psnrs)/len(chunk_psnrs):.2f} dB")
    print(f"* Total Pipeline Time: {total_time:.2f} seconds")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Siren-Zip Cinema Video Compressor.")
    parser.add_argument("--input", type=str, default="Movie_Trailer_1080p.mp4", help="Input MKV/MP4/WebM video file")
    parser.add_argument("--output", type=str, default="cinema_fast.neura", help="Output .neura 2.0 file")
    parser.add_argument("--chunk_duration", type=float, default=3.0, help="GOP chunk duration in seconds")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per chunk")
    parser.add_argument("--fast_hash", action="store_true", default=True, help="Enable Instant-NGP Hash-Grid acceleration")
    parser.add_argument("--max_duration", type=float, default=12.0, help="Max duration in seconds to compress (for testing)")
    args = parser.parse_args()

    compress_cinema_movie(
        input_path=args.input,
        output_neura_path=args.output,
        chunk_duration_sec=args.chunk_duration,
        epochs_per_chunk=args.epochs,
        fast_hash=args.fast_hash,
        max_duration_sec=args.max_duration,
    )


if __name__ == "__main__":
    import math
    main()
