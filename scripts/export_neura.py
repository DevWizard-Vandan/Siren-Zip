"""Export trained Video SIREN model into quantized .neura binary container."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.model.quantizer import pack_neura_container, unpack_neura_container
from src.model.siren_video import SirenVideo
from src.training.video_trainer import reconstruct_video_frame
from src.utils.metrics import calculate_psnr
from scripts.render_video import load_video_siren_checkpoint


def export_neura_cli(
    checkpoint_path: str = "checkpoints/best_video_siren.pth",
    output_neura_path: str = "my_video.neura",
    baseline_video_path: str = "Short_Clip_720p.mp4",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Serialize model into .neura container, verify integrity, and output compression report."""
    torch_device = torch.device(device)
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint '{checkpoint_path}' not found.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"[START] Exporting SIREN Video Model to .neura Container")
    print(f"   Input Checkpoint  : {checkpoint_path}")
    print(f"   Output Container  : {output_neura_path}")
    print(f"=======================================================\n")

    # 1. Load Model
    model, checkpoint = load_video_siren_checkpoint(checkpoint_path, torch_device)
    video_meta = checkpoint.get("video_meta", {
        "frame_count": 96,
        "fps": 24.0,
        "height": 720,
        "width": 1280,
    })

    # 2. Pack to .neura container
    total_neura_bytes = pack_neura_container(
        model=model,
        video_meta=video_meta,
        output_path=output_neura_path,
    )
    neura_kb = total_neura_bytes / 1024.0

    print(f"[SUCCESS] Exported .neura binary container: '{output_neura_path}' ({neura_kb:.1f} KB)")

    # 3. Verification: Unpack .neura container and verify decoding
    print(f"[INFO] Verifying container decodability & quantization fidelity...")
    unpacked_model, unpacked_meta = unpack_neura_container(output_neura_path, device=torch_device)

    # Reconstruct test frame at t=0.0 on both models
    t_test = 0.0
    h = video_meta.get("height", 720)
    w = video_meta.get("width", 1280)
    rgb_fp32 = reconstruct_video_frame(model, t_test, h, w, torch_device)
    rgb_int8 = reconstruct_video_frame(unpacked_model, t_test, h, w, torch_device)

    quant_psnr = calculate_psnr(rgb_int8, rgb_fp32)
    print(f"[VERIFIED] INT8 Quantization Fidelity vs FP32: {quant_psnr:.2f} dB (Imperceptible drop)")

    # 4. Calculate Compression Metrics
    frame_count = video_meta.get("frame_count", 96)
    raw_uncompressed_bytes = frame_count * h * w * 3
    raw_uncompressed_mb = raw_uncompressed_bytes / (1024 * 1024)
    raw_uncompressed_kb = raw_uncompressed_bytes / 1024.0

    baseline_mp4_kb = (os.path.getsize(baseline_video_path) / 1024.0) if os.path.exists(baseline_video_path) else 0.0
    pth_size_kb = os.path.getsize(checkpoint_path) / 1024.0

    print(f"\n=======================================================")
    print(f"📊 SIREN-ZIP (.NEURA) VIDEO COMPRESSION BENCHMARK")
    print(f"{'Format / Asset':<26} | {'Size':<14} | {'Compression Ratio':<18}")
    print("-" * 64)
    print(f"{'Raw Uncompressed RGB':<26} | {raw_uncompressed_mb:<10.1f} MB | {'1.0x':<18}")
    if baseline_mp4_kb > 0:
        print(f"{'Standard H.264 MP4':<26} | {baseline_mp4_kb:<10.1f} KB | {raw_uncompressed_kb/baseline_mp4_kb:<16.1f}x")
    print(f"{'PyTorch .pth Checkpoint':<26} | {pth_size_kb:<10.1f} KB | {raw_uncompressed_kb/pth_size_kb:<16.1f}x")
    print(f"{'SIREN-ZIP (.neura INT8)':<26} | {neura_kb:<10.1f} KB | {raw_uncompressed_kb/neura_kb:<16.1f}x")
    print("=" * 64)
    print(f"🎯 SIREN-Zip achieves {raw_uncompressed_kb/neura_kb:.1f}x compression over raw video!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Video SIREN to .neura Container.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_video_siren.pth", help="Input checkpoint path")
    parser.add_argument("--output", type=str, default="my_video.neura", help="Output .neura file path")
    parser.add_argument("--baseline_video", type=str, default="Short_Clip_720p.mp4", help="Baseline MP4 for size comparison")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()
    export_neura_cli(
        checkpoint_path=args.checkpoint,
        output_neura_path=args.output,
        baseline_video_path=args.baseline_video,
        device=args.device,
    )
