"""Benchmark SIREN model footprint & quality against JPEG and WebP baselines."""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.coordinate_dataset import load_image_as_tensor
from src.model.siren import SirenImage
from src.training.trainer import reconstruct_full_image
from src.utils.metrics import calculate_psnr, calculate_ssim


def get_file_size_kb(filepath: str) -> float:
    """Return size of file in Kilobytes."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / 1024.0
    return 0.0


def benchmark_jpeg(
    img: Image.Image,
    target_tensor: torch.Tensor,
    qualities: List[int] = [10, 20, 35, 50, 75, 90],
) -> List[Dict[str, float]]:
    """Benchmark JPEG codec across compression quality levels."""
    results = []
    for q in qualities:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=q)
        size_kb = buffer.tell() / 1024.0

        buffer.seek(0)
        dec_img = Image.open(buffer).convert("RGB")
        dec_np = np.array(dec_img, dtype=np.float32) / 255.0
        dec_tensor = torch.from_numpy(dec_np).to(target_tensor.device)

        psnr = calculate_psnr(dec_tensor, target_tensor)
        ssim = calculate_ssim(dec_tensor, target_tensor)

        results.append({
            "codec": "JPEG",
            "quality": q,
            "size_kb": size_kb,
            "psnr": psnr,
            "ssim": ssim,
        })
    return results


def benchmark_webp(
    img: Image.Image,
    target_tensor: torch.Tensor,
    qualities: List[int] = [10, 20, 35, 50, 75, 90],
) -> List[Dict[str, float]]:
    """Benchmark WebP codec across compression quality levels."""
    results = []
    for q in qualities:
        buffer = io.BytesIO()
        img.save(buffer, format="WEBP", quality=q)
        size_kb = buffer.tell() / 1024.0

        buffer.seek(0)
        dec_img = Image.open(buffer).convert("RGB")
        dec_np = np.array(dec_img, dtype=np.float32) / 255.0
        dec_tensor = torch.from_numpy(dec_np).to(target_tensor.device)

        psnr = calculate_psnr(dec_tensor, target_tensor)
        ssim = calculate_ssim(dec_tensor, target_tensor)

        results.append({
            "codec": "WebP",
            "quality": q,
            "size_kb": size_kb,
            "psnr": psnr,
            "ssim": ssim,
        })
    return results


def plot_rate_distortion(
    siren_stats: Dict[str, float],
    jpeg_stats: List[Dict[str, float]],
    webp_stats: List[Dict[str, float]],
    save_path: str = "runs/rate_distortion_curve.png",
) -> None:
    """Generate Rate-Distortion (RD) Curves comparing SIREN with JPEG and WebP."""
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. JPEG Curve
    jpeg_sizes = [r["size_kb"] for r in jpeg_stats]
    jpeg_psnrs = [r["psnr"] for r in jpeg_stats]
    plt.plot(jpeg_sizes, jpeg_psnrs, marker="o", color="#E65100", label="JPEG (Discrete DCT)")

    # 2. WebP Curve
    webp_sizes = [r["size_kb"] for r in webp_stats]
    webp_psnrs = [r["psnr"] for r in webp_stats]
    plt.plot(webp_sizes, webp_psnrs, marker="s", color="#1565C0", label="WebP (Predictive Blocks)")

    # 3. SIREN Footprint Points (FP32, FP16, INT8)
    plt.scatter(
        [siren_stats["size_kb_fp32"]],
        [siren_stats["psnr"]],
        color="#2E7D32",
        s=140,
        marker="^",
        zorder=5,
        label=f"SIREN FP32 ({siren_stats['size_kb_fp32']:.1f} KB, {siren_stats['psnr']:.2f} dB)",
    )
    plt.scatter(
        [siren_stats["size_kb_int8"]],
        [siren_stats["psnr"]],
        color="#00C853",
        s=160,
        marker="*",
        zorder=5,
        label=f"SIREN INT8 Quantized ({siren_stats['size_kb_int8']:.1f} KB, {siren_stats['psnr']:.2f} dB)",
    )

    plt.title("Rate-Distortion Benchmark: SIREN Implicit Neural Representation vs JPEG & WebP", fontsize=13, pad=12)
    plt.xlabel("File / Memory Footprint (Kilobytes)", fontsize=11)
    plt.ylabel("Reconstruction PSNR (dB)", fontsize=11)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Rate-Distortion curve: '{save_path}'")


def run_benchmark(
    checkpoint_path: str = "checkpoints/best_siren.pth",
    image_path: str = "test_target.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_plot: str = "runs/rate_distortion_curve.png",
) -> None:
    """Run full baseline comparison and print formatted results."""
    torch_device = torch.device(device)
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint '{checkpoint_path}' not found.")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"[ERROR] Image '{image_path}' not found.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"[START] Benchmarking SIREN INR vs Traditional Codecs (JPEG / WebP)")
    print(f"=======================================================")

    # 1. Load ground truth image
    pil_img = Image.open(image_path).convert("RGB")
    target_tensor, height, width = load_image_as_tensor(image_path)
    target_tensor = target_tensor.to(torch_device)
    raw_uncompressed_kb = (height * width * 3) / 1024.0

    # 2. Load SIREN model
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    config = checkpoint.get("config", {})

    model = SirenImage(
        in_features=config.get("in_features", 2),
        hidden_features=config.get("hidden_features", 256),
        hidden_layers=config.get("hidden_layers", 5),
        out_features=config.get("out_features", 3),
        omega_0=config.get("omega_0", 30.0),
        omega_0_hidden=config.get("omega_0_hidden", 30.0),
        final_activation=config.get("final_activation", "sigmoid"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()

    num_params = model.get_num_params()
    size_fp32 = model.get_model_size_kb("fp32")
    size_fp16 = model.get_model_size_kb("fp16")
    size_int8 = model.get_model_size_kb("int8")
    checkpoint_file_kb = get_file_size_kb(checkpoint_path)

    # Reconstruct full image and evaluate
    recon = reconstruct_full_image(model, height, width, torch_device)
    siren_psnr = calculate_psnr(recon, target_tensor)
    siren_ssim = calculate_ssim(recon, target_tensor)

    siren_stats = {
        "num_params": num_params,
        "size_kb_fp32": size_fp32,
        "size_kb_fp16": size_fp16,
        "size_kb_int8": size_int8,
        "checkpoint_file_kb": checkpoint_file_kb,
        "psnr": siren_psnr,
        "ssim": siren_ssim,
    }

    # 3. Benchmark JPEG & WebP
    jpeg_stats = benchmark_jpeg(pil_img, target_tensor)
    webp_stats = benchmark_webp(pil_img, target_tensor)

    # 4. Print Summary Table
    print(f"\n[SUMMARY] COMPARATIVE COMPRESSION METRICS TABLE")
    print(f"{'Codec / Format':<24} | {'Size (KB)':<12} | {'Ratio vs Raw':<14} | {'PSNR (dB)':<10} | {'SSIM':<8}")
    print("-" * 77)
    print(f"{'Raw Uncompressed 2K':<24} | {raw_uncompressed_kb:<12.1f} | {'1.0x':<14} | {'Inf':<10} | {'1.000':<8}")
    print(f"{'SIREN (FP32 Weights)':<24} | {size_fp32:<12.1f} | {raw_uncompressed_kb/size_fp32:<13.1f}x | {siren_psnr:<10.2f} | {siren_ssim:<8.4f}")
    print(f"{'SIREN (FP16 Weights)':<24} | {size_fp16:<12.1f} | {raw_uncompressed_kb/size_fp16:<13.1f}x | {siren_psnr:<10.2f} | {siren_ssim:<8.4f}")
    print(f"{'SIREN (INT8 Quantized)':<24} | {size_int8:<12.1f} | {raw_uncompressed_kb/size_int8:<13.1f}x | {siren_psnr:<10.2f} | {siren_ssim:<8.4f}")
    print("-" * 77)

    for r in jpeg_stats:
        name = f"JPEG (Quality {r['quality']})"
        ratio = f"{raw_uncompressed_kb/r['size_kb']:.1f}x"
        print(f"{name:<24} | {r['size_kb']:<12.1f} | {ratio:<14} | {r['psnr']:<10.2f} | {r['ssim']:<8.4f}")
    print("-" * 77)

    for r in webp_stats:
        name = f"WebP (Quality {r['quality']})"
        ratio = f"{raw_uncompressed_kb/r['size_kb']:.1f}x"
        print(f"{name:<24} | {r['size_kb']:<12.1f} | {ratio:<14} | {r['psnr']:<10.2f} | {r['ssim']:<8.4f}")
    print("=" * 77)

    # 5. Plot Rate-Distortion curve
    plot_rate_distortion(siren_stats, jpeg_stats, webp_stats, save_path=output_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SIREN vs JPEG/WebP baselines.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_siren.pth", help="Path to trained SIREN checkpoint")
    parser.add_argument("--image_path", type=str, default="test_target.png", help="Path to ground truth image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--output_plot", type=str, default="runs/rate_distortion_curve.png", help="Path to save RD curve plot")

    args = parser.parse_args()
    run_benchmark(
        checkpoint_path=args.checkpoint,
        image_path=args.image_path,
        device=args.device,
        output_plot=args.output_plot,
    )
