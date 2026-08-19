"""CLI Runner for 4K UHD Cinema Stress-Testing and VRAM Benchmarks."""

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

from src.benchmarks.stress_test_4k import run_4k_stress_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4K UHD Cinema Stress Test.")
    parser.add_argument("--neura", type=str, default="cinema_full.neura", help="Path to .neura container")
    parser.add_argument("--ground_truth", type=str, default="Movie_Trailer_1080p.mp4", help="Baseline ground truth video")
    parser.add_argument("--output_image", type=str, default="runs/stress_test_4k_render.png", help="Path to save 4K image")
    parser.add_argument("--time", type=float, default=1.5, help="Timestamp to evaluate")

    args = parser.parse_args()

    print(f"\n=======================================================", flush=True)
    print(f"🎬 SIREN-ZIP 2.0: 4K UHD (3840x2160) CINEMA STRESS-TEST", flush=True)
    print(f"   Container         : {args.neura}", flush=True)
    print(f"   Target Resolution : 3840x2160 (8,294,400 sub-pixel points)", flush=True)
    print(f"   Evaluation Time t : {args.time:.2f}s", flush=True)
    print(f"=======================================================\n", flush=True)

    res = run_4k_stress_test(
        neura_path=args.neura,
        ground_truth_video=args.ground_truth,
        output_image_path=args.output_image,
        eval_timestamp=args.time,
    )

    print(f"=========================================================================================", flush=True)
    print(f"📊 4K UHD STRESS-TEST & VRAM TELEMETRY RESULTS", flush=True)
    print(f"-----------------------------------------------------------------------------------------", flush=True)
    print(f"   Render Resolution            : {res.render_width} x {res.render_height} ({res.total_pixels_per_frame:,} points)", flush=True)
    print(f"   Container Storage Size       : {res.container_size_mb:.2f} MB", flush=True)
    print(f"   4K Bits-Per-Pixel (BPP)      : {res.bpp_4k:.6f} bpp  (vs H.264 ~0.035000 bpp)", flush=True)
    print(f"   BPP Compression Superiority  : {0.035 / max(1e-6, res.bpp_4k):.1f}x higher bit-efficiency!", flush=True)
    print(f"   4K Full Forward Latency      : {res.eval_time_ms:.1f} ms", flush=True)
    print(f"   GPU Inference Throughput     : {res.megapixels_per_sec:.2f} Megapixels / sec", flush=True)
    print(f"   Peak GPU VRAM Allocated      : {res.peak_vram_mb:.2f} MB  (Strictly capped < 3,500 MB)", flush=True)
    print(f"   Total GPU VRAM Reserved      : {res.reserved_vram_mb:.2f} MB", flush=True)
    print(f"   4K Reconstruction PSNR       : {res.psnr_4k:.2f} dB", flush=True)
    print(f"   4K Reconstruction SSIM       : {res.ssim_4k:.4f}", flush=True)
    print(f"   Saved 4K Render Target       : {res.saved_4k_path}", flush=True)
    print(f"=========================================================================================\n", flush=True)


if __name__ == "__main__":
    main()
