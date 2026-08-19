"""Render trained Spatio-Temporal SIREN model to MP4 video."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

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
from tqdm import tqdm

from src.data.video_coordinate_dataset import load_video_tensor
from src.model.siren_video import SirenVideo
from src.training.video_trainer import reconstruct_video_frame
from src.utils.metrics import calculate_psnr, calculate_ssim


def load_video_siren_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[SirenVideo, dict]:
    """Load model architecture and weights from video checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = SirenVideo(
        in_features=config.get("in_features", 3),
        hidden_features=config.get("hidden_features", 384),
        hidden_layers=config.get("hidden_layers", 6),
        out_features=config.get("out_features", 3),
        omega_xy=config.get("omega_xy", 30.0),
        omega_t=config.get("omega_t", 10.0),
        omega_0_hidden=config.get("omega_0_hidden", 30.0),
        final_activation=config.get("final_activation", "clamp"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def render_video(
    checkpoint_path: str,
    output_path: str = "runs/rendered_video.mp4",
    render_height: int | None = None,
    render_width: int | None = None,
    fps: float | None = None,
    ground_truth_path: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Render continuous spatio-temporal SIREN into an MP4 video."""
    torch_device = torch.device(device)
    model, checkpoint = load_video_siren_checkpoint(checkpoint_path, torch_device)
    meta = checkpoint.get("video_meta", {})

    frame_count = meta.get("frame_count", 96)
    out_fps = fps if fps is not None else meta.get("fps", 24.0)
    out_h = render_height if render_height is not None else meta.get("height", 720)
    out_w = render_width if render_width is not None else meta.get("width", 1280)

    print(f"\n=======================================================")
    print(f"[START] Rendering Spatio-Temporal Video from SIREN Neural Codec")
    print(f"   Checkpoint        : {checkpoint_path}")
    print(f"   Resolution        : {out_w}x{out_h}")
    print(f"   Frames / FPS      : {frame_count} frames @ {out_fps:.1f} FPS")
    print(f"   Output Path       : {output_path}")
    print(f"   Device            : {torch_device}")
    print(f"=======================================================\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))

    # Ground truth video for PSNR calculation if available
    gt_tensor = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_tensor, _, _, _, _ = load_video_tensor(ground_truth_path, target_size=(out_h, out_w))
        gt_tensor = gt_tensor.to(torch_device)

    psnr_list: list[float] = []
    ssim_list: list[float] = []

    pbar = tqdm(range(frame_count), desc="Rendering Frames", unit="frame")
    for i in pbar:
        t_val = float(-1.0 + 2.0 * i / (frame_count - 1)) if frame_count > 1 else 0.0

        pred_rgb = reconstruct_video_frame(
            model=model,
            t_val=t_val,
            height=out_h,
            width=out_w,
            device=torch_device,
        )

        if gt_tensor is not None and i < gt_tensor.shape[0]:
            gt_frame = gt_tensor[i]
            frame_psnr = calculate_psnr(pred_rgb, gt_frame)
            frame_ssim = calculate_ssim(pred_rgb, gt_frame)
            psnr_list.append(frame_psnr)
            ssim_list.append(frame_ssim)
            pbar.set_postfix({"PSNR": f"{frame_psnr:.2f}dB", "SSIM": f"{frame_ssim:.4f}"})

        frame_np = (pred_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"\n[SUCCESS] Rendered video saved to: '{output_path}'")

    if psnr_list:
        mean_psnr = float(np.mean(psnr_list))
        mean_ssim = float(np.mean(ssim_list))
        print(f"[METRICS] Full Video PSNR: {mean_psnr:.2f} dB | Mean SSIM: {mean_ssim:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Spatio-Temporal SIREN to MP4.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_video_siren.pth", help="Model checkpoint")
    parser.add_argument("--output", type=str, default="runs/rendered_video.mp4", help="Output MP4 path")
    parser.add_argument("--height", type=int, default=None, help="Render height (default: native)")
    parser.add_argument("--width", type=int, default=None, help="Render width (default: native)")
    parser.add_argument("--fps", type=float, default=None, help="Render FPS (default: native)")
    parser.add_argument("--ground_truth", type=str, default="Short_Clip_720p.mp4", help="Ground truth video for PSNR calculation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()
    render_video(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        render_height=args.height,
        render_width=args.width,
        fps=args.fps,
        ground_truth_path=args.ground_truth if os.path.exists(args.ground_truth) else None,
        device=args.device,
    )
