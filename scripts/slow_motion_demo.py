"""Continuous Temporal Super-Sampling & 4X Slow-Motion Demonstration for SIREN."""

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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from src.data.video_coordinate_dataset import load_video_tensor
from src.model.siren_video import SirenVideo
from src.training.video_trainer import reconstruct_video_frame
from scripts.render_video import load_video_siren_checkpoint


def generate_slow_motion_video(
    checkpoint_path: str,
    output_path: str = "runs/slow_motion_4x.mp4",
    comparison_output_path: str = "runs/slow_motion_comparison.mp4",
    fps_multiplier: float = 4.0,
    render_height: int | None = None,
    render_width: int | None = None,
    ground_truth_path: str | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Evaluate continuous mathematical time t to render 4x slow-motion video."""
    torch_device = torch.device(device)
    model, checkpoint = load_video_siren_checkpoint(checkpoint_path, torch_device)
    meta = checkpoint.get("video_meta", {})

    orig_frames = meta.get("frame_count", 96)
    orig_fps = meta.get("fps", 24.0)
    out_h = render_height if render_height is not None else meta.get("height", 720)
    out_w = render_width if render_width is not None else meta.get("width", 1280)

    # Compute total super-sampled frames: (T - 1) * multiplier + 1
    total_super_frames = int((orig_frames - 1) * fps_multiplier) + 1

    print(f"\n=======================================================")
    print(f"[START] SIREN Continuous Time Super-Sampling ({fps_multiplier:.1f}x Slow-Motion)")
    print(f"   Original Video    : {orig_frames} frames @ {orig_fps:.1f} FPS")
    print(f"   Continuous Output : {total_super_frames} continuous timestamps @ {orig_fps:.1f} FPS")
    print(f"   Resolution        : {out_w}x{out_h}")
    print(f"   Temporal Domain   : Continuous t in [-1.0, 1.0] (dt = {2.0 / (total_super_frames - 1):.6f})")
    print(f"   Device            : {torch_device}")
    print(f"=======================================================\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 1. Main Slow-Motion Video Writer
    writer_slow = cv2.VideoWriter(output_path, fourcc, orig_fps, (out_w, out_h))

    # 2. Side-by-Side Comparison Video Writer (Traditional Frame Duplication vs SIREN Continuous Time)
    comp_w = out_w * 2
    banner_h = 50
    comp_h = out_h + banner_h
    writer_comp = cv2.VideoWriter(comparison_output_path, fourcc, orig_fps, (comp_w, comp_h))

    # Load ground truth video frames if available for nearest-neighbor discrete baseline
    gt_frames = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_tensor, _, _, _, _ = load_video_tensor(ground_truth_path, target_size=(out_h, out_w))
        gt_frames = (gt_tensor.numpy() * 255.0).astype(np.uint8)

    # Fonts for annotation
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    pbar = tqdm(range(total_super_frames), desc="Super-Sampling Time t", unit="frame")
    for i in pbar:
        # Evaluate exact analytical curve at continuous timestamp t in [-1.0, 1.0]
        alpha = i / (total_super_frames - 1) if total_super_frames > 1 else 0.0
        t_val = float(-1.0 + 2.0 * alpha)

        pred_rgb = reconstruct_video_frame(
            model=model,
            t_val=t_val,
            height=out_h,
            width=out_w,
            device=torch_device,
        )

        siren_frame_np = (pred_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        siren_bgr = cv2.cvtColor(siren_frame_np, cv2.COLOR_RGB2BGR)

        # Write to slow motion video
        writer_slow.write(siren_bgr)

        # Build Side-by-Side Comparison Canvas
        # Left: Discrete frame duplication (Nearest original frame)
        nearest_orig_idx = int(round(alpha * (orig_frames - 1)))
        if gt_frames is not None and nearest_orig_idx < len(gt_frames):
            discrete_frame_rgb = gt_frames[nearest_orig_idx]
        else:
            # Reconstruct discrete frame
            t_discrete = float(-1.0 + 2.0 * nearest_orig_idx / (orig_frames - 1))
            disc_rgb = reconstruct_video_frame(model, t_discrete, out_h, out_w, torch_device)
            discrete_frame_rgb = (disc_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)

        canvas = np.full((comp_h, comp_w, 3), 20, dtype=np.uint8)
        canvas[banner_h:, 0:out_w] = discrete_frame_rgb
        canvas[banner_h:, out_w:2*out_w] = siren_frame_np

        # Vertical divider line
        canvas[banner_h:, out_w-2:out_w] = (200, 200, 200)

        # Annotate with PIL
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)

        draw.text((out_w // 2, 25), f"Traditional Discrete Playback ({fps_multiplier:.0f}x Frame Repeat / Stutter)", fill=(220, 100, 100), font=font, anchor="mm")
        draw.text((out_w + out_w // 2, 25), f"SIREN-ZIP Continuous Time f_theta(x, y, t = {t_val:+.4f})", fill=(100, 240, 140), font=font, anchor="mm")

        annotated_bgr = cv2.cvtColor(np.array(pil_canvas, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        writer_comp.write(annotated_bgr)

    writer_slow.release()
    writer_comp.release()

    print(f"\n[SUCCESS] Continuous slow-motion video saved: '{output_path}'")
    print(f"[SUCCESS] Side-by-side temporal comparison saved: '{comparison_output_path}'\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate Continuous Time Slow-Motion.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_video_siren.pth", help="Video SIREN checkpoint")
    parser.add_argument("--fps_multiplier", type=float, default=4.0, help="Temporal super-sampling multiplier (e.g. 2.0, 4.0, 8.0)")
    parser.add_argument("--output", type=str, default="runs/slow_motion_4x.mp4", help="Output slow-motion video path")
    parser.add_argument("--comparison", type=str, default="runs/slow_motion_comparison.mp4", help="Comparison video path")
    parser.add_argument("--height", type=int, default=None, help="Render height")
    parser.add_argument("--width", type=int, default=None, help="Render width")
    parser.add_argument("--ground_truth", type=str, default="Short_Clip_720p.mp4", help="Ground truth video for discrete baseline")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()
    generate_slow_motion_video(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        comparison_output_path=args.comparison,
        fps_multiplier=args.fps_multiplier,
        render_height=args.height,
        render_width=args.width,
        ground_truth_path=args.ground_truth if os.path.exists(args.ground_truth) else None,
        device=args.device,
    )
