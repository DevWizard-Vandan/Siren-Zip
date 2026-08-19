"""Demonstrate continuous sub-pixel analytical zoom vs discrete bitmap interpolation."""

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

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import torch

from src.data.coordinate_dataset import load_image_as_tensor, make_coordinate_grid
from src.model.siren import SirenImage


def load_siren_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[SirenImage, dict]:
    """Load model architecture and weights from .pth checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_subwindow_grid(
    model: SirenImage,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    render_size: int,
    device: torch.device,
    chunk_size: int = 262144,
) -> np.ndarray:
    """Evaluate continuous mathematical function inside arbitrary sub-pixel bounding box."""
    coords_flat = make_coordinate_grid(
        height=render_size,
        width=render_size,
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        device=device,
    )
    num_coords = coords_flat.shape[0]

    preds = []
    for start in range(0, num_coords, chunk_size):
        end = min(start + chunk_size, num_coords)
        batch_rgb = model(coords_flat[start:end])
        preds.append(batch_rgb)

    full_rgb = torch.cat(preds, dim=0).reshape(render_size, render_size, 3)
    rgb_np = (full_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return rgb_np


def crop_discrete_image(
    image_path: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    render_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop original discrete image and scale using Nearest-Neighbor and Bicubic."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Convert [-1, 1] normalized coordinates to pixel coordinates
    px_min = int(((x_min + 1.0) / 2.0) * orig_w)
    px_max = int(((x_max + 1.0) / 2.0) * orig_w)
    py_min = int(((y_min + 1.0) / 2.0) * orig_h)
    py_max = int(((y_max + 1.0) / 2.0) * orig_h)

    # Ensure valid bounding box
    px_min = max(0, min(px_min, orig_w - 2))
    px_max = max(px_min + 1, min(px_max, orig_w))
    py_min = max(0, min(py_min, orig_h - 2))
    py_max = max(py_min + 1, min(py_max, orig_h))

    cropped = img.crop((px_min, py_min, px_max, py_max))

    # 1. Nearest Neighbor (reveals discrete pixel blocks)
    nearest_img = cropped.resize((render_size, render_size), Image.Resampling.NEAREST)
    nearest_np = np.array(nearest_img, dtype=np.uint8)

    # 2. Bicubic (standard image zoom blurring)
    bicubic_img = cropped.resize((render_size, render_size), Image.Resampling.BICUBIC)
    bicubic_np = np.array(bicubic_img, dtype=np.uint8)

    return nearest_np, bicubic_np


def create_zoom_comparison_panel(
    siren_np: np.ndarray,
    bicubic_np: np.ndarray,
    nearest_np: np.ndarray,
    zoom_factor: float,
    save_path: str,
) -> None:
    """Build a publication-quality 3-panel comparison image."""
    h, w, _ = siren_np.shape
    banner_h = 70
    total_w = w * 3
    total_h = h + banner_h

    canvas = np.full((total_h, total_w, 3), 245, dtype=np.uint8)

    # Place the three panels
    canvas[banner_h:, 0:w] = nearest_np
    canvas[banner_h:, w:2*w] = bicubic_np
    canvas[banner_h:, 2*w:3*w] = siren_np

    # Separator vertical lines
    canvas[banner_h:, w-2:w] = (40, 40, 40)
    canvas[banner_h:, 2*w-2:2*w] = (40, 40, 40)

    # Annotate using PIL
    pil_canvas = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_sub = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title

    # Header Titles
    draw.text((w // 2, 22), f"Traditional Pixel Grid ({zoom_factor:.1f}x Zoom)", fill=(180, 40, 40), font=font_title, anchor="mm")
    draw.text((w // 2, 50), "Nearest-Neighbor (Blocky Discrete Macroblocks)", fill=(100, 100, 100), font=font_sub, anchor="mm")

    draw.text((w + w // 2, 22), f"Standard Digital Zoom ({zoom_factor:.1f}x Zoom)", fill=(180, 120, 20), font=font_title, anchor="mm")
    draw.text((w + w // 2, 50), "Bicubic Interpolation (Blurred & Ringing)", fill=(100, 100, 100), font=font_sub, anchor="mm")

    draw.text((2 * w + w // 2, 22), f"SIREN-ZIP Continuous Field ({zoom_factor:.1f}x Zoom)", fill=(20, 140, 60), font=font_title, anchor="mm")
    draw.text((2 * w + w // 2, 50), "Analytical Continuous Calculus: f_theta(x, y)", fill=(20, 140, 60), font=font_sub, anchor="mm")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    pil_canvas.save(save_path, quality=100)
    print(f"[SUCCESS] Saved continuous zoom comparison: '{save_path}'")


def run_continuous_zoom(
    checkpoint_path: str = "checkpoints/best_siren.pth",
    image_path: str = "test_target.png",
    zoom: float = 20.0,
    center: Tuple[float, float] = (0.0, 0.0),
    render_size: int = 1024,
    output_path: str = "runs/continuous_zoom_comparison.png",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Execute continuous sub-pixel analytical evaluation and generate comparison."""
    torch_device = torch.device(device)
    print(f"\n[START] Executing SIREN Continuous Sub-Pixel Zoom Proof...")
    print(f"   Checkpoint        : {checkpoint_path}")
    print(f"   Original Image    : {image_path}")
    print(f"   Zoom Factor       : {zoom:.1f}x")
    print(f"   Center Coordinate : (x={center[0]:.4f}, y={center[1]:.4f})")
    print(f"   Sub-Pixel Grid    : {render_size}x{render_size} ({render_size * render_size:,} analytical evaluations)")
    print(f"   Device            : {torch_device}\n")

    # 1. Load Model
    model, chk = load_siren_from_checkpoint(checkpoint_path, torch_device)
    print(f"[INFO] Model loaded (Trained Epoch {chk.get('epoch', 'N/A')}, PSNR: {chk.get('psnr', 0.0):.2f} dB)")

    # 2. Compute sub-window bounds in [-1, 1]
    half_w = 1.0 / zoom
    half_h = 1.0 / zoom
    x_min = max(-1.0, center[0] - half_w)
    x_max = min(1.0, center[0] + half_w)
    y_min = max(-1.0, center[1] - half_h)
    y_max = min(1.0, center[1] + half_h)

    print(f"[INFO] Sub-Window Domain: X in [{x_min:.4f}, {x_max:.4f}], Y in [{y_min:.4f}, {y_max:.4f}]")

    # 3. Analytical Evaluation of SIREN continuous field
    print(f"[INFO] Calculating analytical continuous curve f_theta(x, y)...")
    siren_np = evaluate_subwindow_grid(
        model=model,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        render_size=render_size,
        device=torch_device,
    )

    # 4. Discrete cropping & standard interpolations
    print(f"[INFO] Cropping baseline discrete bitmap...")
    nearest_np, bicubic_np = crop_discrete_image(
        image_path=image_path,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        render_size=render_size,
    )

    # 5. Build comparison panel
    create_zoom_comparison_panel(
        siren_np=siren_np,
        bicubic_np=bicubic_np,
        nearest_np=nearest_np,
        zoom_factor=zoom,
        save_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Sub-Pixel Zoom Proof for SIREN-Zip.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_siren.pth", help="Path to trained SIREN checkpoint")
    parser.add_argument("--image_path", type=str, default="test_target.png", help="Path to original ground-truth image")
    parser.add_argument("--zoom", type=float, default=20.0, help="Zoom factor (e.g. 4.0, 20.0, 100.0)")
    parser.add_argument("--center", type=float, nargs=2, default=[0.0, 0.0], help="Center (x, y) coordinates in [-1, 1]")
    parser.add_argument("--render_size", type=int, default=1024, help="Rendered resolution (default: 1024)")
    parser.add_argument("--output", type=str, default="runs/continuous_zoom_comparison.png", help="Comparison image output path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")

    args = parser.parse_args()
    run_continuous_zoom(
        checkpoint_path=args.checkpoint,
        image_path=args.image_path,
        zoom=args.zoom,
        center=tuple(args.center),
        render_size=args.render_size,
        output_path=args.output,
        device=args.device,
    )
