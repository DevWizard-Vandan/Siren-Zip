"""Generate a 2048x2048 ultra-sharp synthetic test chart with high-frequency harmonics."""

from __future__ import annotations

import argparse
import math
import os
import sys

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_high_res_test_target(
    size: int = 2048,
    save_path: str = "test_target.png",
) -> str:
    """Generate a 2K synthetic test chart containing high-frequency patterns.

    Features:
    - High-frequency concentric chirp rings in the center.
    - 36-ray Siemens starburst pattern.
    - Sharp diagonal, horizontal, and vertical micro-line resolution grids.
    - Multi-scale typography with micro-text.
    - Radial color gradients and dynamic color calibration blocks.
    - Geometric calibration shapes (squares, circles, triangles).
    """
    print(f"[INFO] Generating {size}x{size} ultra-sharp synthetic test chart...")

    # Initialize canvas in uint8 RGB
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # 1. Background subtle radial gradient
    y_coords, x_coords = np.mgrid[0:size, 0:size]
    center_y, center_x = size / 2.0, size / 2.0
    radius = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    max_radius = np.sqrt(2) * (size / 2.0)
    bg_grad = (1.0 - (radius / max_radius) * 0.3) * 240.0
    canvas[:, :, 0] = np.clip(bg_grad * 0.95, 0, 255).astype(np.uint8)
    canvas[:, :, 1] = np.clip(bg_grad * 0.97, 0, 255).astype(np.uint8)
    canvas[:, :, 2] = np.clip(bg_grad * 1.00, 0, 255).astype(np.uint8)

    # 2. Four Corner Radial Gradients
    corner_r = size // 5
    # Top-Left: Cyan-Blue
    for r in range(corner_r, 0, -2):
        color = (int(0), int(180 * (1 - r / corner_r)), int(240 * (1 - r / corner_r)))
        cv2.circle(canvas, (corner_r, corner_r), r, color, -1)

    # Top-Right: Magenta-Purple
    for r in range(corner_r, 0, -2):
        color = (int(220 * (1 - r / corner_r)), int(30 * (1 - r / corner_r)), int(180 * (1 - r / corner_r)))
        cv2.circle(canvas, (size - corner_r, corner_r), r, color, -1)

    # Bottom-Left: Emerald-Green
    for r in range(corner_r, 0, -2):
        color = (int(20 * (1 - r / corner_r)), int(200 * (1 - r / corner_r)), int(120 * (1 - r / corner_r)))
        cv2.circle(canvas, (corner_r, size - corner_r), r, color, -1)

    # Bottom-Right: Amber-Gold
    for r in range(corner_r, 0, -2):
        color = (int(245 * (1 - r / corner_r)), int(160 * (1 - r / corner_r)), int(20 * (1 - r / corner_r)))
        cv2.circle(canvas, (size - corner_r, size - corner_r), r, color, -1)

    # 3. Outer Calibration Border Grid
    border_margin = 40
    cv2.rectangle(canvas, (border_margin, border_margin), (size - border_margin, size - border_margin), (30, 30, 30), 4)
    cv2.rectangle(canvas, (border_margin + 10, border_margin + 10), (size - border_margin - 10, size - border_margin - 10), (80, 80, 80), 2)

    # 4. Multi-frequency Resolution Grids (USAF style line patterns)
    grid_start_y = size // 3
    grid_start_x = size // 8
    for i, line_width in enumerate([1, 2, 3, 5, 8, 12]):
        x_offset = grid_start_x + i * 70
        for l in range(5):
            # Vertical line groups
            cv2.line(canvas, (x_offset + l * line_width * 2, grid_start_y),
                     (x_offset + l * line_width * 2, grid_start_y + 120), (20, 20, 20), line_width)
            # Horizontal line groups
            cv2.line(canvas, (size - grid_start_x - i * 70 - 120, grid_start_y + l * line_width * 2),
                     (size - grid_start_x - i * 70, grid_start_y + l * line_width * 2), (20, 20, 20), line_width)

    # 5. Center Siemens Starburst (36 black/white wedges)
    center_int = (int(center_x), int(center_y))
    star_radius = size // 6
    num_rays = 36
    for k in range(num_rays):
        if k % 2 == 0:
            angle1 = 2 * math.pi * k / num_rays
            angle2 = 2 * math.pi * (k + 1) / num_rays
            pt1 = (int(center_x + star_radius * math.cos(angle1)), int(center_y + star_radius * math.sin(angle1)))
            pt2 = (int(center_x + star_radius * math.cos(angle2)), int(center_y + star_radius * math.sin(angle2)))
            pts = np.array([center_int, pt1, pt2], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], (25, 25, 30))

    # 6. Center Concentric High-Frequency Chirp Rings
    chirp_radius = size // 10
    cv2.circle(canvas, center_int, chirp_radius, (255, 255, 255), -1)
    cv2.circle(canvas, center_int, chirp_radius, (10, 10, 10), 3)

    # Chirp frequency: r^2 spacing
    for ring_r in range(4, chirp_radius - 4, 3):
        val = 0 if (ring_r // 3) % 2 == 0 else 255
        cv2.circle(canvas, center_int, ring_r, (val, val, val), 2)

    # Center bullseye dot
    cv2.circle(canvas, center_int, 4, (230, 40, 40), -1)

    # 7. Convert to PIL Image for high-fidelity typography
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    try:
        font_huge = ImageFont.truetype("arial.ttf", 60)
        font_large = ImageFont.truetype("arial.ttf", 36)
        font_med = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 16)
        font_micro = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font_huge = ImageFont.load_default()
        font_large = font_huge
        font_med = font_huge
        font_small = font_huge
        font_micro = font_huge

    # Header Text
    draw.text((size // 2, 80), "SIREN-ZIP: IMPLICIT NEURAL REPRESENTATION CODEC", fill=(15, 25, 45), font=font_huge, anchor="mm")
    draw.text((size // 2, 140), "CONTINUOUS SINUSOIDAL CALCULUS • SITZMANN ET AL. 2020", fill=(70, 80, 100), font=font_large, anchor="mm")
    draw.text((size // 2, 185), "f_theta(x, y) -> (R, G, B) | Spatial Domain: (x, y) in [-1.0, 1.0]^2", fill=(100, 110, 130), font=font_med, anchor="mm")

    # Center Sub-Headings
    draw.text((size // 2, size // 2 - star_radius - 40), "INFINITE RESOLUTION CONTINUOUS MANIFOLD", fill=(20, 20, 30), font=font_large, anchor="mm")
    draw.text((size // 2, size // 2 + star_radius + 40), "ZERO DISCRETE PIXELS • ZERO MACROBLOCKS • 400X ANALYTICAL ZOOM", fill=(20, 20, 30), font=font_med, anchor="mm")

    # High-Density Micro-Text Test Block (for continuous zoom inspection)
    micro_lines = [
        "1.0x  STANDARD RESOLUTION BENCHMARK",
        "2.0x  SUB-PIXEL CONTINUITY VERIFIED",
        "4.0x  SINUSOIDAL FREQUENCY HARMONICS ACTIVE",
        "8.0x  CONTINUOUS DERIVATIVE PRESERVATION: d/dx sin(w0 x) = w0 cos(w0 x)",
        "16.0x ZERO BLOCKING ARTIFACTS • SMOOTH ANALYTICAL CURVATURE",
        "32.0x SITZMANN INITIALIZATION: W ~ U(-sqrt(6/n)/w0, sqrt(6/n)/w0)",
        "64.0x REPLACING DISCRETE MEMORY WITH CONTINUOUS MATHEMATICS",
        "128x  SIREN-ZIP PROPRIETARY NEURAL CODEC ARCHITECTURE",
    ]

    block_x = size // 2
    block_y_start = size - 340
    for idx, line in enumerate(micro_lines):
        font_to_use = font_small if idx < 4 else font_micro
        draw.text((block_x, block_y_start + idx * 26), line, fill=(30, 35, 45), font=font_to_use, anchor="mm")

    # Footer Metadata
    draw.text((size // 2, size - 70), "SIREN-ZIP v0.1.0 • 2048x2048 ULTRA-SHARP SYNTHETIC BENCHMARK TARGET", fill=(120, 130, 140), font=font_med, anchor="mm")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    pil_img.save(save_path, quality=100)
    print(f"[SUCCESS] Generated synthetic test target: '{save_path}' ({size}x{size} pixels)")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2K SIREN synthetic test target.")
    parser.add_argument("--size", type=int, default=2048, help="Target image resolution (default: 2048)")
    parser.add_argument("--output", type=str, default="test_target.png", help="Output path (default: test_target.png)")
    args = parser.parse_args()

    create_high_res_test_target(size=args.size, save_path=args.output)
