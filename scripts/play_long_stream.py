"""Continuous Multi-Chunk Cinema Stream Player with Interactive Controls."""

from __future__ import annotations

import argparse
import os
import sys
import time

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

from src.player.engine import ViewportBounds
from src.streaming.stream_engine import StreamEngine


def play_stream(
    neura_path: str,
    width: int = 640,
    height: int = 360,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Continuous playback loop with interactive OpenCV UI."""
    torch_device = torch.device(device)
    engine = StreamEngine(neura_path=neura_path, device=torch_device)
    header = engine.header

    print(f"\n=======================================================", flush=True)
    print(f"🎬 SIREN-ZIP 2.0 Cinema Stream Player", flush=True)
    print(f"   Container : {neura_path} ({os.path.getsize(neura_path)/1024:.1f} KB)", flush=True)
    print(f"   Duration  : {header.total_duration:.2f}s ({header.total_chunks} GOP Chunks)", flush=True)
    print(f"   Controls  : [Space] Play/Pause | [A/D] Seek -5s/+5s | [W/S] Zoom | [1-5] Speed | [Q] Quit", flush=True)
    print(f"=======================================================\n", flush=True)

    window_name = "Siren-Zip 2.0 Cinema Stream Player"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    t_global = 0.0
    speed = 1.0
    is_playing = True
    zoom = 1.0

    fps_count = 0
    fps_start = time.perf_counter()
    measured_fps = 0.0

    while True:
        frame_start = time.perf_counter()

        half_w = 1.0 / zoom
        half_h = 1.0 / zoom
        viewport = ViewportBounds(x_min=-half_w, x_max=half_w, y_min=-half_h, y_max=half_h)

        res = engine.render_at_time(
            t_global=t_global,
            viewport=viewport,
            render_width=width,
            render_height=height,
        )

        # Convert RGB to BGR for OpenCV display
        bgr_frame = cv2.cvtColor(res.rgb_numpy, cv2.COLOR_RGB2BGR)

        # Measure FPS
        fps_count += 1
        elapsed_fps = time.perf_counter() - fps_start
        if elapsed_fps >= 0.5:
            measured_fps = fps_count / elapsed_fps
            fps_count = 0
            fps_start = time.perf_counter()

        # Draw HUD overlay
        hud_line1 = f"Time: {t_global:6.2f}s / {header.total_duration:6.2f}s | Chunk [{res.chunk_idx+1:02d}/{header.total_chunks:02d}]"
        hud_line2 = f"FPS: {measured_fps:4.1f} | Latency: {res.eval_time_ms:4.1f}ms | Paging: {res.paging_time_ms:4.1f}ms | Zoom: {zoom:.1f}x ({res.culling_saved_pct:.0f}% culled)"

        cv2.putText(bgr_frame, hud_line1, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(bgr_frame, hud_line1, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 118), 1)

        cv2.putText(bgr_frame, hud_line2, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(bgr_frame, hud_line2, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (88, 166, 255), 1)

        # Timeline progress bar at bottom
        progress = min(1.0, max(0.0, t_global / max(0.001, header.total_duration)))
        bar_w = int(progress * width)
        cv2.rectangle(bgr_frame, (0, height - 6), (width, height), (30, 30, 30), -1)
        cv2.rectangle(bgr_frame, (0, height - 6), (bar_w, height), (0, 230, 118), -1)

        cv2.imshow(window_name, bgr_frame)

        # Advance time if playing
        if is_playing:
            dt = (1.0 / max(1.0, header.base_fps)) * speed
            t_global += dt
            if t_global > header.total_duration:
                t_global = 0.0

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == 32:  # Space
            is_playing = not is_playing
        elif key == ord("a"):  # Seek -5s
            t_global = max(0.0, t_global - 5.0)
        elif key == ord("d"):  # Seek +5s
            t_global = min(header.total_duration, t_global + 5.0)
        elif key == ord("w"):  # Zoom in
            zoom = min(400.0, zoom * 1.25)
        elif key == ord("s"):  # Zoom out
            zoom = max(1.0, zoom / 1.25)
        elif key == ord("r"):  # Restart
            t_global = 0.0
        elif key == ord("1"):
            speed = 0.25
        elif key == ord("2"):
            speed = 0.5
        elif key == ord("3"):
            speed = 1.0
        elif key == ord("4"):
            speed = 2.0
        elif key == ord("5"):
            speed = 4.0

    cv2.destroyAllWindows()
    engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous Stream Player.")
    parser.add_argument("--neura", type=str, required=True, help="Path to .neura 2.0 container")
    parser.add_argument("--width", type=int, default=640, help="Window width")
    parser.add_argument("--height", type=int, default=360, help="Window height")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()
    play_stream(
        neura_path=args.neura,
        width=args.width,
        height=args.height,
        device=args.device,
    )


if __name__ == "__main__":
    main()
