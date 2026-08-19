"""Siren-Cast: Live Neural Stream Player & Real-Time Viewer CLI.

Connects to a live Siren-Cast broadcast server (ws://<host>:<port>),
unpacks weight deltas in < 1ms, and renders continuous video live at 60 FPS.
"""

from __future__ import annotations

import argparse
import os
import signal
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
from src.live.stream_client import NeuralStreamClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="⚡ Siren-Cast: Watch Real-Time Live Neural Video Stream (WebSockets)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", type=str, default="ws://localhost:8765", help="WebSocket server URL to watch")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Render device")
    parser.add_argument("--width", type=int, default=640, help="Target render canvas width")
    parser.add_argument("--height", type=int, default=360, help="Target render canvas height")
    parser.add_argument("--headless", action="store_true", help="Run in headless benchmark mode without GUI window")
    parser.add_argument("--max_frames", type=int, default=None, help="Stop after rendering N frames (optional)")

    args = parser.parse_args()

    print("\n==========================================================================")
    print("📺 SIREN-CAST: LIVE NEURAL STREAM VIEWER")
    print("==========================================================================")
    print(f"   Stream URL         : {args.url}")
    print(f"   Render Canvas      : {args.width}x{args.height}")
    print(f"   Device             : {args.device.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Mode               : {'Headless Benchmark' if args.headless else 'Interactive Live Viewer'}")
    print("==========================================================================\n")

    client = NeuralStreamClient(url=args.url, device=args.device)

    def handle_exit(signum, frame):
        print("\n\n[CLIENT] Disconnecting from live stream...", flush=True)
        client.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    client.start()
    print("⏳ Connecting to Siren-Cast stream...")

    # Wait for initial connection & first chunk
    timeout = 15.0
    t0 = time.perf_counter()
    while not client.is_connected or client.active_state_dict is None:
        if time.perf_counter() - t0 > timeout:
            print("[CLIENT] Timeout waiting for stream connection.")
            client.stop()
            sys.exit(1)
        time.sleep(0.1)

    print("🟢 Connected and receiving live neural weights! Starting 60 FPS render loop...\n")

    window_name = f"🔴 Siren-Cast Live Player ({args.url})"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    frames_rendered = 0
    t_start = time.perf_counter()

    try:
        while client.is_running:
            frame_t0 = time.perf_counter()

            # Render continuous frame on GPU
            rgb_frame, tel = client.render_frame(render_width=args.width, render_height=args.height)

            if rgb_frame is not None and not args.headless:
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Draw Sleek Live HUD
                hud_line1 = f"LIVE | FPS: {tel['fps']:.1f} | Bitrate: {tel['bitrate_kbps']:.1f} kbps"
                hud_line2 = f"Eval: {tel['eval_time_ms']:.1f}ms | Page: {tel['paging_time_ms']:.2f}ms | GOP #{tel['chunk_idx']}"

                # Semi-transparent HUD background badge
                cv2.rectangle(bgr_frame, (8, 8), (380, 52), (20, 20, 20), -1)
                cv2.rectangle(bgr_frame, (8, 8), (380, 52), (0, 200, 0) if tel['is_connected'] else (0, 0, 255), 1)

                # Red LIVE indicator dot
                cv2.circle(bgr_frame, (20, 24), 5, (0, 0, 255), -1)
                cv2.putText(bgr_frame, hud_line1, (32, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr_frame, hud_line2, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

                cv2.imshow(window_name, bgr_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break

            frames_rendered += 1
            if args.max_frames and frames_rendered >= args.max_frames:
                break

            # Print periodic terminal telemetry every 60 frames
            if frames_rendered % 60 == 0:
                print(
                    f"🔴 LIVE | Chunk: #{tel.get('chunk_idx', 0):03d} "
                    f"| FPS: {tel.get('fps', 0):4.1f} "
                    f"| Eval: {tel.get('eval_time_ms', 0):4.1f}ms "
                    f"| Page: {tel.get('paging_time_ms', 0):4.2f}ms "
                    f"| Bitrate: {tel.get('bitrate_kbps', 0):5.1f} kbps "
                    f"| Extrapolated: {tel.get('extrapolated', 0)}",
                    flush=True,
                )

            # Target 60 FPS pacing (~16.6 ms per frame)
            frame_elapsed = time.perf_counter() - frame_t0
            sleep_sec = (1.0 / 60.0) - frame_elapsed
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    except KeyboardInterrupt:
        pass
    finally:
        total_time = time.perf_counter() - t_start
        print(f"\n[CLIENT] Session complete: {frames_rendered} frames rendered in {total_time:.2f}s ({frames_rendered/max(0.1, total_time):.1f} avg FPS).")
        client.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
