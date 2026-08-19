"""Launch Siren-VLC Media Player with Fused GPU Kernel Acceleration."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui.main_window import launch_player_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Siren-VLC with Fused GPU Acceleration.")
    parser.add_argument("--file", type=str, default="cinema_fast.neura", help="Path to .neura container")
    parser.add_argument("--baseline", type=str, default="Movie_Trailer_1080p.mp4", help="Path to baseline MP4 video")
    args = parser.parse_args()

    neura_file = args.file if args.file and os.path.exists(args.file) else ("cinema_full.neura" if os.path.exists("cinema_full.neura") else None)
    baseline_file = args.baseline if args.baseline and os.path.exists(args.baseline) else None

    print("=" * 80)
    print("[*] LAUNCHING SIREN-VLC (FUSED GPU KERNEL ACCELERATION ENABLED)")
    print("=" * 80)
    if neura_file:
        print(f"* Container File   : {neura_file}")
    if baseline_file:
        print(f"* Baseline Video   : {baseline_file}")
    print("=" * 80)

    launch_player_app(neura_path=neura_file, baseline_path=baseline_file)


if __name__ == "__main__":
    main()
