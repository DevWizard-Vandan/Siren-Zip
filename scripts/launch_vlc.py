"""Entrypoint to launch the complete Siren-VLC Universal Media Player."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui.main_window import launch_player_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Siren-VLC Media Player.")
    parser.add_argument("--file", type=str, default="cinema_full.neura", help="Path to .neura container")
    parser.add_argument("--baseline", type=str, default="Movie_Trailer_1080p.mp4", help="Path to baseline MP4 video")
    args = parser.parse_args()

    neura_file = args.file if args.file and os.path.exists(args.file) else None
    baseline_file = args.baseline if args.baseline and os.path.exists(args.baseline) else None

    launch_player_app(neura_path=neura_file, baseline_path=baseline_file)


if __name__ == "__main__":
    main()
