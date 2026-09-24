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
    parser.add_argument("--file", type=str, default=None, help="Path to .neura container or standard video (.mp4, .mkv)")
    parser.add_argument("--baseline", type=str, default=None, help="Optional path to baseline video for split-screen comparison")
    args = parser.parse_args()

    file_to_load = args.file if args.file and os.path.exists(args.file) else None
    baseline_file = args.baseline if args.baseline and os.path.exists(args.baseline) else None

    # If the user passed a standard video file in --file, launch_player_app will handle it directly
    launch_player_app(neura_path=file_to_load, baseline_path=baseline_file)


if __name__ == "__main__":
    main()
