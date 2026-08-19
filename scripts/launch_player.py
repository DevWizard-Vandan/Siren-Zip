"""Entrypoint CLI to launch the Siren Player Desktop Application."""

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

from src.ui.main_window import launch_player_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Interactive Siren Player Desktop GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        default="my_video.neura" if os.path.exists("my_video.neura") else None,
        help="Path to .neura video container",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="Short_Clip_720p.mp4" if os.path.exists("Short_Clip_720p.mp4") else None,
        help="Path to baseline reference video (e.g. MP4) for split comparison",
    )

    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"🚀 Launching Siren Player Desktop Application")
    print(f"   .neura Container : {args.file or 'None (Select via GUI)'}")
    print(f"   Baseline Video   : {args.baseline or 'None (Select via GUI)'}")
    print(f"=======================================================\n")

    launch_player_app(
        neura_path=args.file,
        baseline_path=args.baseline,
    )


if __name__ == "__main__":
    main()
