"""CLI Tool to build standalone portable distribution bundle."""

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

from src.distribution.portable_builder import PortableBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Standalone Portable Distribution.")
    parser.add_argument("--neura", type=str, default="cinema_full.neura", help="Path to .neura container")
    parser.add_argument("--output", type=str, default="dist", help="Output distribution folder")
    parser.add_argument("--subtitles", type=str, default=None, help="Optional subtitle file")

    args = parser.parse_args()

    print(f"\n=======================================================", flush=True)
    print(f"📦 SIREN-ZIP: Standalone Portable Distribution Builder", flush=True)
    print(f"   Source Container : {args.neura}", flush=True)
    print(f"   Destination      : {args.output}/SirenZip-Portable.zip", flush=True)
    print(f"=======================================================\n", flush=True)

    res = PortableBuilder.build_portable_package(
        neura_path=args.neura,
        output_dir=args.output,
        subtitle_path=args.subtitles,
    )

    print(f"=========================================================================================", flush=True)
    print(f"🎉 STANDALONE PORTABLE BUNDLE BUILT SUCCESSFULLY", flush=True)
    print(f"-----------------------------------------------------------------------------------------", flush=True)
    print(f"   Folder Directory : {res['bundle_dir']}", flush=True)
    print(f"   Portable Archive : {res['zip_path']}", flush=True)
    print(f"   Archive Size     : {res['zip_size_mb']:.2f} MB  (Ready to send over WhatsApp/Telegram)", flush=True)
    print(f"   Container SHA-256: {res['sha256']}", flush=True)
    print(f"   1-Click Launcher : play_movie.bat (Windows) & play_movie.sh (Linux/macOS)", flush=True)
    print(f"=========================================================================================\n", flush=True)


if __name__ == "__main__":
    main()
