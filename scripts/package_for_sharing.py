"""CLI Tool to package and verify .neura cinema files for WhatsApp / Cold Storage."""

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

from src.sharing.share_packer import SharePacker


def main() -> None:
    parser = argparse.ArgumentParser(description="Package .neura files for WhatsApp/Telegram/Cold Storage.")
    parser.add_argument("--neura", type=str, required=True, help="Path to .neura file")
    parser.add_argument("--platform", type=str, default="whatsapp", choices=["whatsapp", "discord", "telegram", "cold_storage"], help="Target sharing platform")
    parser.add_argument("--output", type=str, default=None, help="Output ZIP bundle path")
    parser.add_argument("--subtitles", type=str, default=None, help="Optional SRT/VTT subtitles")

    args = parser.parse_args()
    if not os.path.exists(args.neura):
        print(f"❌ Error: File not found: {args.neura}")
        sys.exit(1)

    out_zip = args.output or f"{os.path.splitext(args.neura)[0]}_{args.platform}_bundle.zip"

    print(f"\n=======================================================", flush=True)
    print(f"📦 SIREN-ZIP: Cold-Storage & WhatsApp Share Packager", flush=True)
    print(f"   Input Container  : {args.neura}", flush=True)
    print(f"   Target Platform  : {args.platform.upper()}", flush=True)
    print(f"   Output Bundle    : {out_zip}", flush=True)
    print(f"=======================================================\n", flush=True)

    res = SharePacker.create_share_bundle(
        neura_path=args.neura,
        output_zip_path=out_zip,
        platform=args.platform,
        subtitle_path=args.subtitles,
    )

    comp = res["compliance"]
    print(f"📊 Compliance Results:", flush=True)
    print(f"   Container Size   : {comp['file_size_mb']:.2f} MB", flush=True)
    print(f"   Platform Limit   : {comp['limit_mb']:.1f} MB", flush=True)
    print(f"   Margin Available : {comp['margin_mb']:.2f} MB", flush=True)
    print(f"   Status           : {comp['status']}", flush=True)
    print(f"   Final ZIP Size   : {res['bundle_size_mb']:.2f} MB\n", flush=True)
    print(f"🎉 Bundle is ready to send to your friend on {args.platform.upper()}!", flush=True)


if __name__ == "__main__":
    main()
