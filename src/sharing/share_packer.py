"""WhatsApp, Telegram, Discord, and Cold-Storage Share Packager."""

from __future__ import annotations

import os
import zipfile
from typing import Any, Dict, Optional

from src.container.neura_v2_reader import NeuraV2Reader


class SharePacker:
    """Packages .neura cinema files into standalone bundles with platform verification."""

    PLATFORM_LIMITS_MB = {
        "whatsapp": 16.0,
        "whatsapp_doc": 2048.0,
        "discord": 25.0,
        "telegram": 2048.0,
        "cold_storage": 100_000.0,
    }

    @classmethod
    def check_platform_compliance(cls, file_size_bytes: int, platform: str = "whatsapp") -> Dict[str, Any]:
        """Verify if file size meets target platform upload limits."""
        p = platform.lower()
        limit_mb = cls.PLATFORM_LIMITS_MB.get(p, 16.0)
        size_mb = file_size_bytes / (1024.0 * 1024.0)

        is_eligible = size_mb <= limit_mb
        margin_mb = limit_mb - size_mb

        return {
            "platform": platform,
            "file_size_mb": size_mb,
            "limit_mb": limit_mb,
            "is_eligible": is_eligible,
            "margin_mb": margin_mb,
            "status": "APPROVED (Ready for instant sending)" if is_eligible else "OVERSIZE",
        }

    @classmethod
    def create_share_bundle(
        cls,
        neura_path: str,
        output_zip_path: str,
        platform: str = "whatsapp",
        subtitle_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create standalone ZIP bundle with 1-click launcher for recipient."""
        if not os.path.exists(neura_path):
            raise FileNotFoundError(f".neura file not found: {neura_path}")

        file_size = os.path.getsize(neura_path)
        compliance = cls.check_platform_compliance(file_size, platform)

        reader = NeuraV2Reader(neura_path)
        header = reader.header
        reader.close()

        neura_basename = os.path.basename(neura_path)

        # 1-click launcher batch script for Windows recipient
        launcher_bat = f"""@echo off
title Siren-VLC Player
echo Starting Siren-VLC Player for {neura_basename}...
python -m src.ui.main_window --file "{neura_basename}"
if errorlevel 1 (
    python scripts/launch_vlc.py --file "{neura_basename}"
)
pause
"""

        # Information manifest for recipient
        manifest_txt = f"""=======================================================
⚡ SIREN-ZIP CINEMA CONTAINER BUNDLE
=======================================================
Container File    : {neura_basename}
Container Size    : {compliance['file_size_mb']:.2f} MB
Platform Target   : {platform.upper()} (Limit: {compliance['limit_mb']:.1f} MB)
Status            : {compliance['status']}

Movie Information:
- Native Res     : {header.native_width}x{header.native_height} Full HD
- Total Duration : {header.total_duration:.2f} seconds
- Neural GOPs    : {header.total_chunks} Chunks
- Audio Track    : {header.audio_channels} channels @ {header.audio_sample_rate} Hz
- Color Space    : {'Rec.2020 HDR' if header.color_primaries == 9 else 'Rec.709 SDR'}

How to Play:
1. Ensure Python 3.10+ and requirements (pip install -r requirements.txt) are installed.
2. Double-click 'play_movie.bat' or run:
   python scripts/launch_vlc.py --file "{neura_basename}"
=======================================================
"""

        os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)) or ".", exist_ok=True)

        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(neura_path, arcname=neura_basename)
            zf.writestr("play_movie.bat", launcher_bat)
            zf.writestr("README_MOVIE.txt", manifest_txt)
            if subtitle_path and os.path.exists(subtitle_path):
                zf.write(subtitle_path, arcname=os.path.basename(subtitle_path))

        bundle_size = os.path.getsize(output_zip_path)

        return {
            "bundle_path": output_zip_path,
            "bundle_size_mb": bundle_size / (1024.0 * 1024.0),
            "compliance": compliance,
            "included_subtitles": bool(subtitle_path),
        }
