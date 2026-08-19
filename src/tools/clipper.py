"""One-Click GIF and WebM Video Clipper for WhatsApp, Discord, and Social Sharing."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional


class VideoClipper:
    """Clips video between Point A and Point B and exports compressed GIF or WebM."""

    @staticmethod
    def clip_to_gif(
        input_video: str,
        output_gif: str,
        start_sec: float,
        end_sec: float,
        fps: int = 15,
        width: int = 480,
    ) -> Dict[str, Any]:
        """Export high-quality 256-color palette-optimized GIF."""
        duration = max(0.5, end_sec - start_sec)
        os.makedirs(os.path.dirname(os.path.abspath(output_gif)) or ".", exist_ok=True)

        # 2-pass palettegen filter for crisp GIF export
        vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            input_video,
            "-vf",
            vf,
            output_gif,
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        size_kb = os.path.getsize(output_gif) / 1024.0

        return {
            "output_path": output_gif,
            "format": "gif",
            "size_kb": size_kb,
            "duration": duration,
        }

    @staticmethod
    def clip_to_webm(
        input_video: str,
        output_webm: str,
        start_sec: float,
        end_sec: float,
        width: int = 720,
    ) -> Dict[str, Any]:
        """Export highly compressed WebM (VP9 + Opus) optimized for Discord and WhatsApp."""
        duration = max(0.5, end_sec - start_sec)
        os.makedirs(os.path.dirname(os.path.abspath(output_webm)) or ".", exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            input_video,
            "-vf",
            f"scale={width}:-1",
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-c:a",
            "libopus",
            "-b:a",
            "96k",
            output_webm,
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        size_kb = os.path.getsize(output_webm) / 1024.0

        return {
            "output_path": output_webm,
            "format": "webm",
            "size_kb": size_kb,
            "duration": duration,
        }
