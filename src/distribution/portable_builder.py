"""Standalone Portable Distribution Builder with SHA-256 Checksum Verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from typing import Any, Dict, List, Optional

from src.container.neura_v2_reader import NeuraV2Reader


class PortableBuilder:
    """Packages Siren-Zip cinema containers into 1-click standalone portable bundles."""

    @staticmethod
    def calculate_sha256(filepath: str) -> str:
        """Calculate SHA-256 hash of a file for integrity verification."""
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    @classmethod
    def build_portable_package(
        cls,
        neura_path: str,
        output_dir: str = "dist",
        include_source_tree: bool = True,
        subtitle_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble self-contained portable distribution directory and ZIP archive."""
        if not os.path.exists(neura_path):
            raise FileNotFoundError(f".neura file not found: {neura_path}")

        os.makedirs(output_dir, exist_ok=True)
        bundle_dir = os.path.join(output_dir, "SirenZip-Portable")
        if os.path.exists(bundle_dir):
            shutil.rmtree(bundle_dir)
        os.makedirs(bundle_dir, exist_ok=True)

        reader = NeuraV2Reader(neura_path)
        header = reader.header
        reader.close()

        neura_name = os.path.basename(neura_path)
        dest_neura = os.path.join(bundle_dir, neura_name)
        shutil.copy2(neura_path, dest_neura)

        neura_hash = cls.calculate_sha256(dest_neura)

        # Copy source modules so recipient has full playback engine
        if include_source_tree:
            src_dest = os.path.join(bundle_dir, "src")
            shutil.copytree("src", src_dest)
            if os.path.exists("requirements.txt"):
                shutil.copy2("requirements.txt", os.path.join(bundle_dir, "requirements.txt"))

        # Copy Subtitles if provided
        if subtitle_path and os.path.exists(subtitle_path):
            shutil.copy2(subtitle_path, os.path.join(bundle_dir, os.path.basename(subtitle_path)))

        # 1-Click Windows Launcher Batch Script
        launcher_bat = f"""@echo off
title Siren-VLC Portable Cinema Player
echo =======================================================
echo ⚡ SIREN-ZIP PORTABLE MEDIA PLAYER
echo Playing Container: {neura_name}
echo =======================================================
python -m src.ui.main_window --file "{neura_name}"
if errorlevel 1 (
    echo.
    echo [ERROR] Python environment not detected or missing dependencies.
    echo Please install dependencies: pip install -r requirements.txt
    echo.
    pause
)
"""
        with open(os.path.join(bundle_dir, "play_movie.bat"), "w", encoding="utf-8") as f:
            f.write(launcher_bat)

        # 1-Click Linux / macOS Launcher Shell Script
        launcher_sh = f"""#!/usr/bin/env bash
echo "======================================================="
echo "⚡ SIREN-ZIP PORTABLE MEDIA PLAYER"
echo "Playing Container: {neura_name}"
echo "======================================================="
python3 -m src.ui.main_window --file "{neura_name}"
"""
        with open(os.path.join(bundle_dir, "play_movie.sh"), "w", encoding="utf-8") as f:
            f.write(launcher_sh)

        # Integrity Check & Manifest
        manifest = {
            "application": "Siren-Zip Universal Cinema Codec",
            "version": "2.0.0",
            "container_filename": neura_name,
            "container_sha256": neura_hash,
            "native_resolution": f"{header.native_width}x{header.native_height}",
            "duration_seconds": header.total_duration,
            "neural_gop_chunks": header.total_chunks,
            "audio_track": f"{header.audio_channels}ch @ {header.audio_sample_rate}Hz",
            "color_primaries": "Rec.2020 HDR" if header.color_primaries == 9 else "Rec.709 SDR",
        }

        with open(os.path.join(bundle_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        # Build ZIP archive for WhatsApp / Cloud sharing
        zip_output_path = os.path.join(output_dir, "SirenZip-Portable.zip")
        with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(bundle_dir):
                for file in files:
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, bundle_dir)
                    zf.write(full_p, arcname=os.path.join("SirenZip-Portable", rel_p))

        zip_size_mb = os.path.getsize(zip_output_path) / (1024.0 * 1024.0)

        return {
            "bundle_dir": bundle_dir,
            "zip_path": zip_output_path,
            "zip_size_mb": zip_size_mb,
            "manifest": manifest,
            "sha256": neura_hash,
        }
