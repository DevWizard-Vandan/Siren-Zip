"""Universal Cinema Demuxer for MKV, MP4, WebM, AVI, and MOV with Multi-Track Audio & HDR Extraction."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import av
import cv2


class AudioTrackInfo(NamedTuple):
    index: int
    codec: str
    channels: int
    sample_rate: int
    title: str
    language: str


class HDRMetadataInfo(NamedTuple):
    is_hdr: bool
    color_primaries: str
    color_transfer: str
    color_space: str
    max_cll: int = 1000
    max_fall: int = 400


class UniversalDemuxResult(NamedTuple):
    filepath: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    audio_tracks: List[AudioTrackInfo]
    hdr_info: HDRMetadataInfo
    extracted_audio_path: Optional[str] = None
    extracted_subtitle_path: Optional[str] = None


class UniversalDemuxer:
    """Ingests any cinema video file and extracts video, multi-channel spatial audio, and HDR metadata."""

    @staticmethod
    def inspect(filepath: str) -> UniversalDemuxResult:
        """Inspect media file structure and metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Media file not found: {filepath}")

        # Probe with PyAV
        container = av.open(filepath)
        video_stream = container.streams.video[0] if container.streams.video else None

        if video_stream is None:
            container.close()
            raise ValueError(f"No video stream found in: {filepath}")

        width = video_stream.width
        height = video_stream.height
        fps = float(video_stream.average_rate or video_stream.base_rate or 24.0)
        total_frames = int(video_stream.frames or 0)
        duration_sec = float(container.duration / av.time_base) if container.duration else 0.0

        if total_frames == 0 and duration_sec > 0:
            total_frames = int(duration_sec * fps)

        # 1. Parse Audio Tracks
        audio_tracks: List[AudioTrackInfo] = []
        for i, a_stream in enumerate(container.streams.audio):
            codec_name = a_stream.codec_context.name
            channels = a_stream.codec_context.channels or 2
            s_rate = a_stream.codec_context.sample_rate or 48000
            meta = a_stream.metadata or {}
            title = meta.get("title", f"Track {i+1}")
            lang = meta.get("language", "und")
            audio_tracks.append(
                AudioTrackInfo(
                    index=i,
                    codec=codec_name,
                    channels=channels,
                    sample_rate=s_rate,
                    title=title,
                    language=lang,
                )
            )

        # 2. Parse HDR / Color Metadata
        # Color primaries: bt2020, bt709, etc.
        primaries = str(video_stream.codec_context.color_primaries or "bt709")
        transfer = str(video_stream.codec_context.color_trc or "bt709")
        space = str(video_stream.codec_context.colorspace or "bt709")

        is_hdr = "2020" in primaries or "smpte2084" in transfer or "arib-std-b67" in transfer

        hdr_info = HDRMetadataInfo(
            is_hdr=is_hdr,
            color_primaries=primaries,
            color_transfer=transfer,
            color_space=space,
        )

        container.close()

        return UniversalDemuxResult(
            filepath=filepath,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            audio_tracks=audio_tracks,
            hdr_info=hdr_info,
        )

    @staticmethod
    def extract_audio_payload(
        filepath: str,
        output_audio_path: str,
        codec: str = "opus",
        bitrate_kbps: int = 192,
    ) -> bool:
        """Extract and transcode audio to high-efficiency Opus or AAC for container multiplexing."""
        os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)) or ".", exist_ok=True)
        codec_flag = "libopus" if codec == "opus" else "aac"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            filepath,
            "-vn",
            "-c:a",
            codec_flag,
            "-b:a",
            f"{bitrate_kbps}k",
            output_audio_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0 and os.path.exists(output_audio_path)
