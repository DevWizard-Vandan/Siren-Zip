"""Audio Stream Extractor and Video Color Metadata Probe using PyAV and FFmpeg."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from typing import Any, Dict, NamedTuple, Optional

import av


class AudioExtractionResult(NamedTuple):
    audio_bytes: bytes
    codec_name: str
    codec_id: int  # 0=None, 1=AAC, 2=Opus, 3=MP3, 4=PCM
    sample_rate: int
    channels: int
    duration: float
    bitrate: int


AUDIO_CODEC_MAP = {
    "none": 0,
    "aac": 1,
    "opus": 2,
    "mp3": 3,
    "pcm_s16le": 4,
    "pcm": 4,
    "wav": 4,
}

AUDIO_CODEC_MAP_REV = {
    0: "none",
    1: "aac",
    2: "opus",
    3: "mp3",
    4: "pcm_s16le",
}


class AudioExtractor:
    """Extracts audio streams and probes HDR color metadata from cinema video files."""

    @staticmethod
    def extract_audio(
        video_path: str,
        target_codec: str = "aac",
        bitrate: int = 192_000,
    ) -> AudioExtractionResult:
        """Extract primary audio track from video file into memory buffer.

        Args:
            video_path: Path to video file.
            target_codec: Target compressed format ('aac', 'opus', 'mp3', or 'wav').
            bitrate: Target bitrate in bps (default: 192 kbps).

        Returns:
            AudioExtractionResult with audio payload bytes and stream descriptors.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            container = av.open(video_path)
            audio_streams = container.streams.audio

            if len(audio_streams) == 0:
                container.close()
                return AudioExtractionResult(
                    audio_bytes=b"",
                    codec_name="none",
                    codec_id=0,
                    sample_rate=48000,
                    channels=2,
                    duration=0.0,
                    bitrate=0,
                )

            src_audio = audio_streams[0]
            sample_rate = src_audio.rate or 48000
            channels = src_audio.channels or 2
            duration_sec = float(src_audio.duration * src_audio.time_base) if src_audio.duration else 0.0
            container.close()

            # Transcode/Remux audio cleanly using ffmpeg or PyAV into memory
            ext = "aac" if target_codec == "aac" else ("opus" if target_codec == "opus" else "mp3")
            codec_id = AUDIO_CODEC_MAP.get(target_codec, 1)

            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_out:
                tmp_out_path = tmp_out.name

            # Extract via ffmpeg CLI with high fidelity
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-c:a",
                target_codec,
                "-b:a",
                str(bitrate),
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                tmp_out_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            with open(tmp_out_path, "rb") as f:
                audio_bytes = f.read()

            try:
                os.remove(tmp_out_path)
            except Exception:
                pass

            return AudioExtractionResult(
                audio_bytes=audio_bytes,
                codec_name=target_codec,
                codec_id=codec_id,
                sample_rate=sample_rate,
                channels=channels,
                duration=duration_sec,
                bitrate=bitrate,
            )

        except Exception as e:
            # Fallback if PyAV or ffmpeg fails on audio extraction
            return AudioExtractionResult(
                audio_bytes=b"",
                codec_name="none",
                codec_id=0,
                sample_rate=48000,
                channels=2,
                duration=0.0,
                bitrate=0,
            )

    @staticmethod
    def probe_color_metadata(video_path: str) -> Dict[str, Any]:
        """Inspect video stream for 10-bit HDR10+, Rec.2020, and ST.2084 PQ color characteristics."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        meta: Dict[str, Any] = {
            "color_primaries": 1,  # 1 = BT.709 (SDR), 9 = BT.2020 (HDR)
            "transfer_characteristics": 1,  # 1 = BT.709, 16 = ST.2084 (PQ), 18 = HLG
            "color_space": 1,
            "bits_per_raw_sample": 8,
            "is_hdr": False,
            "peak_nits": 100.0,
        }

        try:
            container = av.open(video_path)
            if len(container.streams.video) > 0:
                v_stream = container.streams.video[0]
                codec_ctx = v_stream.codec_context

                # Extract primaries & transfer characteristics
                primaries = getattr(codec_ctx, "color_primaries", None)
                transfer = getattr(codec_ctx, "color_trc", None)
                pix_fmt = getattr(codec_ctx, "pix_fmt", "")

                if primaries is not None:
                    # 'bt2020' or enum 9
                    if "2020" in str(primaries).lower() or str(primaries) == "9":
                        meta["color_primaries"] = 9
                    else:
                        meta["color_primaries"] = 1

                if transfer is not None:
                    trc_str = str(transfer).lower()
                    if "smpte2084" in trc_str or "st2084" in trc_str or "16" in trc_str or "arib-std-b67" in trc_str:
                        meta["transfer_characteristics"] = 16
                        meta["is_hdr"] = True
                        meta["peak_nits"] = 1000.0
                    elif "hlg" in trc_str or "18" in trc_str:
                        meta["transfer_characteristics"] = 18
                        meta["is_hdr"] = True
                        meta["peak_nits"] = 1000.0

                if "10" in str(pix_fmt) or "p010" in str(pix_fmt) or "yuv420p10" in str(pix_fmt):
                    meta["bits_per_raw_sample"] = 10
                    meta["is_hdr"] = True

            container.close()
        except Exception:
            pass

        return meta
