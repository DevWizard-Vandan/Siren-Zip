"""Direct-from-Web Video Streamer using yt-dlp."""

from __future__ import annotations

from typing import Any, Dict, Optional


class WebStreamer:
    """Resolves direct playable stream URLs from YouTube, Twitch, Vimeo, and web links."""

    @staticmethod
    def resolve_stream_url(web_url: str) -> Dict[str, Any]:
        """Extract direct progressive video stream URL and metadata using yt-dlp."""
        import yt_dlp

        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(web_url, download=False)
            stream_url = info.get("url") or (info.get("entries", [{}])[0].get("url") if "entries" in info else "")
            title = info.get("title", "Online Stream")
            duration = float(info.get("duration", 0.0) or 0.0)
            thumbnail = info.get("thumbnail", "")

            return {
                "title": title,
                "stream_url": stream_url,
                "duration": duration,
                "thumbnail": thumbnail,
                "webpage_url": web_url,
            }
