"""Shared video metadata extractor with yt-dlp and oEmbed fallback."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

try:
    import yt_dlp  # type: ignore
except ImportError:
    yt_dlp = None


def extract_video_summary(video_link: str, timeout: int = 8) -> dict[str, Any]:
    """Extract pitch video title and description without downloading video bytes."""
    if not video_link or not isinstance(video_link, str):
        return {"success": False, "error": "video_link is required", "video_summary": {}}

    cleaned = video_link.strip()
    if yt_dlp is not None:
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
            "force_generic_extractor": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(cleaned, download=False)
                return {
                    "success": True,
                    "video_summary": {
                        "title": info.get("title") or "N/A",
                        "description": info.get("description") or "N/A",
                    },
                }
        except Exception as exc:
            last_error = str(exc)
    else:
        last_error = "yt-dlp not installed"

    # Fallback: YouTube oEmbed works cleanly without API keys or yt-dlp
    if "youtube.com" in cleaned.lower() or "youtu.be" in cleaned.lower():
        try:
            oembed_url = f"https://www.youtube.com/oembed?url={quote(cleaned, safe='')}&format=json"
            resp = requests.get(oembed_url, timeout=timeout)
            if resp.ok:
                data = resp.json()
                return {
                    "success": True,
                    "video_summary": {
                        "title": data.get("title") or "YouTube Pitch Video",
                        "description": f"Author: {data.get('author_name', 'Unknown')}",
                    },
                }
        except Exception as exc:
            last_error = str(exc)

    return {
        "success": False,
        "error": f"Could not process video link: {last_error}",
        "video_summary": {"title": "N/A", "description": "N/A"},
    }
