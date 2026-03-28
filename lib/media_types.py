from __future__ import annotations

from typing import Optional, Set


PHOTO_MIME_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/heic",
    "image/heif",
}

VIDEO_MIME_TYPES: Set[str] = {
    "video/mp4",
}

SUPPORTED_MIME_TYPES: Set[str] = PHOTO_MIME_TYPES | VIDEO_MIME_TYPES

# Ingest pipeline treats videos as "base-only" media (no CLIP image embedding/caption path)
BASE_ONLY_MIME_TYPES: Set[str] = set(VIDEO_MIME_TYPES)

VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
}


def normalize_mime(mime: Optional[str]) -> Optional[str]:
    if not mime:
        return None

    mime_clean = str(mime).strip().lower()
    if not mime_clean:
        return None

    return mime_clean


def is_supported_mime(mime: Optional[str]) -> bool:
    mime_clean = normalize_mime(mime)
    return bool(mime_clean and mime_clean in SUPPORTED_MIME_TYPES)


def is_video_mime(mime: Optional[str]) -> bool:
    mime_clean = normalize_mime(mime)
    if not mime_clean:
        return False
    return mime_clean in VIDEO_MIME_TYPES


def is_base_only_mime(mime: Optional[str]) -> bool:
    mime_clean = normalize_mime(mime)
    return bool(mime_clean and mime_clean in BASE_ONLY_MIME_TYPES)
