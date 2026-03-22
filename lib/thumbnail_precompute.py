#!/usr/bin/env python3


import logging
from pathlib import Path
from typing import Dict, Any
from PIL import Image, ImageFile, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("thumbnail_precompute")


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation to an image.
    Falls back gracefully if EXIF data is missing or corrupted.
    """
    try:
        oriented = ImageOps.exif_transpose(img)
        return oriented if oriented is not None else img
    except Exception:
        try:
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            if exif_data and 274 in exif_data:
                orientation = exif_data[274]
                rotations = {
                    3: 180,
                    6: 270,
                    8: 90
                }
                if orientation in rotations:
                    return img.rotate(rotations[orientation], expand=True)
        except Exception:
            pass
        return img


def load_and_orient_image(path: Path) -> Image.Image:
    """Load image from path and apply EXIF orientation if present."""
    img = Image.open(path)
    return _apply_exif_orientation(img)


def make_thumb(src_path: Path, thumb_path: Path, size: int = 256) -> bool:
    """
    Create JPEG thumbnail.
    Returns True if created, False if already exists and valid.
    Raises error if thumb exists but is corrupted or invalid.
    """
    if thumb_path.exists():
        # Verify existing thumb is valid
        try:
            with Image.open(thumb_path) as img:
                # Try to load the image to check it's not corrupted
                img.verify()
            logger.debug("Thumb already exists and is valid: %s", thumb_path)
            return False
        except Exception as e:
            logger.error("Existing thumb is corrupted/invalid: %s (%s)", thumb_path, e)
            raise RuntimeError(f"Corrupted thumb: {thumb_path}") from e

    try:
        im = load_and_orient_image(src_path)
        im = im.convert("RGB")
        im.thumbnail((size, size))
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(thumb_path, "JPEG", quality=85, optimize=True)
        return True
    except Exception as e:
        logger.warning("Failed to create thumb for %s: %s", src_path, e)
        return False


def make_video_thumb(src_path: Path, thumb_path: Path, size: int = 256) -> bool:
    """
    Extract a thumbnail frame from a video file using ffmpeg.
    Returns True if created, False if already exists.
    """
    if thumb_path.exists():
        return False

    import subprocess

    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = f"scale={size}:{size}:force_original_aspect_ratio=decrease"

    def _run(extra_args: list) -> bool:
        result = subprocess.run(
            ["ffmpeg", "-y"] + extra_args + [
                "-i", str(src_path),
                "-vframes", "1",
                "-vf", scale_filter,
                str(thumb_path),
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0 and thumb_path.exists()

    try:
        # Try seeking to 1s first (better representative frame)
        if _run(["-ss", "00:00:01"]):
            return True
        # Fallback: first frame (very short videos)
        if _run([]):
            return True
        logger.warning("ffmpeg failed to extract thumb for %s", src_path)
        return False
    except Exception as e:
        logger.warning("Failed to create video thumb for %s: %s", src_path, e)
        return False


def pick_image_path(meta: Dict[str, Any], images_root: Path) -> Path | None:
    """
    Resolve image path from metadata.
    Prefer relpath if present, fallback to absolute path.
    """
    rel = meta.get("relpath")
    if rel:
        p = (images_root / rel).expanduser()
        return p

    abs_path = meta.get("path")
    if abs_path:
        p = Path(abs_path).expanduser()
        if p.is_absolute():
            return p
        return images_root / abs_path

    return None



