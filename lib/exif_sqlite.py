#!/usr/bin/env python3
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.filename_date_parser import (
    parse_datetime_from_stem,
    parse_date_from_stem,
)


EXIF_DATE_KEYS = (
    "EXIF DateTimeOriginal",
    "EXIF DateTimeDigitized",
    "Image DateTime",
)

EXIFTOOL_DATE_KEYS = (
    "DateTimeOriginal",
    "DateTimeDigitized",
    "CreateDate",
    "MediaCreateDate",
    "TrackCreateDate",
)

def must_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (define it in .env or environment variables).")
    return value


def re_drive_letter(path: str) -> bool:
    return len(path) >= 2 and path[1] == ":" and path[0].isalpha()


def normalize_relpath(path: str) -> Optional[str]:
    if not path:
        return None

    path = path.strip().replace("\\", "/")

    if path.startswith("/") or re_drive_letter(path):
        return None

    posix_path = PurePosixPath(path)
    parts = []
    for part in posix_path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)

    if not parts:
        return None

    return "/".join(parts)


def exif_tag_value(tags: Dict[str, object], key: str) -> Optional[str]:
    value = tags.get(key)
    if value is None:
        return None
    return str(value)


def parse_exif_datetime(tags: Dict[str, object]) -> Optional[int]:
    for key in EXIF_DATE_KEYS:
        raw = exif_tag_value(tags, key)
        if not raw:
            continue
        try:
            dt = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
            # EXIF often has no timezone: assume UTC for consistency with taken_at.
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def parse_exiftool_datetime(raw: object) -> Optional[int]:
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return None

    normalized = text.replace("Z", "+00:00")
    for fmt in (
        "%Y:%m:%d %H:%M:%S.%f%z",
        "%Y:%m:%d %H:%M:%S%z",
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y:%m:%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(normalized, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue

    return None


def parse_datetime_from_filename(path: Path) -> Optional[int]:
    stem = path.stem
    parsed_dt, _pattern = parse_datetime_from_stem(stem)
    if parsed_dt is not None:
        dt = parsed_dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    return None


def ratio_to_float(value: object) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    num = getattr(value, "num", None)
    den = getattr(value, "den", None)
    if num is not None and den not in (None, 0):
        return float(num) / float(den)

    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator not in (None, 0):
        return float(numerator) / float(denominator)

    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def extract_tag_values(tags: Dict[str, object], key: str) -> Tuple[object, ...]:
    tag = tags.get(key)
    if tag is None:
        return ()

    values = getattr(tag, "values", None)
    if values is None:
        return (tag,)

    if isinstance(values, (list, tuple)):
        return tuple(values)

    return (values,)


def dms_to_decimal(dms_values: Tuple[object, ...], ref: Optional[str]) -> Optional[float]:
    if len(dms_values) != 3:
        return None

    degrees = ratio_to_float(dms_values[0])
    minutes = ratio_to_float(dms_values[1])
    seconds = ratio_to_float(dms_values[2])

    if degrees is None or minutes is None or seconds is None:
        return None

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    ref_clean = (ref or "").strip().upper()
    if ref_clean in ("S", "W"):
        decimal = -decimal

    return decimal


def parse_exif_gps(tags: Dict[str, object]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    lat_values = extract_tag_values(tags, "GPS GPSLatitude")
    lon_values = extract_tag_values(tags, "GPS GPSLongitude")

    lat_ref = exif_tag_value(tags, "GPS GPSLatitudeRef")
    lon_ref = exif_tag_value(tags, "GPS GPSLongitudeRef")

    lat = dms_to_decimal(lat_values, lat_ref) if lat_values else None
    lon = dms_to_decimal(lon_values, lon_ref) if lon_values else None

    alt_values = extract_tag_values(tags, "GPS GPSAltitude")
    alt = ratio_to_float(alt_values[0]) if alt_values else None

    alt_ref_raw = exif_tag_value(tags, "GPS GPSAltitudeRef")
    if alt is not None and alt_ref_raw is not None:
        alt_ref_clean = alt_ref_raw.strip()
        if alt_ref_clean in ("1", "b'\\x01'", "\x01"):
            alt = -abs(alt)

    return lat, lon, alt


def read_exiftool_data(image_path: Path) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    try:
        result = subprocess.run(
            ["exiftool", "-j", "-n", "-api", "largefilesupport=1", str(image_path)],
            capture_output=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None, None, None, None

    if result.returncode != 0 or not result.stdout:
        return None, None, None, None

    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError):
        return None, None, None, None

    if not isinstance(payload, list) or not payload:
        return None, None, None, None
    tags = payload[0]
    if not isinstance(tags, dict):
        return None, None, None, None

    taken_at: Optional[int] = None
    for key in EXIFTOOL_DATE_KEYS:
        raw_value = tags.get(key)
        parsed = parse_exiftool_datetime(raw_value)
        if parsed is not None:
            taken_at = parsed
            break

    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None

    try:
        raw_lat = tags.get("GPSLatitude")
        if raw_lat is not None:
            lat = float(raw_lat)
    except (TypeError, ValueError):
        lat = None

    try:
        raw_lon = tags.get("GPSLongitude")
        if raw_lon is not None:
            lon = float(raw_lon)
    except (TypeError, ValueError):
        lon = None

    try:
        raw_alt = tags.get("GPSAltitude")
        if raw_alt is not None:
            alt = float(raw_alt)
    except (TypeError, ValueError):
        alt = None

    return taken_at, lat, lon, alt


def read_exif_data(image_path: Path) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    taken_at: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None

    try:
        import exifread  # type: ignore

        with image_path.open("rb") as handle:
            tags: Dict[str, object] = exifread.process_file(handle, details=False)

        taken_at = parse_exif_datetime(tags)
        lat, lon, alt = parse_exif_gps(tags)
    except Exception:
        pass

    exiftool_taken_at, exiftool_lat, exiftool_lon, exiftool_alt = read_exiftool_data(image_path)

    if taken_at is None:
        taken_at = exiftool_taken_at
    if lat is None:
        lat = exiftool_lat
    if lon is None:
        lon = exiftool_lon
    if alt is None:
        alt = exiftool_alt

    return taken_at, lat, lon, alt


@dataclass
class Stats:
    scanned: int = 0
    updated: int = 0
    missing_file: int = 0
    invalid_path: int = 0
    no_exif_values: int = 0
    conflict_skipped: int = 0
    errors: int = 0


def sqlite_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con



