#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from PIL import Image

import chromadb
import open_clip
import torch
from chromadb.config import Settings
from lib.media_types import (
    BASE_ONLY_MIME_TYPES,
    SUPPORTED_MIME_TYPES,
    is_base_only_mime,
    is_supported_mime,
)

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass


def must_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (define it in .env or environment variables).")
    return value


def step_header(name: str) -> None:
    print(f"\n=== {name} ===")


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def detect_mime(path: Path) -> Optional[str]:
    from lib import mime_sqlite as mimemod

    return mimemod.detect_mime(path)


def read_image_size(path: Path) -> Tuple[int, int]:
    try:
        with Image.open(path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        meta = probe_video_metadata(path)
        return int(meta.get("w") or 0), int(meta.get("h") or 0)


def probe_video_metadata(abs_path: Path) -> Dict[str, Any]:
    import json
    import re
    import subprocess

    def _to_int(value: Any) -> int:
        try:
            return int(float(value))
        except Exception:
            return 0

    def _to_duration(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            d = float(value)
            return d if d > 0 else None
        except Exception:
            pass

        text = str(value).strip().lower()
        if not text:
            return None

        text = text.replace("sec", "s").replace("secs", "s").replace("seconds", "s")

        m = re.match(r"^(\d+(?:\.\d+)?)\s*s$", text)
        if m:
            try:
                d = float(m.group(1))
                return d if d > 0 else None
            except Exception:
                return None

        m = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", text)
        if m:
            try:
                hh = int(m.group(1))
                mm = int(m.group(2))
                ss = float(m.group(3))
                d = hh * 3600 + mm * 60 + ss
                return d if d > 0 else None
            except Exception:
                return None

        return None

    def _build_iso6709(lat: float, lon: float, alt: Optional[float]) -> str:
        lat_str = f"{lat:+.8f}".rstrip("0").rstrip(".")
        lon_str = f"{lon:+.8f}".rstrip("0").rstrip(".")
        if alt is None:
            return f"{lat_str}{lon_str}/"
        alt_str = f"{alt:+.2f}".rstrip("0").rstrip(".")
        return f"{lat_str}{lon_str}{alt_str}/"

    def _parse_datetime_like(value: Any) -> Optional[str]:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None

        # ExifTool often uses "YYYY:MM:DD HH:MM:SS[.sss][Z|+HH:MM]"
        if re.match(r"^\d{4}:\d{2}:\d{2}\s+\d{2}:\d{2}:\d{2}", raw):
            normalized = raw.replace(" ", "T", 1)
            normalized = re.sub(r"^(\d{4}):(\d{2}):(\d{2})T", r"\1-\2-\3T", normalized)
            return normalized

        # Already close to ISO
        return raw

    def _probe_with_exiftool(path: Path) -> Optional[Dict[str, Any]]:
        try:
            result = subprocess.run(
                ["exiftool", "-j", "-n", "-api", "largefilesupport=1", str(path)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0 or not result.stdout:
                return None

            payload = json.loads(result.stdout)
            if not isinstance(payload, list) or not payload:
                return None
            tags = payload[0] if isinstance(payload[0], dict) else {}
            if not isinstance(tags, dict):
                return None

            w = 0
            h = 0
            duration: Optional[float] = None

            for key in ("ImageWidth", "SourceImageWidth", "ExifImageWidth", "Width"):
                w = _to_int(tags.get(key))
                if w > 0:
                    break
            for key in ("ImageHeight", "SourceImageHeight", "ExifImageHeight", "Height"):
                h = _to_int(tags.get(key))
                if h > 0:
                    break

            for key in ("Duration", "MediaDuration", "TrackDuration"):
                duration = _to_duration(tags.get(key))
                if duration is not None:
                    break

            format_tags: Dict[str, Any] = {}
            for key in (
                "CreationTime",
                "MediaCreateDate",
                "TrackCreateDate",
                "CreateDate",
                "DateTimeOriginal",
            ):
                creation_like = _parse_datetime_like(tags.get(key))
                if creation_like:
                    format_tags["creation_time"] = creation_like
                    break

            lat = tags.get("GPSLatitude")
            lon = tags.get("GPSLongitude")
            alt_raw = tags.get("GPSAltitude")
            try:
                gps_lat = float(lat) if lat is not None else None
                gps_lon = float(lon) if lon is not None else None
            except Exception:
                gps_lat, gps_lon = None, None

            gps_alt: Optional[float] = None
            if alt_raw is not None:
                try:
                    gps_alt = float(alt_raw)
                except Exception:
                    gps_alt = None

            if gps_lat is not None and gps_lon is not None:
                format_tags["location"] = _build_iso6709(gps_lat, gps_lon, gps_alt)

            result = {
                "w": w,
                "h": h,
                "duration": duration,
                "format_tags": format_tags,
                "stream_tags": [],
            }
            return result
        except Exception:
            return None

    def _merge_meta(base: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not extra:
            return base

        merged: Dict[str, Any] = {
            "w": int(base.get("w") or 0),
            "h": int(base.get("h") or 0),
            "duration": base.get("duration"),
            "format_tags": dict(base.get("format_tags") or {}),
            "stream_tags": list(base.get("stream_tags") or []),
        }

        if merged["w"] <= 0:
            merged["w"] = int(extra.get("w") or 0)
        if merged["h"] <= 0:
            merged["h"] = int(extra.get("h") or 0)
        if merged["duration"] is None:
            merged["duration"] = extra.get("duration")

        for key, value in (extra.get("format_tags") or {}).items():
            if key not in merged["format_tags"] or not merged["format_tags"][key]:
                merged["format_tags"][key] = value

        extra_stream_tags = extra.get("stream_tags") or []
        if extra_stream_tags:
            merged["stream_tags"].extend(extra_stream_tags)

        return merged

    def _parse_ffprobe(data: Dict[str, Any]) -> Dict[str, Any]:
        stream_tags: List[Dict[str, Any]] = []
        for stream in data.get("streams", []):
            tags = stream.get("tags") or {}
            if tags:
                stream_tags.append(tags)

        out: Dict[str, Any] = {
            "w": 0,
            "h": 0,
            "duration": None,
            "format_tags": data.get("format", {}).get("tags") or {},
            "stream_tags": stream_tags,
        }

        for stream in data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue

            w = int(stream.get("width") or 0)
            h = int(stream.get("height") or 0)

            rotate_raw = None
            for side_data in stream.get("side_data_list", []):
                rotate_raw = side_data.get("rotation")
                if rotate_raw is not None:
                    break
            if rotate_raw is None:
                rotate_raw = (stream.get("tags") or {}).get("rotate")

            try:
                rot = abs(int(rotate_raw or 0))
            except Exception:
                rot = 0
            if rot in (90, 270):
                w, h = h, w

            out["w"] = w
            out["h"] = h

            try:
                stream_duration = float(stream.get("duration") or 0)
                if stream_duration > 0:
                    out["duration"] = stream_duration
            except Exception:
                pass
            break

        if out["duration"] is None:
            try:
                format_duration = float(data.get("format", {}).get("duration") or 0)
                if format_duration > 0:
                    out["duration"] = format_duration
            except Exception:
                pass

        return out

    ffprobe_meta: Optional[Dict[str, Any]] = None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                str(abs_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            ffprobe_meta = _parse_ffprobe(json.loads(result.stdout))
    except Exception:
        pass

    if ffprobe_meta is not None:
        needs_extra = (
            ffprobe_meta.get("duration") is None
            or not _extract_video_location_tag(ffprobe_meta)
            or not (ffprobe_meta.get("format_tags") or {}).get("creation_time")
        )
        if needs_extra:
            return _merge_meta(ffprobe_meta, _probe_with_exiftool(abs_path))
        return ffprobe_meta

    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(abs_path)],
            capture_output=True,
            timeout=30,
        )
        stderr = (result.stderr or b"").decode("utf-8", errors="ignore")

        w, h = 0, 0
        duration: Optional[float] = None

        size_match = re.search(r"\b(\d{2,5})x(\d{2,5})\b", stderr)
        if size_match:
            w = int(size_match.group(1))
            h = int(size_match.group(2))

        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
        if duration_match:
            hh = int(duration_match.group(1))
            mm = int(duration_match.group(2))
            ss = float(duration_match.group(3))
            duration = hh * 3600 + mm * 60 + ss

        ffmpeg_meta = {
            "w": w,
            "h": h,
            "duration": duration,
            "format_tags": {},
            "stream_tags": [],
        }
        needs_extra = duration is None
        if needs_extra:
            return _merge_meta(ffmpeg_meta, _probe_with_exiftool(abs_path))
        return ffmpeg_meta
    except Exception:
        empty_meta = {
            "w": 0,
            "h": 0,
            "duration": None,
            "format_tags": {},
            "stream_tags": [],
        }
        return _merge_meta(empty_meta, _probe_with_exiftool(abs_path))


def _extract_video_location_tag(meta: Dict[str, Any]) -> Optional[str]:
    format_tags = meta.get("format_tags") or {}
    location = format_tags.get("location") or format_tags.get("com.apple.quicktime.location.ISO6709")
    if location:
        return str(location).strip()

    for tags in meta.get("stream_tags") or []:
        location = tags.get("location") or tags.get("com.apple.quicktime.location.ISO6709")
        if location:
            return str(location).strip()

    return None


def _extract_video_creation_time_tag(meta: Dict[str, Any]) -> Optional[str]:
    format_tags = meta.get("format_tags") or {}
    for key in (
        "creation_time",
        "com.apple.quicktime.creationdate",
        "CreationTime",
        "MediaCreateDate",
        "TrackCreateDate",
        "CreateDate",
    ):
        value = format_tags.get(key)
        if value:
            return str(value).strip()

    for tags in meta.get("stream_tags") or []:
        for key in (
            "creation_time",
            "com.apple.quicktime.creationdate",
            "CreationTime",
            "MediaCreateDate",
            "TrackCreateDate",
            "CreateDate",
        ):
            value = tags.get(key)
            if value:
                return str(value).strip()

    return None


def _parse_iso6709_location(location_str: str) -> Optional[Tuple[float, float, Optional[float]]]:
    import re

    match = re.match(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?", location_str)
    if not match:
        return None

    try:
        lat = float(match.group(1))
        lon = float(match.group(2))
    except Exception:
        return None

    alt: Optional[float] = None
    alt_group = match.group(3)
    if alt_group:
        try:
            alt = float(alt_group)
        except Exception:
            alt = None

    return lat, lon, alt


def has_caption(con: sqlite3.Connection, sha1: str) -> bool:
    row = con.execute("SELECT 1 FROM captions WHERE sha1=? LIMIT 1", (sha1,)).fetchone()
    return row is not None


def has_tags(con: sqlite3.Connection, sha1: str) -> bool:
    row = con.execute("SELECT 1 FROM tags WHERE sha1=? LIMIT 1", (sha1,)).fetchone()
    return row is not None


def chroma_has_id(collection: Any, sha1: str) -> bool:
    try:
        got = collection.get(ids=[sha1], include=[])
        ids = got.get("ids", [])
        return bool(ids and ids[0] == sha1)
    except Exception:
        return False


def check_components_complete(
    con: sqlite3.Connection,
    pipeline_ctx: Dict[str, Any],
    sha1: str,
) -> Tuple[bool, List[str]]:
    missing: List[str] = []

    thumb_path = pipeline_ctx["thumb_dir"] / f"{sha1}.jpg"
    if not thumb_path.exists():
        missing.append("thumb")

    if not chroma_has_id(pipeline_ctx["img_col"], sha1):
        missing.append("chroma_image")

    if not chroma_has_id(pipeline_ctx["cap_col"], sha1):
        missing.append("chroma_caption")

    if not has_caption(con, sha1):
        missing.append("caption")

    if not has_tags(con, sha1):
        missing.append("tags")

    return len(missing) == 0, missing


def image_info_matches(
    row: sqlite3.Row,
    relpath: str,
    file_size: int,
    mtime: float,
    width: int,
    height: int,
    mime: Optional[str],
) -> bool:
    row_path = str(row["path"] or "")
    row_size = int(row["file_size"] or 0)
    row_w = int(row["w"] or 0)
    row_h = int(row["h"] or 0)
    row_mime = (row["mime"] or "").strip() if row["mime"] is not None else ""
    in_mime = (mime or "").strip()

    return (
        row_path == relpath
        and row_size == int(file_size)
        and row_w == int(width)
        and row_h == int(height)
        and row_mime == in_mime
    )


def image_info_mismatch_reasons(
    row: Optional[sqlite3.Row],
    relpath: str,
    file_size: int,
    mtime: float,
    width: int,
    height: int,
    mime: Optional[str],
) -> List[str]:
    if row is None:
        return ["missing_row"]

    reasons: List[str] = []
    row_path = str(row["path"] or "")
    row_size = int(row["file_size"] or 0)
    row_w = int(row["w"] or 0)
    row_h = int(row["h"] or 0)
    row_mime = (row["mime"] or "").strip() if row["mime"] is not None else ""
    in_mime = (mime or "").strip()

    if row_path != relpath:
        reasons.append("path")
    if row_size != int(file_size):
        reasons.append("size")
    if row_w != int(width) or row_h != int(height):
        reasons.append("dimensions")
    if row_mime != in_mime:
        reasons.append("mime")

    return reasons


def run_sync_missing_photos_dir(
    photos_dir: Path,
    db_path: Path,
    chroma_dir: Path,
    image_collection: str,
    captions_collection: str,
    device: str,
    dtype: str,
    caption_model: str,
    translate_model: str,
    dry_run: bool,
    summary_only: bool,
    limit: int,
    subdir: str,
    only_mime: str,
) -> None:
    mode_label = " (dry-run)" if dry_run else ""
    step_header(f"Sync missing photos dir -> SQLite + Chroma{mode_label}")

    photos_dir_resolved = photos_dir.resolve()
    scan_root = photos_dir_resolved
    subdir_clean = subdir.strip().replace("\\", "/").strip("/")
    if subdir_clean:
        scan_root = (photos_dir_resolved / subdir_clean).resolve()
        try:
            scan_root.relative_to(photos_dir_resolved)
        except ValueError as exc:
            raise RuntimeError("--sync-subdir must stay within PHOTOAI_PHOTOS_DIR") from exc
        if not scan_root.exists() or not scan_root.is_dir():
            raise RuntimeError(f"--sync-subdir does not exist or is not a directory: {scan_root}")

    only_mime_clean = only_mime.strip().lower()

    if dry_run:
        thumb_dir = Path(os.environ.get("PHOTOAI_THUMB_DIR", "cache/thumbs")).expanduser().resolve()
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        img_col = client.get_or_create_collection(name=image_collection, metadata={"hnsw:space": "cosine"})
        cap_col = client.get_or_create_collection(name=captions_collection)
        pipeline_ctx: Dict[str, Any] = {
            "photos_dir": photos_dir_resolved,
            "thumb_dir": thumb_dir,
            "img_col": img_col,
            "cap_col": cap_col,
        }
        location_ctx = None
    else:
        pipeline_ctx = build_pipeline_context(
            photos_dir=photos_dir_resolved,
            chroma_dir=chroma_dir,
            image_collection=image_collection,
            captions_collection=captions_collection,
            device=device,
            skip_captions=False,
            skip_thumbs=False,
            dtype_name=dtype,
            caption_model=caption_model,
            translate_model=translate_model,
        )

        location_ctx = load_location_context(db_path)
        if location_ctx is None:
            print("[WARN] Location backfill disabled in sync-missing mode: GeoNames data or scipy unavailable.")

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    con.row_factory = sqlite3.Row

    scanned = 0
    done_count = 0
    updated_count = 0
    done_base_count = 0
    skipped_unsupported_count = 0
    error_count = 0
    update_reason_counts: Dict[str, int] = {
        "missing_row": 0,
        "path": 0,
        "size": 0,
        "dimensions": 0,
        "mime": 0,
    }

    try:
        all_files = [p for p in scan_root.rglob("*") if p.is_file()]
        if only_mime_clean:
            all_files = [p for p in all_files if (detect_mime(p) or "").strip().lower() == only_mime_clean]
        if limit > 0:
            all_files = all_files[:limit]
        total_files = len(all_files)
        if not summary_only:
            print(f"[INFO] Files found in PHOTOAI_PHOTOS_DIR: {total_files}")
            if subdir_clean:
                print(f"[INFO] Subdir filter: {subdir_clean}")
            if only_mime_clean:
                print(f"[INFO] MIME filter  : {only_mime_clean}")
        else:
            print(f"[INFO] Quiet mode: scanning {total_files} files")

        for img_path in all_files:
            scanned += 1
            if summary_only and (scanned == 1 or scanned % 500 == 0 or scanned == total_files):
                print(f"[QUIET] visited {scanned}/{total_files}")
            relpath = str(img_path.relative_to(photos_dir_resolved)).replace("\\", "/")

            try:
                st = img_path.stat()
                file_size = int(st.st_size)
                mtime = float(st.st_mtime)

                mime = detect_mime(img_path)

                if not is_supported_mime(mime):
                    skipped_unsupported_count += 1
                    if not summary_only:
                        print(f"[SKIP] {relpath} -> unsupported mime: {mime or 'unknown'}")
                    continue

                sha1 = sha1_file(img_path)
                w, h = read_image_size(img_path)

                if is_base_only_mime(mime):
                    video_meta = probe_video_metadata(img_path)
                    w = int(video_meta.get("w") or w or 0)
                    h = int(video_meta.get("h") or h or 0)
                    video_duration = video_meta.get("duration")
                    if dry_run:
                        done_base_count += 1
                        if not summary_only:
                            print(
                                f"[DRY-RUN] {relpath} -> would_upsert_base_info_extract_taken_at_and_mark_done_base "
                                f"(mime={mime})"
                            )
                    else:
                        upsert_image_row(
                            con=con,
                            sha1=sha1,
                            relpath=relpath,
                            mtime=mtime,
                            w=w,
                            h=h,
                            duration=float(video_duration) if video_duration else None,
                            file_size=file_size,
                            mime=mime,
                        )
                        enrich_video_metadata(con, sha1, img_path)
                        enrich_exif(con, sha1, img_path)  # filename-based taken_at fallback
                        if location_ctx is not None:
                            enrich_location(con, sha1, location_ctx)
                        thumb_path = pipeline_ctx["thumb_dir"] / f"{sha1}.jpg"
                        pipeline_ctx["thumbsmod"].make_video_thumb(img_path, thumb_path, size=pipeline_ctx["thumb_size"])
                        ensure_job_state(con, sha1, "add_all", "done_base", f"base_only_mime:{mime}")
                        con.commit()
                        done_base_count += 1
                    continue

                row = con.execute(
                    "SELECT sha1, path, mtime, w, h, file_size, mime FROM images WHERE sha1=? LIMIT 1",
                    (sha1,),
                ).fetchone()
                mismatch_reasons = image_info_mismatch_reasons(row, relpath, file_size, mtime, w, h, mime)
                same_info = len(mismatch_reasons) == 0

                if row is not None:
                    complete_existing, missing_components = check_components_complete(con, pipeline_ctx, sha1)

                    if dry_run:
                        if not same_info:
                            for reason in mismatch_reasons:
                                if reason in update_reason_counts:
                                    update_reason_counts[reason] += 1
                            if not summary_only:
                                print(
                                    f"[DRY-RUN] {relpath} -> would_update_info_only_existing_sha1 "
                                    f"(reasons={','.join(mismatch_reasons)})"
                                )

                        if not complete_existing:
                            updated_count += 1
                            if not summary_only:
                                print(
                                    f"[DRY-RUN] {relpath} -> would_add_missing_contents_existing_sha1 "
                                    f"(missing={','.join(missing_components)})"
                                )
                        elif not same_info:
                            updated_count += 1
                        else:
                            done_count += 1
                        continue

                    info_updated = False
                    if not same_info:
                        upsert_image_row(
                            con=con,
                            sha1=sha1,
                            relpath=relpath,
                            mtime=mtime,
                            w=w,
                            h=h,
                            duration=0.0,
                            file_size=file_size,
                            mime=mime,
                        )
                        enrich_exif(con, sha1, img_path)
                        if location_ctx is not None:
                            enrich_location(con, sha1, location_ctx)

                        info_updated = True

                    if complete_existing:
                        ensure_job_state(
                            con,
                            sha1,
                            "add_all",
                            "done",
                            "sync_missing_info_updated" if info_updated else "sync_missing_ok",
                        )
                        con.commit()
                        if info_updated:
                            updated_count += 1
                        else:
                            done_count += 1
                    else:
                        sha1_out, status = process_one_image(
                            con=con,
                            img_path=img_path,
                            pipeline_ctx=pipeline_ctx,
                            location_ctx=location_ctx,
                            add_step="add_all",
                            skip_captions=False,
                            skip_thumbs=False,
                            sha1=sha1,
                        )

                        if status != "done":
                            ensure_job_state(con, sha1_out, "add_all", "error", status)
                            con.commit()
                            error_count += 1
                            if not summary_only:
                                print(f"[ERROR] {relpath} -> {status}")
                            continue

                        complete_after, missing_after = check_components_complete(con, pipeline_ctx, sha1)
                        if not complete_after:
                            ensure_job_state(
                                con,
                                sha1,
                                "add_all",
                                "error",
                                "missing:" + ",".join(missing_after),
                            )
                            con.commit()
                            error_count += 1
                            if not summary_only:
                                print(f"[ERROR] {relpath} -> missing {missing_after}")
                            continue

                        ensure_job_state(
                            con,
                            sha1,
                            "add_all",
                            "done",
                            "sync_missing_info_and_contents_added" if info_updated else "sync_missing_contents_added",
                        )
                        con.commit()
                        updated_count += 1
                    continue

                if dry_run:
                    updated_count += 1
                    update_reason_counts["missing_row"] += 1
                    if not summary_only:
                        print(f"[DRY-RUN] {relpath} -> would_insert_and_generate_contents (reasons=missing_row)")
                    continue

                sha1_out, status = process_one_image(
                    con=con,
                    img_path=img_path,
                    pipeline_ctx=pipeline_ctx,
                    location_ctx=location_ctx,
                    add_step="add_all",
                    skip_captions=False,
                    skip_thumbs=False,
                    sha1=sha1,
                )

                if status != "done":
                    ensure_job_state(con, sha1_out, "add_all", "error", status)
                    con.commit()
                    error_count += 1
                    if not summary_only:
                        print(f"[ERROR] {relpath} -> {status}")
                    continue

                complete_after, missing_after = check_components_complete(con, pipeline_ctx, sha1)
                if not complete_after:
                    ensure_job_state(
                        con,
                        sha1,
                        "add_all",
                        "error",
                        "missing:" + ",".join(missing_after),
                    )
                    con.commit()
                    error_count += 1
                    if not summary_only:
                        print(f"[ERROR] {relpath} -> missing {missing_after}")
                    continue

                ensure_job_state(con, sha1, "add_all", "done", "sync_missing_inserted")
                con.commit()
                updated_count += 1

            except Exception as exc:
                error_count += 1
                if not summary_only:
                    print(f"[ERROR] {relpath} -> {exc}")

        print("\nSync-missing summary:")
        print(f"  scanned    : {scanned}")
        print(f"  done       : {done_count}")
        print(f"  updated    : {updated_count}")
        print(f"  done_base  : {done_base_count}")
        print(f"  skipped    : {skipped_unsupported_count} (unsupported mime)")
        print(f"  errors     : {error_count}")
        print(f"  step       : jobs.step='add_all'")
        if dry_run:
            print("  mode       : dry-run (no DB/Chroma/thumb/caption/tag writes)")
            print(
                "  update_reasons: "
                + " ".join(
                    f"{key}={value}"
                    for key, value in update_reason_counts.items()
                )
            )
    finally:
        con.close()


def geonames_available(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    con = sqlite3.connect(str(db_path))
    try:
        has_table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='geonames_cities' LIMIT 1"
        ).fetchone()
        if not has_table:
            return False
        row = con.execute("SELECT COUNT(*) FROM geonames_cities").fetchone()
        return bool(row and int(row[0]) > 0)
    finally:
        con.close()


def run_imgsort(
    src_dir: str,
    dst_dir: Path,
    move: bool,
    recursive: bool,
    max_depth: int,
    prefer_metadata: bool,
    dry_run: bool,
    on_saved: Optional[Any] = None,
) -> None:
    from lib import media_sorter

    step_header("Sort/Copy images (media_sorter)")
    options = media_sorter.Options(
        dir_from=src_dir,
        dir_to=str(dst_dir),
        copy=not move,
        dry_run=dry_run,
        recursive=recursive,
        max_depth=max_depth,
        verbose=False,
        prefer_metadata_on_conflict=prefer_metadata,
        count_extensions=False,
        on_saved=on_saved,
    )

    file_stats = media_sorter.Stats()
    ext_count = media_sorter.ExtensionCount.new()
    media_sorter.visit_dirs(Path(options.dir_from), media_sorter.compute_file, options, file_stats, ext_count, 0)
    file_stats.print_all()


def ensure_job_state(con: sqlite3.Connection, sha1: str, step: str, status: str, detail: str = "") -> None:
    con.execute(
        """
        INSERT INTO jobs (sha1, step, status, detail, updated_at)
        VALUES (?, ?, ?, ?, unixepoch())
        ON CONFLICT(sha1, step) DO UPDATE SET
          status=excluded.status,
          detail=excluded.detail,
          updated_at=excluded.updated_at
        """,
        (sha1, step, status, detail[:500]),
    )


def upsert_image_row(
    con: sqlite3.Connection,
    sha1: str,
    relpath: str,
    mtime: float,
    w: int,
    h: int,
    duration: Optional[float],
    file_size: int,
    mime: Optional[str],
) -> None:
    con.execute(
        """
                INSERT INTO images (sha1, path, mtime, w, h, duration, file_size, mime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sha1) DO UPDATE SET
          path=excluded.path,
          mtime=excluded.mtime,
          w=CASE WHEN excluded.w > 0 THEN excluded.w ELSE images.w END,
          h=CASE WHEN excluded.h > 0 THEN excluded.h ELSE images.h END,
                duration=CASE WHEN excluded.duration IS NOT NULL AND excluded.duration > 0 THEN excluded.duration ELSE images.duration END,
          file_size=excluded.file_size,
          mime=excluded.mime
        """,
                (sha1, relpath, mtime, w, h, duration, file_size, mime),
    )


def enrich_video_metadata(con: sqlite3.Connection, sha1: str, abs_path: Path) -> None:
    """Extract width, height, duration, taken_at and GPS from video via ffprobe."""
    meta = probe_video_metadata(abs_path)
    w = int(meta.get("w") or 0)
    h = int(meta.get("h") or 0)
    duration: Optional[float] = None
    try:
        d = float(meta.get("duration") or 0)
        if d > 0:
            duration = d
    except Exception:
        pass

    # taken_at from creation_time tag
    creation_time_str = _extract_video_creation_time_tag(meta)
    taken_at: Optional[int] = None
    if creation_time_str:
        try:
            dt = datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))
            taken_at = int(dt.timestamp())
        except Exception:
            pass

    # GPS from location tag (ISO 6709: "+lat+lon/" or similar)
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
    location_str = _extract_video_location_tag(meta)
    if location_str:
        parsed_location = _parse_iso6709_location(location_str)
        if parsed_location is not None:
            gps_lat, gps_lon, gps_alt = parsed_location

    # Fetch current row values
    row = con.execute(
        "SELECT w, h, taken_at, gps_lat, gps_lon, gps_alt FROM images WHERE sha1=?",
        (sha1,),
    ).fetchone()
    if row is None:
        return

    updates: Dict[str, Any] = {}
    if w > 0 and not (row["w"] and row["w"] > 0):
        updates["w"] = w
    if h > 0 and not (row["h"] and row["h"] > 0):
        updates["h"] = h
    if duration is not None:
        updates["duration"] = duration
    if taken_at is not None and row["taken_at"] is None:
        updates["taken_at"] = taken_at
    if gps_lat is not None and row["gps_lat"] is None:
        updates["gps_lat"] = gps_lat
    if gps_lon is not None and row["gps_lon"] is None:
        updates["gps_lon"] = gps_lon
    if gps_alt is not None and row["gps_alt"] is None:
        updates["gps_alt"] = gps_alt

    if updates:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        con.execute(
            f"UPDATE images SET {set_clause} WHERE sha1 = ?",
            (*updates.values(), sha1),
        )


def enrich_exif(con: sqlite3.Connection, sha1: str, abs_path: Path) -> None:
    from lib import exif_sqlite as exifmod

    row = con.execute(
        "SELECT taken_at, gps_lat, gps_lon, gps_alt FROM images WHERE sha1=?",
        (sha1,),
    ).fetchone()
    if row is None:
        return

    current_taken, current_lat, current_lon, current_alt = row

    try:
        exif_taken_at, exif_lat, exif_lon, exif_alt = exifmod.read_exif_data(abs_path)
    except Exception:
        exif_taken_at, exif_lat, exif_lon, exif_alt = None, None, None, None

    filename_taken_at = exifmod.parse_datetime_from_filename(abs_path)
    resolved_taken_at = exif_taken_at if exif_taken_at is not None else filename_taken_at

    if exif_taken_at is not None and filename_taken_at is not None:
        exif_date = datetime.fromtimestamp(exif_taken_at, tz=timezone.utc).date()
        filename_date = datetime.fromtimestamp(filename_taken_at, tz=timezone.utc).date()
        if exif_date != filename_date:
            resolved_taken_at = exif_taken_at

    next_taken = resolved_taken_at if current_taken is None and resolved_taken_at is not None else current_taken
    next_lat = exif_lat if current_lat is None and exif_lat is not None else current_lat
    next_lon = exif_lon if current_lon is None and exif_lon is not None else current_lon
    next_alt = exif_alt if current_alt is None and exif_alt is not None else current_alt

    con.execute(
        """
        UPDATE images
        SET
          taken_at = ?,
          gps_lat = ?,
          gps_lon = ?,
          gps_alt = ?
        WHERE sha1 = ?
        """,
        (next_taken, next_lat, next_lon, next_alt, sha1),
    )


def enrich_mime(con: sqlite3.Connection, sha1: str, abs_path: Path) -> Optional[str]:
    from lib import mime_sqlite as mimemod

    mime = mimemod.detect_mime(abs_path)
    if mime:
        con.execute("UPDATE images SET mime = ? WHERE sha1 = ?", (mime, sha1))
    return mime


def load_location_context(db_path: Path) -> Optional[Dict[str, Any]]:
    from lib import geonames_location as locmod

    if not geonames_available(db_path):
        return None

    con = sqlite3.connect(str(db_path))
    try:
        locmod.ensure_images_columns(con)
        locmod.ensure_cache_table(con)
        cc_map = locmod.load_country_map(con)
        admin1_map = locmod.load_admin1_map(con)
        _, lat_arr, lon_arr, meta = locmod.load_geonames(con)
        tree = locmod.build_tree(lat_arr, lon_arr)
        if tree is None:
            return None
    finally:
        con.close()

    return {
        "cc_map": cc_map,
        "admin1_map": admin1_map,
        "lat_arr": lat_arr,
        "lon_arr": lon_arr,
        "meta": meta,
        "tree": tree,
        "round_digits": 3,
        "max_km": 50.0,
    }


def process_one_base_media(
    con: sqlite3.Connection,
    img_path: Path,
    pipeline_ctx: Dict[str, Any],
    location_ctx: Optional[Dict[str, Any]],
    add_step: str,
    skip_thumbs: bool,
    sha1: Optional[str] = None,
    mime: Optional[str] = None,
) -> Tuple[str, str]:
    photos_dir = pipeline_ctx["photos_dir"]
    try:
        relpath = str(img_path.resolve().relative_to(photos_dir)).replace("\\", "/")
    except ValueError as exc:
        raise RuntimeError(f"Media outside photos dir: {img_path}") from exc

    if sha1 is None:
        sha1 = sha1_file(img_path)

    if mime is None:
        mime = detect_mime(img_path)

    if not is_base_only_mime(mime):
        raise RuntimeError(f"Unsupported base-only mime: {mime or 'unknown'}")

    def safe_job_state(status: str, detail: str) -> None:
        try:
            ensure_job_state(con, sha1, add_step, status, detail)
        except sqlite3.IntegrityError:
            return

    st = img_path.stat()

    try:
        video_meta = probe_video_metadata(img_path)
        w = int(video_meta.get("w") or 0)
        h = int(video_meta.get("h") or 0)
        duration: Optional[float] = None
        try:
            d = float(video_meta.get("duration") or 0)
            if d > 0:
                duration = d
        except Exception:
            duration = None

        upsert_image_row(
            con=con,
            sha1=sha1,
            relpath=relpath,
            mtime=float(st.st_mtime),
            w=w,
            h=h,
            duration=duration,
            file_size=int(st.st_size),
            mime=mime,
        )
        safe_job_state("processing", "start_base_only")
        con.commit()

        enrich_video_metadata(con, sha1, img_path)
        enrich_exif(con, sha1, img_path)

        if location_ctx is not None:
            enrich_location(con, sha1, location_ctx)

        if not skip_thumbs:
            thumb_path = pipeline_ctx["thumb_dir"] / f"{sha1}.jpg"
            pipeline_ctx["thumbsmod"].make_video_thumb(img_path, thumb_path, size=pipeline_ctx["thumb_size"])

        safe_job_state("done_base", f"base_only_mime:{mime}")
        con.commit()
        return sha1, "done_base"

    except Exception as exc:
        safe_job_state("error", str(exc))
        con.commit()
        return sha1, f"error: {exc}"


def enrich_location(con: sqlite3.Connection, sha1: str, location_ctx: Dict[str, Any]) -> None:
    from lib import geonames_location as locmod

    row = con.execute("SELECT gps_lat, gps_lon FROM images WHERE sha1=?", (sha1,)).fetchone()
    if row is None:
        return
    gps_lat, gps_lon = row
    if gps_lat is None or gps_lon is None:
        return

    round_digits = location_ctx["round_digits"]
    max_km = location_ctx["max_km"]
    latr = round(float(gps_lat), round_digits)
    lonr = round(float(gps_lon), round_digits)

    c = con.execute(
        "SELECT country_code, country, region, city, place_name FROM geocode_cache WHERE lat_round=? AND lon_round=?",
        (latr, lonr),
    ).fetchone()

    if c:
        country_code, country, region, city, place_name = c
    else:
        city_rec, dist_km = locmod.nearest_city(
            float(gps_lat),
            float(gps_lon),
            location_ctx["lat_arr"],
            location_ctx["lon_arr"],
            location_ctx["meta"],
            location_ctx["tree"],
            k=1,
        )
        if dist_km > max_km:
            return

        country_code = (city_rec.get("country_code") or "").upper()
        country = location_ctx["cc_map"].get(country_code, "") if country_code else ""
        admin1_code = city_rec.get("admin1_code") or ""
        region = (
            location_ctx["admin1_map"].get(f"{country_code}.{admin1_code}", "")
            if (country_code and admin1_code)
            else ""
        )
        city = city_rec.get("name") or city_rec.get("asciiname") or ""
        place_name = locmod.format_place(city, region, country)

        con.execute(
            """
            INSERT INTO geocode_cache
              (lat_round, lon_round, country_code, country, region, city, place_name, geonameid, dist_km)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(lat_round, lon_round) DO UPDATE SET
              country_code=excluded.country_code,
              country=excluded.country,
              region=excluded.region,
              city=excluded.city,
              place_name=excluded.place_name,
              geonameid=excluded.geonameid,
              dist_km=excluded.dist_km,
              updated_at=unixepoch()
            """,
            (
                latr,
                lonr,
                country_code,
                country,
                region,
                city,
                place_name,
                int(city_rec.get("geonameid") or 0),
                float(dist_km),
            ),
        )

    con.execute(
        """
        UPDATE images
        SET gps_lat_round=?, gps_lon_round=?,
            country_code=?, country=?, region=?, city=?, place_name=?,
            location_source=?
        WHERE sha1=?
        """,
        (latr, lonr, country_code, country, region, city, place_name, "geonames", sha1),
    )


def build_pipeline_context(
    photos_dir: Path,
    chroma_dir: Path,
    image_collection: str,
    captions_collection: str,
    device: str,
    skip_captions: bool,
    skip_thumbs: bool,
    dtype_name: str,
    caption_model: str,
    translate_model: str,
) -> Dict[str, Any]:
    from lib import thumbnail_precompute as thumbsmod
    from lib import caption_pipeline as capmod

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available: switching to CPU.")
        device = "cpu"

    ctx: Dict[str, Any] = {
        "device": device,
        "photos_dir": photos_dir.resolve(),
        "thumbsmod": thumbsmod,
        "thumb_dir": Path(os.environ.get("PHOTOAI_THUMB_DIR", "cache/thumbs")).expanduser().resolve(),
        "thumb_size": int(os.environ.get("PHOTOAI_THUMB_SIZE", "256")),
        "capmod": capmod,
        "skip_captions": skip_captions,
        "skip_thumbs": skip_thumbs,
    }

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-336", pretrained="openai", force_quick_gelu=True
    )
    model = model.to(device).eval()
    ctx["image_model"] = model
    ctx["image_preprocess"] = preprocess

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    img_col = client.get_or_create_collection(name=image_collection, metadata={"hnsw:space": "cosine"})
    cap_col = client.get_or_create_collection(name=captions_collection)
    ctx["img_col"] = img_col
    ctx["cap_col"] = cap_col

    if skip_captions:
        return ctx

    if dtype_name == "fp16":
        dtype = torch.float16
    elif dtype_name == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    cap_proc, cap_model_obj = capmod.load_blip2(caption_model, device=device, dtype=dtype)
    tr_tok, tr_model = capmod.load_translator_nllb(translate_model, device=device, dtype=dtype)
    clip_model, clip_tok = capmod.load_openclip_text(device=device)

    ctx["caption_model_id"] = caption_model
    ctx["translate_model_id"] = translate_model
    ctx["cap_proc"] = cap_proc
    ctx["cap_model"] = cap_model_obj
    ctx["tr_tok"] = tr_tok
    ctx["tr_model"] = tr_model
    ctx["clip_model"] = clip_model
    ctx["clip_tok"] = clip_tok
    return ctx


def process_one_image(
    con: sqlite3.Connection,
    img_path: Path,
    pipeline_ctx: Dict[str, Any],
    location_ctx: Optional[Dict[str, Any]],
    add_step: str,
    skip_captions: bool,
    skip_thumbs: bool,
    sha1: Optional[str] = None,
) -> Tuple[str, str]:
    from lib import chroma_image_index as idxmod

    mime = detect_mime(img_path)
    if not is_supported_mime(mime):
        if sha1 is None:
            sha1 = sha1_file(img_path)
        return sha1, f"unsupported mime: {mime or 'unknown'}"

    if is_base_only_mime(mime):
        return process_one_base_media(
            con=con,
            img_path=img_path,
            pipeline_ctx=pipeline_ctx,
            location_ctx=location_ctx,
            add_step=add_step,
            skip_thumbs=skip_thumbs,
            sha1=sha1,
            mime=mime,
        )

    photos_dir = pipeline_ctx["photos_dir"]
    try:
        relpath = str(img_path.resolve().relative_to(photos_dir)).replace("\\", "/")
    except ValueError as exc:
        raise RuntimeError(f"Image outside photos dir: {img_path}") from exc

    if sha1 is None:
        sha1 = sha1_file(img_path)

    def safe_job_state(status: str, detail: str) -> None:
        try:
            ensure_job_state(con, sha1, add_step, status, detail)
        except sqlite3.IntegrityError:
            return

    st = img_path.stat()
    upsert_image_row(
        con=con,
        sha1=sha1,
        relpath=relpath,
        mtime=float(st.st_mtime),
        w=0,
        h=0,
        duration=0.0,
        file_size=int(st.st_size),
        mime=None,
    )
    safe_job_state("processing", "start")
    con.commit()

    try:
        embs, metas = idxmod.embed_batch(
            model=pipeline_ctx["image_model"],
            preprocess=pipeline_ctx["image_preprocess"],
            device=pipeline_ctx["device"],
            paths=[img_path],
            use_fp16=True,
            base_dir=photos_dir,
        )
        if not embs or not metas:
            raise RuntimeError("Cannot compute image embedding")

        meta0 = metas[0]
        meta0["file_size"] = int(st.st_size)
        pipeline_ctx["img_col"].upsert(ids=[sha1], embeddings=[embs[0]], metadatas=[meta0])

        upsert_image_row(
            con=con,
            sha1=sha1,
            relpath=relpath,
            mtime=float(meta0.get("mtime") or st.st_mtime),
            w=int(meta0.get("w") or 0),
            h=int(meta0.get("h") or 0),
            duration=0.0,
            file_size=int(st.st_size),
            mime=None,
        )

        enrich_exif(con, sha1, img_path)
        mime = enrich_mime(con, sha1, img_path)

        if location_ctx is not None:
            enrich_location(con, sha1, location_ctx)

        if not skip_thumbs:
            thumb_path = pipeline_ctx["thumb_dir"] / f"{sha1}.jpg"
            pipeline_ctx["thumbsmod"].make_thumb(img_path, thumb_path, size=pipeline_ctx["thumb_size"])

        if not skip_captions:
            capmod = pipeline_ctx["capmod"]
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                dir_ctx = capmod.dir_context_from_relpath(relpath, levels=2)
                prompt = capmod.make_caption_prompt_en(dir_ctx)
                caption_en = capmod.generate_caption_en(
                    pipeline_ctx["cap_proc"],
                    pipeline_ctx["cap_model"],
                    im,
                    prompt,
                    max_new_tokens=90,
                )

            caption_it = capmod.translate_en_to_it_nllb(
                pipeline_ctx["tr_tok"],
                pipeline_ctx["tr_model"],
                caption_en,
                device=pipeline_ctx["device"],
                max_new_tokens=128,
            )

            tags_en = capmod.extract_tags_en(caption_en, dir_ctx=dir_ctx, max_tags=14)
            tag_texts_en = [t for (t, _, _) in tags_en]
            tag_texts_it = capmod.translate_tags_en_to_it_nllb(
                pipeline_ctx["tr_tok"],
                pipeline_ctx["tr_model"],
                tag_texts_en,
                device=pipeline_ctx["device"],
                max_new_tokens=16,
            )

            tags_it_map: Dict[str, Tuple[float, str]] = {}
            for (tag_en, score, src), tag_it in zip(tags_en, tag_texts_it):
                t_it = capmod.normalize_tag_it(tag_it)
                if not t_it or len(t_it) < 3:
                    continue
                prev = tags_it_map.get(t_it)
                if prev is None or score > prev[0]:
                    tags_it_map[t_it] = (score, f"{src}_translated")

            tags_it: List[Tuple[str, float, str]] = [(t, sc, src) for t, (sc, src) in tags_it_map.items()]
            tags_it.sort(key=lambda x: (-x[1], x[0]))
            tags_it = tags_it[:14]

            params = {
                "relpath": relpath,
                "dirctx": dir_ctx,
                "prompt_en": prompt,
                "translator_model": pipeline_ctx["translate_model_id"],
                "tags_en": tag_texts_en,
            }

            capmod.upsert_caption_it(
                con,
                sha1=sha1,
                caption_it=caption_it,
                caption_en=caption_en,
                model_name=pipeline_ctx["caption_model_id"],
                params=params,
            )
            capmod.replace_tags(con, sha1, tags_it)

            emb_it = capmod.embed_text_openclip(
                pipeline_ctx["clip_model"],
                pipeline_ctx["clip_tok"],
                device=pipeline_ctx["device"],
                text=caption_it,
            )
            cap_meta = {
                "sha1": sha1,
                "path": relpath,
                "lang": "it",
                "caption_it": caption_it,
                "caption_en": caption_en,
                "dirctx": dir_ctx,
            }
            pipeline_ctx["cap_col"].upsert(
                ids=[sha1],
                embeddings=[emb_it],
                metadatas=[cap_meta],
                documents=[caption_it],
            )

        safe_job_state("done", "ok")
        con.commit()
        return sha1, "done"

    except Exception as exc:
        safe_job_state("error", str(exc))
        con.commit()
        return sha1, f"error: {exc}"


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "One-shot ingest pipeline per image: sort/copy new images, then for each image run "
            "Chroma image index + SQLite sync + EXIF/MIME/location + thumbs + captions/tags."
        )
    )
    parser.add_argument("--from", dest="src_dir", default="", help="Source directory with new images")
    parser.add_argument(
        "--to",
        dest="dst_dir",
        default="",
        help="Destination photos directory (default: PHOTOAI_PHOTOS_DIR)",
    )
    parser.add_argument("--move", action="store_true", help="Move files instead of copying (default: copy)")
    parser.add_argument("--recursive", action="store_true", help="Recurse source subdirectories in media_sorter")
    parser.add_argument("--max-depth", type=int, default=0, help="Max depth for media_sorter if not recursive")
    parser.add_argument(
        "--prefer-metadata",
        action="store_true",
        help="On date conflicts in media_sorter, prefer EXIF date and rename",
    )
    parser.add_argument("--index-batch", type=int, default=64, help="Compatibility option (unused in per-image mode)")
    parser.add_argument("--sync-batch", type=int, default=2000, help="Compatibility option (unused in per-image mode)")
    parser.add_argument("--skip-location", action="store_true", help="Skip GeoNames location backfill")
    parser.add_argument("--skip-thumbs", action="store_true", help="Skip thumbnail generation")
    parser.add_argument("--skip-captions", action="store_true", help="Skip caption/tag generation")
    parser.add_argument(
        "--sync-missing",
        action="store_true",
        help="Scan PHOTOAI_PHOTOS_DIR and sync missing SQLite/Chroma/thumb/captions/tags for all files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no writes). In --sync-missing mode it performs checks and prints planned actions.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="With --sync-missing, print only progress + final summary (no per-file output).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="With --sync-missing, visit only the first N files (0 = no limit).",
    )
    parser.add_argument(
        "--sync-subdir",
        default="",
        help="With --sync-missing, scan only this subfolder under PHOTOAI_PHOTOS_DIR.",
    )
    parser.add_argument(
        "--sync-mime",
        default="",
        help="With --sync-missing, scan only files with this MIME (e.g. image/jpeg).",
    )
    args = parser.parse_args()

    device = os.environ.get("PHOTOAI_DEVICE", "cuda").strip().lower()
    if device not in {"cuda", "cpu"}:
        raise RuntimeError("PHOTOAI_DEVICE must be one of: cuda, cpu")

    image_collection = os.environ.get("PHOTOAI_COLLECTION", "images_openclip_vitl14_336").strip()
    if not image_collection:
        raise RuntimeError("PHOTOAI_COLLECTION must not be empty")

    captions_collection = os.environ.get("PHOTOAI_CAPTIONS_COLLECTION", "captions_openclip_vitl14_336").strip()
    if not captions_collection:
        raise RuntimeError("PHOTOAI_CAPTIONS_COLLECTION must not be empty")

    caption_step = os.environ.get("PHOTOAI_CAPTION_STEP", "caption").strip()
    if not caption_step:
        raise RuntimeError("PHOTOAI_CAPTION_STEP must not be empty")

    caption_pass_limit_raw = os.environ.get("PHOTOAI_CAPTION_PASS_LIMIT", "16").strip()
    try:
        caption_pass_limit = int(caption_pass_limit_raw)
    except ValueError as exc:
        raise RuntimeError("PHOTOAI_CAPTION_PASS_LIMIT must be an integer") from exc
    if caption_pass_limit < 1:
        raise RuntimeError("PHOTOAI_CAPTION_PASS_LIMIT must be >= 1")

    dtype = os.environ.get("PHOTOAI_DTYPE", "bf16").strip().lower()
    if dtype not in {"fp16", "bf16", "fp32"}:
        raise RuntimeError("PHOTOAI_DTYPE must be one of: fp16, bf16, fp32")

    caption_model = os.environ.get("PHOTOAI_CAPTION_MODEL", "Salesforce/blip2-flan-t5-xxl").strip()
    if not caption_model:
        raise RuntimeError("PHOTOAI_CAPTION_MODEL must not be empty")

    translate_model = os.environ.get("PHOTOAI_TRANSLATE_MODEL", "facebook/nllb-200-distilled-600M").strip()
    if not translate_model:
        raise RuntimeError("PHOTOAI_TRANSLATE_MODEL must not be empty")

    photos_dir = Path(args.dst_dir).expanduser() if args.dst_dir else Path(must_env("PHOTOAI_PHOTOS_DIR")).expanduser()
    sqlite_dir = Path(must_env("PHOTOAI_SQLITE_DIR")).expanduser()
    chroma_dir = Path(must_env("PHOTOAI_CHROMA_DIR")).expanduser()
    db_path = sqlite_dir / "photo_ai.sqlite"

    print("Pipeline config:")
    print(f"  Source      : {Path(args.src_dir).expanduser()}")
    print(f"  Photos      : {photos_dir}")
    print(f"  SQLite      : {db_path}")
    print(f"  Chroma      : {chroma_dir}")
    print(f"  Collection  : {image_collection}")
    print(f"  Device      : {device}")

    if args.sync_missing:
        if args.limit < 0:
            raise RuntimeError("--limit must be >= 0")
        sync_mime_clean = args.sync_mime.strip().lower()
        if sync_mime_clean and sync_mime_clean not in SUPPORTED_MIME_TYPES:
            raise RuntimeError(f"--sync-mime must be one of: {', '.join(sorted(SUPPORTED_MIME_TYPES))}")
        if args.skip_captions or args.skip_thumbs:
            print("[WARN] --skip-captions/--skip-thumbs are ignored in --sync-missing mode.")
        run_sync_missing_photos_dir(
            photos_dir=photos_dir,
            db_path=db_path,
            chroma_dir=chroma_dir,
            image_collection=image_collection,
            captions_collection=captions_collection,
            device=device,
            dtype=dtype,
            caption_model=caption_model,
            translate_model=translate_model,
            dry_run=args.dry_run,
            summary_only=args.quiet,
            limit=args.limit,
            subdir=args.sync_subdir,
            only_mime=sync_mime_clean,
        )
        return

    if args.sync_subdir.strip() or args.sync_mime.strip():
        raise RuntimeError("--sync-subdir and --sync-mime require --sync-missing")

    if not args.src_dir:
        raise RuntimeError("--from is required unless --sync-missing is used")

    photos_dir_resolved = photos_dir.resolve()

    con: Optional[sqlite3.Connection] = None
    done = 0
    done_base = 0
    failed = 0
    add_step = "add_all"

    if not args.dry_run:
        step_header("Prepare pipeline models/clients")
        pipeline_ctx = build_pipeline_context(
            photos_dir=photos_dir_resolved,
            chroma_dir=chroma_dir,
            image_collection=image_collection,
            captions_collection=captions_collection,
            device=device,
            skip_captions=args.skip_captions,
            skip_thumbs=args.skip_thumbs,
            dtype_name=dtype,
            caption_model=caption_model,
            translate_model=translate_model,
        )

        location_ctx: Optional[Dict[str, Any]] = None
        if not args.skip_location:
            location_ctx = load_location_context(db_path)
            if location_ctx is None:
                print("[WARN] Location backfill disabled: GeoNames data or scipy unavailable.")

        con = sqlite3.connect(str(db_path))
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        con.row_factory = sqlite3.Row

        def on_saved(dst_path: Path) -> None:
            nonlocal done, done_base, failed

            if con is None:
                return

            sha1 = None

            sha1_out, status = process_one_image(
                con=con,
                img_path=dst_path.resolve(),
                pipeline_ctx=pipeline_ctx,
                location_ctx=location_ctx,
                add_step=add_step,
                skip_captions=args.skip_captions,
                skip_thumbs=args.skip_thumbs,
                sha1=sha1,
            )
            if status == "done":
                done += 1
                print(f"[OK] {sha1_out}")
            elif status == "done_base":
                done_base += 1
                print(f"[OK-BASE] {sha1_out}")
            else:
                failed += 1
                print(f"[ERR] {sha1_out} -> {status}")
    else:
        on_saved = None

    run_imgsort(
        src_dir=args.src_dir,
        dst_dir=photos_dir,
        move=args.move,
        recursive=args.recursive,
        max_depth=args.max_depth,
        prefer_metadata=args.prefer_metadata,
        dry_run=args.dry_run,
        on_saved=on_saved,
    )

    if args.dry_run:
        print("\nDry-run complete: stopped after media_sorter.")
        return

    if con is not None:
        con.close()

    print("\nPipeline completed.")
    print(f"  done  : {done}")
    print(f"  done_base : {done_base}")
    print(f"  error : {failed}")
    print(f"  state : jobs.step='{add_step}'")
    print(f"  env   : PHOTOAI_CAPTION_STEP='{caption_step}' PHOTOAI_CAPTION_PASS_LIMIT={caption_pass_limit}")


if __name__ == "__main__":
    main()
