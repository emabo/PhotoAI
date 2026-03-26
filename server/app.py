# server/app.py
from __future__ import annotations

import os
import re
import time
import sqlite3
import uuid
import warnings
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse, Response

import chromadb
from chromadb.config import Settings

import torch
import open_clip
from PIL import Image, ImageFile, ImageOps
from dotenv import load_dotenv
from lib.media_types import SUPPORTED_MIME_TYPES, VIDEO_EXTENSIONS, is_video_mime

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:
    pass

load_dotenv()

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings(
    "ignore",
    message="QuickGELU mismatch.*"
)

# Ensure logging is configured so startup info is visible under uvicorn
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


# ---------------------------
# Config (override via env)
# ---------------------------

BASE = Path(__file__).resolve().parents[1]

CHROMA_DIR = Path(os.environ.get("PHOTOAI_CHROMA_DIR", str(BASE / "db" / "chroma"))).expanduser().resolve()
THUMB_DIR = Path(os.environ.get("PHOTOAI_THUMB_DIR", str(BASE / "cache" / "thumbs"))).expanduser().resolve()
PHOTOS_DIR = Path(os.environ.get("PHOTOAI_PHOTOS_DIR", str(BASE / "photos"))).expanduser().resolve()

COLLECTION = os.environ.get("PHOTOAI_COLLECTION", "images_openclip_vitl14_336")
DEVICE = os.environ.get("PHOTOAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
THUMB_SIZE = int(os.environ.get("PHOTOAI_THUMB_SIZE", "256"))
DEFAULT_K = int(os.environ.get("PHOTOAI_DEFAULT_K", "60"))
MAX_K = int(os.environ.get("PHOTOAI_MAX_K", "200"))
MAX_CANDIDATES = int(os.environ.get("PHOTOAI_MAX_CANDIDATES", "1000"))
SQLITE_DIR = Path(os.environ.get("PHOTOAI_SQLITE_DIR", str(BASE / "db" / "sqlite"))).expanduser().resolve()
SQLITE_DB_PATH = SQLITE_DIR / "photo_ai.sqlite"

# Safety: if true, only serve images whose resolved path is under PHOTOS_DIR
RESTRICT_TO_PHOTOS_DIR = os.environ.get("PHOTOAI_RESTRICT_TO_PHOTOS_DIR", "1").strip() not in ("0", "false", "False")

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger("photoai")
logger.info(
    "PHOTOAI config | CHROMA_DIR=%s THUMB_DIR=%s PHOTOS_DIR=%s SQLITE_DB=%s COLLECTION=%s DEVICE=%s THUMB_SIZE=%s DEFAULT_K=%s MAX_K=%s MAX_CANDIDATES=%s RESTRICT_TO_PHOTOS_DIR=%s",
    CHROMA_DIR,
    THUMB_DIR,
    PHOTOS_DIR,
    SQLITE_DB_PATH,
    COLLECTION,
    DEVICE,
    THUMB_SIZE,
    DEFAULT_K,
    MAX_K,
    MAX_CANDIDATES,
    RESTRICT_TO_PHOTOS_DIR,
)

SEARCH_CONTEXT_TTL_SEC = 30 * 60
SEARCH_CONTEXTS: Dict[str, Dict[str, Any]] = {}

# ---------------------------
# Model + Chroma helpers
# ---------------------------

def chroma_client(persist_dir: str):
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def chroma_collection():
    client = chroma_client(str(CHROMA_DIR))
    return client.get_collection(COLLECTION)


def load_openclip(device: str):
    model_name = "ViT-L-14-336"
    pretrained = "openai"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        force_image_size=336,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device).eval()
    return model, preprocess, tokenizer


MODEL, PREPROCESS, TOKENIZER = load_openclip(DEVICE)


@torch.inference_mode()
def embed_text(model, tokenizer, device: str, text: str) -> List[float]:
    tokens = tokenizer([text]).to(device)
    feats = model.encode_text(tokens)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.detach().cpu().float().tolist()[0]


def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation to an image.
    Falls back gracefully if EXIF data is missing or corrupted.
    """
    try:
        # Try ImageOps.exif_transpose first (PIL >= 6.0)
        oriented = ImageOps.exif_transpose(img)
        return oriented if oriented is not None else img
    except Exception:
        # If EXIF parsing fails, try manual approach
        try:
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            if exif_data and 274 in exif_data:  # 274 = Orientation tag
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
    """
    Load image from path and apply EXIF orientation if present.
    """
    img = Image.open(path)
    return _apply_exif_orientation(img)


def orient_image(img: Image.Image) -> Image.Image:
    """
    Apply EXIF orientation to an already-loaded image.
    Used for images loaded from bytes (e.g., uploaded files).
    """
    return _apply_exif_orientation(img)


@torch.inference_mode()
def embed_image(model, preprocess, device: str, image: Image.Image) -> List[float]:
    im = image.convert("RGB")
    x = preprocess(im).unsqueeze(0).to(device)
    feats = model.encode_image(x)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.detach().cpu().float().tolist()[0]


def is_video_media(path: Path, mime: Optional[str] = None) -> bool:
    mime_clean = (mime or "").strip().lower()
    if mime_clean:
        if is_video_mime(mime_clean):
            return True
    return path.suffix.lower() in VIDEO_EXTENSIONS


def ensure_thumb(sha1: str, img_path: Path, size: int = THUMB_SIZE, mime: Optional[str] = None) -> Optional[Path]:
    out = THUMB_DIR / f"{sha1}.jpg"
    if out.exists():
        return out
    # Use ffmpeg for video files
    if is_video_media(img_path, mime):
        try:
            import subprocess
            scale_filter = f"scale={size}:{size}:force_original_aspect_ratio=decrease"
            def _run(extra_args: list) -> bool:
                result = subprocess.run(
                    ["ffmpeg", "-y"] + extra_args + [
                        "-i", str(img_path),
                        "-vframes", "1",
                        "-vf", scale_filter,
                        str(out),
                    ],
                    capture_output=True,
                    timeout=30,
                )
                return result.returncode == 0 and out.exists()
            out.parent.mkdir(parents=True, exist_ok=True)
            if _run(["-ss", "00:00:01"]) or _run([]):
                return out
        except Exception:
            pass
        return None
    try:
        im = load_and_orient_image(img_path)
        im = im.convert("RGB")
        im.thumbnail((size, size))
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, "JPEG", quality=85, optimize=True)
        return out
    except Exception:
        return None


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def within_photos_dir(p: Path) -> bool:
    try:
        p_res = p.resolve()
        root = PHOTOS_DIR.resolve()
        return root == p_res or root in p_res.parents
    except Exception:
        return False


def pick_image_path(meta: Dict[str, Any]) -> Optional[Path]:
    """
    Prefer relpath if present (portable). Fallback to absolute path.
    Both are resolved relative to PHOTOS_DIR when needed.
    """
    # Try relative path first (more portable)
    rel = meta.get("relpath")
    if rel:
        p = (PHOTOS_DIR / rel).expanduser()
        return p
    
    # Fallback to absolute path
    abs_path = meta.get("path")
    if abs_path:
        p = Path(abs_path).expanduser()
        # If it's already absolute, use it directly
        if p.is_absolute():
            return p
        # Otherwise treat as relative to PHOTOS_DIR
        return PHOTOS_DIR / abs_path
    
    return None


def sqlite_connect() -> sqlite3.Connection:
    con = sqlite3.connect(str(SQLITE_DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def parse_date_to_epoch(date_str: str, end_of_day: bool = False) -> Optional[int]:
    s = (date_str or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None
    if end_of_day:
        dt = dt.replace(hour=23, minute=59, second=59)
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_images_filtered(
    candidate_ids: Optional[List[str]],
    folder: str,
    mime: str,
    country_code: str,
    region: str,
    city: str,
    date_from: str,
    date_to: str,
    has_caption_en: bool,
    only_complete: bool,
) -> List[sqlite3.Row]:
    if not SQLITE_DB_PATH.exists():
        return []

    where = []
    params: List[Any] = []

    if candidate_ids is not None:
        if not candidate_ids:
            return []
        placeholders = ",".join(["?"] * len(candidate_ids))
        where.append(f"i.sha1 IN ({placeholders})")
        params.extend(candidate_ids)

    folder_clean = (folder or "").strip()
    if folder_clean:
        where.append("i.path LIKE ?")
        params.append(f"%{folder_clean}%")

    mime_clean = (mime or "").strip().lower()
    if mime_clean:
        where.append("lower(ifnull(i.mime,'')) = ?")
        params.append(mime_clean)

    cc = (country_code or "").strip().upper()
    if cc:
        where.append("upper(ifnull(i.country_code,'')) = ?")
        params.append(cc)

    region_clean = (region or "").strip()
    if region_clean:
        where.append("lower(ifnull(i.region,'')) LIKE lower(?)")
        params.append(f"%{region_clean}%")

    city_clean = (city or "").strip()
    if city_clean:
        where.append("lower(ifnull(i.city,'')) LIKE lower(?)")
        params.append(f"%{city_clean}%")

    ts_from = parse_date_to_epoch(date_from, end_of_day=False)
    ts_to = parse_date_to_epoch(date_to, end_of_day=True)
    if ts_from is not None:
        where.append("i.taken_at >= ?")
        params.append(ts_from)
    if ts_to is not None:
        where.append("i.taken_at <= ?")
        params.append(ts_to)

    if has_caption_en:
        where.append(
            "EXISTS ("
            "SELECT 1 FROM captions c "
            "WHERE c.sha1 = i.sha1 "
            "AND json_extract(c.params_json, '$.caption_en') IS NOT NULL "
            "AND trim(json_extract(c.params_json, '$.caption_en')) <> ''"
            ")"
        )

    if only_complete:
        where.append("i.taken_at IS NOT NULL")
        where.append("i.gps_lat IS NOT NULL AND i.gps_lon IS NOT NULL")
        where.append("i.country_code IS NOT NULL AND trim(i.country_code) <> ''")
        where.append("i.city IS NOT NULL AND trim(i.city) <> ''")
        where.append("i.place_name IS NOT NULL AND trim(i.place_name) <> ''")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    sql = f"""
        SELECT
          i.sha1,
          i.path,
          i.mime,
          i.w,
          i.h,
          i.mtime,
          i.taken_at,
          i.gps_lat,
          i.gps_lon,
          i.country,
          i.country_code,
          i.region,
          i.city,
          i.place_name,
          i.location_source
        FROM images i
        {where_sql}
    """

    con = sqlite_connect()
    try:
        return con.execute(sql, tuple(params)).fetchall()
    finally:
        con.close()


def resolve_image_path_from_rel(rel_path: Optional[str]) -> Optional[Path]:
    rel = (rel_path or "").strip()
    if not rel:
        return None
    p = (PHOTOS_DIR / rel).expanduser()
    return p


def fetch_distinct_mimes() -> List[str]:
    db_mimes: set[str] = set()
    if SQLITE_DB_PATH.exists():
        con = sqlite_connect()
        try:
            rows = con.execute(
                """
                SELECT DISTINCT trim(mime) AS mime
                FROM images
                WHERE mime IS NOT NULL AND trim(mime) <> ''
                ORDER BY lower(trim(mime)) ASC
                """
            ).fetchall()
            db_mimes = {str(r["mime"]) for r in rows if r["mime"]}
        finally:
            con.close()
    # Always include all supported MIME types so the filter is usable
    # even before any file of that type has been indexed.
    all_mimes = db_mimes | SUPPORTED_MIME_TYPES
    return sorted(all_mimes, key=str.lower)


def get_photo_detail(sha1: str) -> Optional[Dict[str, Any]]:
    if not SQLITE_DB_PATH.exists():
        return None

    con = sqlite_connect()
    try:
        image_row = con.execute(
            """
            SELECT *
            FROM images
            WHERE sha1 = ?
            LIMIT 1
            """,
            (sha1,),
        ).fetchone()
        if not image_row:
            return None

        caption_row = con.execute(
            """
            SELECT caption, lang, model, params_json, created_at, updated_at
            FROM captions
            WHERE sha1 = ?
            LIMIT 1
            """,
            (sha1,),
        ).fetchone()

        tags_rows = con.execute(
            """
            SELECT tag, score, source
            FROM tags
            WHERE sha1 = ?
            ORDER BY score DESC, tag ASC
            """,
            (sha1,),
        ).fetchall()

        jobs_rows = con.execute(
            """
            SELECT step, status, detail, updated_at
            FROM jobs
            WHERE sha1 = ?
            ORDER BY step ASC, updated_at DESC
            """,
            (sha1,),
        ).fetchall()

        nav_row = con.execute(
            """
            WITH ordered AS (
              SELECT
                sha1,
                row_number() OVER (
                  ORDER BY (taken_at IS NULL) ASC, taken_at ASC, path ASC
                ) AS rn
              FROM images
            )
            SELECT
              p.sha1 AS prev_sha1,
                            n.sha1 AS next_sha1,
                            (SELECT sha1 FROM ordered ORDER BY rn ASC LIMIT 1) AS first_sha1,
                            (SELECT sha1 FROM ordered ORDER BY rn DESC LIMIT 1) AS last_sha1
            FROM ordered c
            LEFT JOIN ordered p ON p.rn = c.rn - 1
            LEFT JOIN ordered n ON n.rn = c.rn + 1
            WHERE c.sha1 = ?
            LIMIT 1
            """,
            (sha1,),
        ).fetchone()

        return {
            "image": image_row,
            "caption": caption_row,
            "tags": tags_rows,
            "jobs": jobs_rows,
            "prev_sha1": nav_row["prev_sha1"] if nav_row else None,
            "next_sha1": nav_row["next_sha1"] if nav_row else None,
            "first_sha1": nav_row["first_sha1"] if nav_row else None,
            "last_sha1": nav_row["last_sha1"] if nav_row else None,
        }
    finally:
        con.close()


def format_epoch_local(ts: Any) -> str:
    if isinstance(ts, (int, float)):
        return time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(ts))
    return "-"


def _fmt_duration(seconds: Any) -> str:
    """Format a duration in seconds to mm:ss or h:mm:ss."""
    if seconds is None:
        return "-"
    try:
        total = int(float(seconds))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
    except Exception:
        return "-"


def store_search_context(ids: List[str]) -> str:
    now = time.time()
    expired = [k for k, v in SEARCH_CONTEXTS.items() if now - float(v.get("created_at", 0)) > SEARCH_CONTEXT_TTL_SEC]
    for k in expired:
        SEARCH_CONTEXTS.pop(k, None)

    token = uuid.uuid4().hex
    SEARCH_CONTEXTS[token] = {
        "ids": list(ids),
        "created_at": now,
    }
    return token


def load_search_context(token: str) -> Optional[List[str]]:
    t = (token or "").strip()
    if not t:
        return None
    item = SEARCH_CONTEXTS.get(t)
    if not item:
        return None

    now = time.time()
    created = float(item.get("created_at", 0))
    if now - created > SEARCH_CONTEXT_TTL_SEC:
        SEARCH_CONTEXTS.pop(t, None)
        return None

    ids = item.get("ids")
    if not isinstance(ids, list):
        return None
    return [str(x) for x in ids]


def sqlite_table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def purge_context_ids(ids_to_remove: List[str]) -> None:
    if not ids_to_remove:
        return
    removed = set(ids_to_remove)
    for key in list(SEARCH_CONTEXTS.keys()):
        item = SEARCH_CONTEXTS.get(key) or {}
        ids = item.get("ids")
        if not isinstance(ids, list):
            continue
        new_ids = [sid for sid in ids if sid not in removed]
        if not new_ids:
            SEARCH_CONTEXTS.pop(key, None)
        elif len(new_ids) != len(ids):
            item["ids"] = new_ids


def delete_images_everywhere(ids: List[str]) -> Dict[str, Any]:
    ordered_ids: List[str] = []
    seen = set()
    for raw in ids:
        sid = str(raw or "").strip()
        if not sid or sid in seen:
            continue
        seen.add(sid)
        ordered_ids.append(sid)

    if not ordered_ids:
        return {
            "requested": 0,
            "deleted_db": 0,
            "deleted_chroma": 0,
            "deleted_files": 0,
            "missing_files": 0,
            "deleted_thumbs": 0,
            "missing_thumbs": 0,
            "file_delete_errors": 0,
            "thumb_delete_errors": 0,
        }

    file_paths_by_id: Dict[str, Optional[Path]] = {}
    db_rows_found = 0

    con = sqlite_connect()
    try:
        placeholders = ",".join(["?"] * len(ordered_ids))
        rows = con.execute(
            f"SELECT sha1, path FROM images WHERE sha1 IN ({placeholders})",
            tuple(ordered_ids),
        ).fetchall()
        db_rows_found = len(rows)
        for row in rows:
            file_paths_by_id[row["sha1"]] = resolve_image_path_from_rel(row["path"])

        deleted_files = 0
        missing_files = 0
        file_delete_errors = 0
        deleted_thumbs = 0
        missing_thumbs = 0
        thumb_delete_errors = 0

        for sid in ordered_ids:
            p = file_paths_by_id.get(sid)
            if p is None:
                missing_files += 1
            else:
                try:
                    p_res = p.resolve()
                    if not within_photos_dir(p_res):
                        file_delete_errors += 1
                    elif p_res.exists():
                        p_res.unlink()
                        deleted_files += 1
                    else:
                        missing_files += 1
                except Exception:
                    file_delete_errors += 1

            thumb = THUMB_DIR / f"{sid}.jpg"
            try:
                if thumb.exists():
                    thumb.unlink()
                    deleted_thumbs += 1
                else:
                    missing_thumbs += 1
            except Exception:
                thumb_delete_errors += 1

        if sqlite_table_exists(con, "captions_fts"):
            con.execute(
                f"DELETE FROM captions_fts WHERE sha1 IN ({placeholders})",
                tuple(ordered_ids),
            )

        con.execute(
            f"DELETE FROM captions WHERE sha1 IN ({placeholders})",
            tuple(ordered_ids),
        )
        con.execute(
            f"DELETE FROM tags WHERE sha1 IN ({placeholders})",
            tuple(ordered_ids),
        )
        con.execute(
            f"DELETE FROM jobs WHERE sha1 IN ({placeholders})",
            tuple(ordered_ids),
        )
        cur = con.execute(
            f"DELETE FROM images WHERE sha1 IN ({placeholders})",
            tuple(ordered_ids),
        )
        deleted_db = cur.rowcount if cur.rowcount is not None else 0
        con.commit()
    finally:
        con.close()

    deleted_chroma = 0
    chroma_error = None
    try:
        col = chroma_collection()
        col.delete(ids=ordered_ids)
        deleted_chroma = len(ordered_ids)
    except Exception as e:
        chroma_error = str(e)

    purge_context_ids(ordered_ids)

    out = {
        "requested": len(ordered_ids),
        "db_rows_found": db_rows_found,
        "deleted_db": deleted_db,
        "deleted_chroma": deleted_chroma,
        "deleted_files": deleted_files,
        "missing_files": missing_files,
        "deleted_thumbs": deleted_thumbs,
        "missing_thumbs": missing_thumbs,
        "file_delete_errors": file_delete_errors,
        "thumb_delete_errors": thumb_delete_errors,
    }
    if chroma_error:
        out["chroma_error"] = chroma_error
    return out


# ---------------------------
# App
# ---------------------------

app = FastAPI(title="photo_ai search", version="0.1")


@app.get("/", response_class=HTMLResponse)
def index():
    mime_options = fetch_distinct_mimes()
    mime_options_html = "".join(
        f"<option value=\"{html_escape(m)}\">{html_escape(m)}</option>"
        for m in mime_options
    )

    return f"""<!doctype html>
<html lang="it-IT">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://unpkg.com/htmx.org@1.9.12"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/flatpickr/dist/flatpickr.min.css" />
    <script src="https://cdn.jsdelivr.net/npm/flatpickr"></script>
    <script src="https://cdn.jsdelivr.net/npm/flatpickr/dist/l10n/it.js"></script>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
    .row {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    input[type="text"] {{ width: 170px; padding: 10px 12px; border: 1px solid #ccc; border-radius: 10px; }}
    #q {{ width: min(720px, 95vw); }}
    input[type="number"] {{ width: 110px; padding: 10px 12px; border: 1px solid #ccc; border-radius: 10px; }}
    .hint {{ color: #666; font-size: 12px; margin-top: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin-top: 14px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 10px; box-shadow: 0 1px 2px rgba(0,0,0,.05); }}
    .card.selected {{ border-color: #d33; background: #fff5f5; box-shadow: 0 0 0 2px rgba(211, 51, 51, 0.15); }}
    img.thumb {{ width: 100%; height: 180px; object-fit: contain; background: #f5f5f5; border-radius: 10px; }}
    .small {{ color: #666; font-size: 12px; word-break: break-all; margin-top: 6px; }}
    .dist {{ font-variant-numeric: tabular-nums; margin-top: 8px; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
    .badge {{ font-size: 12px; color: #333; background: #f3f3f3; border: 1px solid #e2e2e2; padding: 4px 8px; border-radius: 999px; }}
  </style>
</head>
<body>
  <div class="topbar">
    <h2 style="margin: 0;">Photo AI Search</h2>
    <span class="badge">collection: {html_escape(COLLECTION)} • chroma: {html_escape(str(CHROMA_DIR))}</span>
  </div>

  <div class="row" style="margin-top: 12px;">
        <form id="filters" hx-get="/api/search_html" hx-trigger="submit, change, keyup changed delay:350ms from:#q, keyup changed delay:350ms from:#folder" hx-target="#results" style="margin: 0; width: 100%; display: flex; flex-direction: column; gap: 10px;">
            <div class="row">
            <input
                id="q"
                type="text"
                name="q"
                placeholder="Cerca (es: 'tramonto sul mare', 'cane in montagna'...)"
                autocomplete="off"
            />

            <input id="folder" type="text" name="folder" placeholder="Filtro path contiene" style="width: 220px;" />
            <select id="mime" name="mime" style="padding: 10px 12px; border: 1px solid #ccc; border-radius: 10px; min-width: 220px;">
                <option value="" selected>Tutti i mime</option>
                {mime_options_html}
            </select>
            <span style="flex-basis: 100%; height: 0;"></span>
            <input id="city" type="text" name="city" placeholder="Città" style="width: 160px;" />
            <input id="region" type="text" name="region" placeholder="Regione" style="width: 160px;" />
            <input id="country_code" type="text" name="country_code" placeholder="Country code (es. IT)" style="width: 170px;" />
            <span style="flex-basis: 100%; height: 0;"></span>
            <input id="date_from" type="text" name="date_from" title="Data da" placeholder="GG/MM/AAAA" autocomplete="off" />
            <input id="date_to" type="text" name="date_to" title="Data a" placeholder="GG/MM/AAAA" autocomplete="off" />
            <span style="flex-basis: 100%; height: 0;"></span>

            <label style="display:flex; align-items:center; gap:6px;"><input type="checkbox" name="has_caption_en" value="true" />caption EN</label>
            <label style="display:flex; align-items:center; gap:6px;"><input type="checkbox" name="only_complete" value="true" />solo complete</label>
            </div>

            <div class="row">
                <input id="k" type="number" name="k" value="{DEFAULT_K}" min="1" max="{MAX_K}" title="Numero risultati (k)" />
                <select id="sort_by" name="sort_by" style="padding: 10px 12px; border: 1px solid #ccc; border-radius: 10px;">
                    <option value="semantic" selected>Sort: semantic</option>
                    <option value="date_desc">Sort: data ↓</option>
                    <option value="date_asc">Sort: data ↑</option>
                </select>
            </div>
        </form>
  </div>

  <div class="hint">
        Tip: query testuale + filtri strutturati (data/geo/località). Se lasci query vuota, cerca solo con i filtri SQLite.
  </div>

    <div class="row" style="margin-top: 8px;">
        <button id="delete-selected-btn" type="button" style="padding: 8px 12px; border-radius: 10px; border: 1px solid #ccc; background: #fff; cursor: pointer;" disabled>
            Elimina selezionate
        </button>
        <span id="selected-count" class="small" style="margin-top:0;">Selezionate: 0</span>
    </div>

  <div id="results"></div>

  <hr style="margin: 18px 0; border: none; border-top: 1px solid #eee;" />

  <h3 style="margin: 0 0 10px;">Cerca per immagine (upload)</h3>
  <form hx-post="/api/search_by_image_html" hx-target="#results_img" hx-encoding="multipart/form-data">
    <div class="row">
      <input type="file" name="file" accept="image/*" />
      <input type="number" name="k" value="{DEFAULT_K}" min="1" max="{MAX_K}" />
      <button type="submit" style="padding: 10px 12px; border-radius: 10px; border: 1px solid #ccc; background: white; cursor: pointer;">Search</button>
    </div>
  </form>
  <div id="results_img"></div>

    <dialog id="delete-dialog" style="max-width: min(860px, 95vw); width: 95vw; border: 1px solid #ddd; border-radius: 12px; padding: 14px;">
        <h3 style="margin: 0 0 8px;">Conferma cancellazione</h3>
        <div class="small" style="margin: 0 0 8px;">Stai per cancellare queste immagini da disco, SQLite, Chroma e thumbnail:</div>
        <ul id="delete-list" style="max-height: 40vh; overflow: auto; margin: 0 0 12px 18px;"></ul>
        <div class="row" style="justify-content: flex-end;">
            <button id="delete-cancel-btn" type="button" style="padding: 8px 12px; border-radius: 10px; border: 1px solid #ccc; background: #fff; cursor: pointer;">Annulla</button>
            <button id="delete-confirm-btn" type="button" style="padding: 8px 12px; border-radius: 10px; border: 1px solid #c33; background: #d33; color: #fff; cursor: pointer;">Conferma eliminazione</button>
        </div>
    </dialog>

    <script>
        (function () {{
            if (!window.flatpickr) return;

            if (window.flatpickr.l10ns && window.flatpickr.l10ns.it) {{
                window.flatpickr.localize(window.flatpickr.l10ns.it);
            }}

            const opts = {{
                allowInput: true,
                dateFormat: "Y-m-d",
                altInput: true,
                altFormat: "d/m/Y",
            }};

            window.flatpickr("#date_from", opts);
            window.flatpickr("#date_to", opts);

            const selected = new Map();
            const selectedCount = document.getElementById("selected-count");
            const deleteBtn = document.getElementById("delete-selected-btn");
            const dialog = document.getElementById("delete-dialog");
            const deleteList = document.getElementById("delete-list");
            const deleteCancelBtn = document.getElementById("delete-cancel-btn");
            const deleteConfirmBtn = document.getElementById("delete-confirm-btn");
            const results = document.getElementById("results");
            const filters = document.getElementById("filters");

            function updateSelectionUI() {{
                const n = selected.size;
                if (selectedCount) selectedCount.textContent = `Selezionate: ${{n}}`;
                if (deleteBtn) deleteBtn.disabled = n === 0;
            }}

            function applySelectionToDOM() {{
                if (!results) return;
                const checks = results.querySelectorAll("input.photo-select[type='checkbox']");
                checks.forEach((cb) => {{
                    const sid = cb.getAttribute("data-sha1") || "";
                    cb.checked = selected.has(sid);
                    const card = cb.closest(".card");
                    if (card) card.classList.toggle("selected", cb.checked);
                }});
            }}

            function openDeleteDialog() {{
                if (!dialog || !deleteList || selected.size === 0) return;
                deleteList.innerHTML = "";
                Array.from(selected.values()).forEach((item) => {{
                    const li = document.createElement("li");
                    li.textContent = item.path || item.sha1;
                    deleteList.appendChild(li);
                }});
                if (typeof dialog.showModal === "function") {{
                    dialog.showModal();
                }} else {{
                    alert("Conferma cancellazione:\\n" + Array.from(selected.values()).map((x) => x.path || x.sha1).join("\\n"));
                }}
            }}

            async function confirmDelete() {{
                if (selected.size === 0) return;
                const ids = Array.from(selected.keys());
                deleteConfirmBtn.disabled = true;
                try {{
                    const res = await fetch("/api/delete_images", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify({{ ids }}),
                    }});
                    if (!res.ok) {{
                        const txt = await res.text();
                        throw new Error(txt || `HTTP ${{res.status}}`);
                    }}
                    const payload = await res.json();
                    if (dialog && typeof dialog.close === "function") dialog.close();
                    selected.clear();
                    updateSelectionUI();
                    if (filters && window.htmx) {{
                        window.htmx.trigger(filters, "change");
                    }}
                    alert(`Eliminazione completata.\\nDB: ${{payload.deleted_db}} | Chroma: ${{payload.deleted_chroma}} | File: ${{payload.deleted_files}} | Thumbs: ${{payload.deleted_thumbs}}`);
                }} catch (err) {{
                    alert("Errore in cancellazione: " + (err?.message || err));
                }} finally {{
                    deleteConfirmBtn.disabled = false;
                }}
            }}

            if (results) {{
                results.addEventListener("change", (ev) => {{
                    const target = ev.target;
                    if (!(target instanceof HTMLInputElement)) return;
                    if (!target.classList.contains("photo-select")) return;
                    const sid = target.getAttribute("data-sha1") || "";
                    const path = target.getAttribute("data-path") || sid;
                    if (!sid) return;
                    if (target.checked) selected.set(sid, {{ sha1: sid, path }});
                    else selected.delete(sid);
                    const card = target.closest(".card");
                    if (card) card.classList.toggle("selected", target.checked);
                    updateSelectionUI();
                }});
            }}

            if (deleteBtn) deleteBtn.addEventListener("click", openDeleteDialog);
            if (deleteCancelBtn) deleteCancelBtn.addEventListener("click", () => {{ if (dialog?.close) dialog.close(); }});
            if (deleteConfirmBtn) deleteConfirmBtn.addEventListener("click", confirmDelete);

            document.body.addEventListener("htmx:afterSwap", (ev) => {{
                const target = ev?.detail?.target;
                if (target && target.id === "results") applySelectionToDOM();
            }});

            updateSelectionUI();
        }})();
    </script>
</body>
</html>
"""


def render_cards(ids: List[str], dists: List[float], metas: List[Dict[str, Any]]) -> str:
    rows_data: List[Dict[str, Any]] = []
    for _id, dist, meta in zip(ids, dists, metas):
        meta = meta or {}
        p = pick_image_path(meta)
        if not p or not p.exists():
            continue

        # Restrict serving if requested
        if RESTRICT_TO_PHOTOS_DIR and not within_photos_dir(p):
            continue

        mime_str = str(meta.get("mime") or "").lower() or None
        is_vid = is_video_media(p, mime_str)
        thumb = ensure_thumb(_id, p, mime=mime_str)
        if not thumb and not is_vid:
            # Image thumbnail generation failed — skip the card
            continue
        # Videos without a thumbnail still get a card (placeholder shown)

        # Prefer relpath for display; fallback to absolute
        display_path = meta.get("relpath") or meta.get("path") or str(p)

        taken_at = meta.get("taken_at")
        if isinstance(taken_at, (int, float)):
            taken_at_str = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(taken_at))
        else:
            taken_at_str = "?"

        place_name = meta.get("place_name") or ""
        dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "-"

        rows_data.append(
            {
                "sha1": _id,
                "dist_str": dist_str,
                "display_path": display_path,
                "taken_at_str": taken_at_str,
                "place_name": str(place_name),
                "has_thumb": thumb is not None,
                "is_video": is_vid,
            }
        )

    if not rows_data:
        return "<div class='small'>Nessun risultato (o file non accessibili / fuori da PHOTOS_DIR).</div>"

    ctx = store_search_context([r["sha1"] for r in rows_data])
    cards = []
    for idx, row in enumerate(rows_data):
        sid = row["sha1"]
        detail_href = f"/photo/{html_escape(sid)}?ctx={html_escape(ctx)}&pos={idx}"

        if row["has_thumb"]:
            thumb_html = f'<img class="thumb" src="/thumb/{html_escape(sid)}.jpg" alt="thumb" loading="lazy">'
        else:
            # Video without a pre-generated thumbnail
            thumb_html = (
                '<div class="thumb" style="display:flex;align-items:center;justify-content:center;'
                'background:#111;color:#fff;font-size:2rem;">&#9654;</div>'
            )
        cards.append(f"""
          <div class="card">
                        <label class="small" style="display:flex; align-items:center; gap:6px; margin-top:0;">
                            <input type="checkbox" class="photo-select" data-sha1="{html_escape(sid)}" data-path="{html_escape(str(row['display_path']))}" />
                            seleziona
                        </label>
                        <a href="{detail_href}" target="_blank" rel="noreferrer">
              {thumb_html}
            </a>
                        <div class="dist"><b>dist</b>: {row['dist_str']}</div>
            <div class="small">{html_escape(row['display_path'])}</div>
                        <div class="small">data: {html_escape(row['taken_at_str'])}</div>
                        <div class="small">luogo: {html_escape(row['place_name'])}</div>
          </div>
        """)

    return f"<div class='grid'>{''.join(cards)}</div>"


@app.get("/api/search_html", response_class=HTMLResponse)
def search_html(
    q: str = Query(default="", min_length=0),
    k: int = Query(default=DEFAULT_K, ge=1, le=MAX_K),
    page: int = Query(default=1, ge=1),
    folder: str = Query(default="", min_length=0),
    mime: str = Query(default="", min_length=0),
    country_code: str = Query(default="", min_length=0),
    region: str = Query(default="", min_length=0),
    city: str = Query(default="", min_length=0),
    date_from: str = Query(default="", min_length=0),
    date_to: str = Query(default="", min_length=0),
    has_caption_en: bool = Query(default=False),
    only_complete: bool = Query(default=False),
    sort_by: str = Query(default="semantic", pattern="^(semantic|date_desc|date_asc)$"),
):
    q = (q or "").strip()

    candidate_ids: Optional[List[str]] = None
    dist_map: Dict[str, float] = {}
    chroma_meta_map: Dict[str, Dict[str, Any]] = {}

    # Videos are not embedded in Chroma (BASE_ONLY_MIME_TYPES).
    # If the user explicitly filters by a video MIME type, bypass Chroma entirely
    # so that all matching DB rows are returned regardless of embedding status.
    mime_clean_for_check = (mime or "").strip().lower()
    skip_chroma = is_video_mime(mime_clean_for_check)
    use_semantic_candidates = bool(q) and sort_by == "semantic" and not skip_chroma

    if use_semantic_candidates:
        candidate_k = int(min(max(k * 5, 200), MAX_CANDIDATES))
        col = chroma_collection()
        vec = embed_text(MODEL, TOKENIZER, DEVICE, q)
        res = col.query(
            query_embeddings=[vec],
            n_results=candidate_k,
            include=["metadatas", "distances"],
        )

        candidate_ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        for sid, dist, meta in zip(candidate_ids, dists, metas):
            dist_map[sid] = dist
            chroma_meta_map[sid] = meta or {}

    rows = fetch_images_filtered(
        candidate_ids=candidate_ids if use_semantic_candidates else None,
        folder=folder,
        mime=mime,
        country_code=country_code,
        region=region,
        city=city,
        date_from=date_from,
        date_to=date_to,
        has_caption_en=has_caption_en,
        only_complete=only_complete,
    )

    if not rows:
        return "<div class='small'>Nessun risultato con i filtri correnti.</div>"

    total_rows = len(rows)
    row_map = {r["sha1"]: r for r in rows}

    if use_semantic_candidates:
        # Semantic ranking from Chroma distances
        ordered_ids = [sid for sid in (candidate_ids or []) if sid in row_map]
    elif sort_by == "date_asc":
        ordered_rows = sorted(
            rows,
            key=lambda r: ((r["taken_at"] is None), (r["taken_at"] or 0), r["path"] or ""),
        )
        ordered_ids = [r["sha1"] for r in ordered_rows]
    else:
        # date_desc, or "semantic" with no query / with skip_chroma → show most recent first
        ordered_rows = sorted(
            rows,
            key=lambda r: ((r["taken_at"] is None), -(r["taken_at"] or 0), r["path"] or ""),
        )
        ordered_ids = [r["sha1"] for r in ordered_rows]

    total_pages = max(1, (len(ordered_ids) + k - 1) // k)
    page = min(max(page, 1), total_pages)
    start_idx = (page - 1) * k
    end_idx = start_idx + k
    ordered_ids = ordered_ids[start_idx:end_idx]

    out_ids: List[str] = []
    out_dists: List[float] = []
    out_metas: List[Dict[str, Any]] = []

    for sid in ordered_ids:
        r = row_map.get(sid)
        if not r:
            continue

        meta = dict(chroma_meta_map.get(sid) or {})
        meta["sha1"] = sid
        meta["relpath"] = r["path"]
        meta["path"] = r["path"]
        meta["mime"] = r["mime"]
        meta["w"] = r["w"]
        meta["h"] = r["h"]
        meta["mtime"] = r["mtime"]
        meta["taken_at"] = r["taken_at"]
        meta["gps_lat"] = r["gps_lat"]
        meta["gps_lon"] = r["gps_lon"]
        meta["country"] = r["country"]
        meta["country_code"] = r["country_code"]
        meta["region"] = r["region"]
        meta["city"] = r["city"]
        meta["place_name"] = r["place_name"]
        meta["location_source"] = r["location_source"]

        out_ids.append(sid)
        out_dists.append(dist_map.get(sid, 0.0 if q else float("nan")))
        out_metas.append(meta)

    cards_html = render_cards(out_ids, out_dists, out_metas)

    start_label = start_idx + 1 if total_rows else 0
    end_label = min(end_idx, total_rows)
    pager_parts = [
        f"<div class='small' style='margin:8px 0 12px 0;'>Risultati: {start_label}-{end_label} di {total_rows} • pagina {page}/{total_pages}</div>"
    ]
    if total_pages > 1:
        prev_disabled = "disabled" if page <= 1 else ""
        next_disabled = "disabled" if page >= total_pages else ""
        prev_page = max(1, page - 1)
        next_page = min(total_pages, page + 1)
        pager_parts.append(
            "<div class='row' style='margin: 0 0 12px 0;'>"
            f"<button type='button' {prev_disabled} hx-get='/api/search_html?page={prev_page}' hx-include='#filters' hx-target='#results' style='padding: 8px 12px; border-radius: 10px; border: 1px solid #ccc; background: #fff; cursor: pointer;'>← Precedenti</button>"
            f"<button type='button' {next_disabled} hx-get='/api/search_html?page={next_page}' hx-include='#filters' hx-target='#results' style='padding: 8px 12px; border-radius: 10px; border: 1px solid #ccc; background: #fff; cursor: pointer;'>Successivi →</button>"
            "</div>"
        )

    return "".join(pager_parts) + cards_html


@app.post("/api/delete_images")
def api_delete_images(payload: Dict[str, Any] = Body(...)):
    ids = payload.get("ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list):
        raise HTTPException(400, "payload must contain 'ids' list")

    cleaned = [str(x).strip() for x in ids if str(x).strip()]
    if not cleaned:
        raise HTTPException(400, "no ids provided")

    result = delete_images_everywhere(cleaned)
    return result


@app.get("/photo/{sha1}", response_class=HTMLResponse)
def photo_detail(
    sha1: str,
    ctx: str = Query(default="", min_length=0),
    pos: int = Query(default=-1),
):
        detail = get_photo_detail(sha1)
        if not detail:
                raise HTTPException(404, "photo not found")

        image = detail["image"]
        mime_value = str(image["mime"] or "").lower()
        rel_path = image["path"] or ""
        is_video = is_video_media(Path(str(rel_path)), mime_value)
        caption = detail["caption"]
        tags = detail["tags"]
        jobs = detail["jobs"]
        prev_sha1 = detail["prev_sha1"]
        next_sha1 = detail["next_sha1"]
        first_sha1 = detail["first_sha1"]
        last_sha1 = detail["last_sha1"]
        current_pos = -1

        rel_path = image["path"] or ""
        taken_at_str = format_epoch_local(image["taken_at"])
        added_at_str = format_epoch_local(image["added_at"])
        mtime_str = format_epoch_local(image["mtime"])

        caption_text = caption["caption"] if caption else "-"
        caption_lang = caption["lang"] if caption and caption["lang"] else "-"
        caption_model = caption["model"] if caption and caption["model"] else "-"

        tags_html = "".join(
                f"<li>{html_escape(str(t['tag']))}"
                f" (score: {html_escape('-' if t['score'] is None else str(round(float(t['score']), 4)))}, "
                f"source: {html_escape(str(t['source'] or '-'))})</li>"
                for t in tags
        ) or "<li>-</li>"

        jobs_html = "".join(
                f"<li>{html_escape(str(j['step']))}: {html_escape(str(j['status']))}"
                f" | aggiornato: {html_escape(format_epoch_local(j['updated_at']))}"
                f" | dettaglio: {html_escape(str((j['detail'] or '')[:160]))}</li>"
                for j in jobs
        ) or "<li>-</li>"

        ctx_ids = load_search_context(ctx)
        if ctx_ids:
            idx = -1
            if 0 <= pos < len(ctx_ids) and ctx_ids[pos] == sha1:
                idx = pos
            else:
                try:
                    idx = ctx_ids.index(sha1)
                except ValueError:
                    idx = -1

            if idx >= 0:
                prev_sha1 = ctx_ids[idx - 1] if idx > 0 else None
                next_sha1 = ctx_ids[idx + 1] if idx < len(ctx_ids) - 1 else None
                first_sha1 = ctx_ids[0] if ctx_ids else None
                last_sha1 = ctx_ids[-1] if ctx_ids else None
                current_pos = idx

        original_href = f"/viewer/{html_escape(sha1)}"
        if ctx_ids and current_pos >= 0:
            original_href = f"/viewer/{html_escape(sha1)}?ctx={html_escape(ctx)}&pos={current_pos}"

        def nav_href(target_sha1: Optional[str], target_pos: int) -> Optional[str]:
            if not target_sha1:
                return None
            if ctx_ids:
                return f"/photo/{html_escape(target_sha1)}?ctx={html_escape(ctx)}&pos={target_pos}"
            return f"/photo/{html_escape(target_sha1)}"

        nav_pos = current_pos if current_pos >= 0 else pos
        prev_href = nav_href(prev_sha1, (nav_pos - 1) if nav_pos > 0 else -1)
        next_href = nav_href(next_sha1, (nav_pos + 1) if nav_pos >= 0 else -1)
        first_href = nav_href(first_sha1, 0)
        last_href = nav_href(last_sha1, (len(ctx_ids) - 1) if ctx_ids else -1)
        first_swipe_href = first_href or ""
        last_swipe_href = last_href or ""

        first_btn = (
            f"<a class='btn' href='{first_href}'>⏮ Inizio</a>"
            if first_href
            else "<span class='btn disabled'>⏮ Inizio</span>"
        )
        last_btn = (
            f"<a class='btn' href='{last_href}'>Fine ⏭</a>"
            if last_href
            else "<span class='btn disabled'>Fine ⏭</span>"
        )

        prev_btn = (
            f"<a class='btn' href='{prev_href}'>← Precedente</a>"
            if prev_href
                else "<span class='btn disabled'>← Precedente</span>"
        )
        next_btn = (
            f"<a class='btn' href='{next_href}'>Successiva →</a>"
            if next_href
                else "<span class='btn disabled'>Successiva →</span>"
        )

        media_html = (
            f"<video id='detail-photo' class='photo' src='/img/{html_escape(sha1)}' controls playsinline preload='metadata' style='display:block;'></video>"
            if is_video
            else f"<img id='detail-photo' class='photo' src='/img/{html_escape(sha1)}' alt='photo' style='display:block;' />"
        )

        return f"""<!doctype html>
<html lang="it-IT">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Photo {html_escape(sha1)}</title>
    <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 18px; }}
        .top {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }}
        .btn {{ border: 1px solid #ccc; border-radius: 10px; padding: 8px 12px; text-decoration: none; color: #222; background: #fff; }}
        .disabled {{ color: #888; background: #f3f3f3; border-color: #ddd; }}
        .photo {{ width: 100%; max-height: 80vh; object-fit: contain; background: #f5f5f5; border-radius: 10px; border: 1px solid #ddd; }}
        .box {{ margin-top: 14px; border: 1px solid #e2e2e2; border-radius: 10px; padding: 12px; }}
        .k {{ color: #666; font-size: 13px; }}
        ul {{ margin: 8px 0 0 20px; }}
    </style>
</head>
<body>
    <div class="top">
        {first_btn}
        {prev_btn}
        {next_btn}
        {last_btn}
        <a class="btn" href="{original_href}" target="_self">Apri file originale</a>
    </div>

    {media_html}

    <div class="box">
        <h3 style="margin:0 0 10px;">Informazioni</h3>
        <div><span class="k">sha1:</span> {html_escape(str(image['sha1']))}</div>
        <div><span class="k">path:</span> {html_escape(str(rel_path))}</div>
        <div><span class="k">dimensioni:</span> {html_escape(str(image['w']))}×{html_escape(str(image['h']))}</div>
        <div><span class="k">file_size:</span> {html_escape(str(image['file_size']))}</div>
        <div><span class="k">mime:</span> {html_escape(str(image['mime'] or '-'))}</div>
        <div><span class="k">durata:</span> {html_escape(_fmt_duration(image['duration'] if 'duration' in image.keys() else None))}</div>
        <div><span class="k">data:</span> {html_escape(taken_at_str)}</div>
        <div><span class="k">mtime:</span> {html_escape(mtime_str)}</div>
        <div><span class="k">aggiunta:</span> {html_escape(added_at_str)}</div>
        <div><span class="k">gps:</span> {html_escape(str(image['gps_lat']))}, {html_escape(str(image['gps_lon']))} (alt: {html_escape(str(image['gps_alt']))})</div>
        <div><span class="k">luogo:</span> {html_escape(str(image['city'] or '-'))}, {html_escape(str(image['region'] or '-'))}, {html_escape(str(image['country'] or '-'))} [{html_escape(str(image['country_code'] or '-'))}]</div>
        <div><span class="k">place_name:</span> {html_escape(str(image['place_name'] or '-'))}</div>
        <div><span class="k">location_source:</span> {html_escape(str(image['location_source'] or '-'))}</div>
    </div>

    <div class="box">
        <h3 style="margin:0 0 10px;">Caption</h3>
        <div><span class="k">lang:</span> {html_escape(str(caption_lang))}</div>
        <div><span class="k">model:</span> {html_escape(str(caption_model))}</div>
        <div style="margin-top:8px;">{html_escape(str(caption_text))}</div>
    </div>

    <div class="box">
        <h3 style="margin:0 0 10px;">Tags</h3>
        <ul>{tags_html}</ul>
    </div>

    <div class="box">
        <h3 style="margin:0 0 10px;">Jobs</h3>
        <ul>{jobs_html}</ul>
    </div>

        <script>
            (function () {{
                const prevHref = {('"' + prev_href + '"') if prev_href else 'null'};
                const nextHref = {('"' + next_href + '"') if next_href else 'null'};
                const firstHref = {('"' + first_swipe_href + '"') if first_swipe_href else 'null'};
                const lastHref = {('"' + last_swipe_href + '"') if last_swipe_href else 'null'};
                const originalHref = "{original_href}";

                function goPrev() {{ if (prevHref) window.location.href = prevHref; }}
                function goNext() {{ if (nextHref) window.location.href = nextHref; }}
                function goFirst() {{ if (firstHref) window.location.href = firstHref; }}
                function goLast() {{ if (lastHref) window.location.href = lastHref; }}

                document.addEventListener("keydown", (ev) => {{
                    if (ev.key === "ArrowLeft") goPrev();
                    if (ev.key === "ArrowRight") goNext();
                    if (ev.key === "Home") goFirst();
                    if (ev.key === "End") goLast();
                }});

                const img = document.getElementById("detail-photo");
                if (!img) return;
                const isVideo = img.tagName === "VIDEO";

                if (isVideo) {{
                    return;
                }}

                let startX = 0;
                let startY = 0;
                let tracking = false;
                let suppressTapUntil = 0;

                img.addEventListener("touchstart", (ev) => {{
                    const t = ev.changedTouches && ev.changedTouches[0];
                    if (!t) return;
                    tracking = true;
                    startX = t.clientX;
                    startY = t.clientY;
                }}, {{ passive: true }});

                img.addEventListener("touchend", (ev) => {{
                    if (!tracking) return;
                    tracking = false;
                    const t = ev.changedTouches && ev.changedTouches[0];
                    if (!t) return;

                    const dx = t.clientX - startX;
                    const dy = t.clientY - startY;
                    if (Math.abs(dx) < 40 || Math.abs(dx) < Math.abs(dy)) return;
                    suppressTapUntil = Date.now() + 400;
                    if (dx < 0) goNext();
                    else goPrev();
                }}, {{ passive: true }});

                img.addEventListener("click", (ev) => {{
                    if (Date.now() < suppressTapUntil) {{
                        ev.preventDefault();
                        return;
                    }}
                    window.location.href = originalHref;
                }});
            }})();
        </script>
</body>
</html>
"""


@app.get("/viewer/{sha1}", response_class=HTMLResponse)
def photo_viewer(
        sha1: str,
        ctx: str = Query(default="", min_length=0),
        pos: int = Query(default=-1),
):
        detail = get_photo_detail(sha1)
        if not detail:
                raise HTTPException(404, "photo not found")

        image = detail["image"]
        mime_value = str(image["mime"] or "").lower()
        rel_path = image["path"] or ""
        is_video = is_video_media(Path(str(rel_path)), mime_value)
        prev_sha1 = detail["prev_sha1"]
        next_sha1 = detail["next_sha1"]
        current_pos = -1
        first_sha1 = detail["first_sha1"]
        last_sha1 = detail["last_sha1"]

        ctx_ids = load_search_context(ctx)
        if ctx_ids:
            idx = -1
            if 0 <= pos < len(ctx_ids) and ctx_ids[pos] == sha1:
                idx = pos
            else:
                try:
                    idx = ctx_ids.index(sha1)
                except ValueError:
                    idx = -1

            if idx >= 0:
                prev_sha1 = ctx_ids[idx - 1] if idx > 0 else None
                next_sha1 = ctx_ids[idx + 1] if idx < len(ctx_ids) - 1 else None
                first_sha1 = ctx_ids[0] if ctx_ids else None
                last_sha1 = ctx_ids[-1] if ctx_ids else None
                current_pos = idx

        def viewer_href(target_sha1: Optional[str], target_pos: int) -> str:
            if not target_sha1:
                return ""
            if ctx_ids and target_pos >= 0:
                return f"/viewer/{html_escape(target_sha1)}?ctx={html_escape(ctx)}&pos={target_pos}"
            return f"/viewer/{html_escape(target_sha1)}"

        prev_href = viewer_href(prev_sha1, (current_pos - 1) if current_pos > 0 else -1)
        next_href = viewer_href(next_sha1, (current_pos + 1) if current_pos >= 0 else -1)
        first_href = viewer_href(first_sha1, 0 if current_pos >= 0 else -1)
        last_href = viewer_href(last_sha1, (len(ctx_ids) - 1) if ctx_ids else -1)
        back_href = f"/photo/{html_escape(sha1)}"
        if ctx_ids and current_pos >= 0:
            back_href = f"/photo/{html_escape(sha1)}?ctx={html_escape(ctx)}&pos={current_pos}"

        media_html = (
            f"<video id='viewer-media' class='viewer-media' src='/img/{html_escape(sha1)}' controls playsinline autoplay preload='metadata'></video>"
            if is_video
            else f"<img id='viewer-media' class='viewer-media' src='/img/{html_escape(sha1)}' alt='photo' />"
        )

        return f"""<!doctype html>
<html lang="it-IT">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Viewer {html_escape(sha1)}</title>
    <style>
        html, body {{ margin: 0; height: 100%; background: #000; overflow: hidden; }}
        .wrap {{ position: fixed; inset: 0; display: flex; align-items: center; justify-content: center; }}
        .viewer-media {{ width: 100%; height: 100%; object-fit: contain; user-select: none; -webkit-user-drag: none; }}
        .top {{ position: fixed; top: 10px; left: 10px; right: 10px; display: flex; justify-content: space-between; gap: 8px; z-index: 10; }}
        .btn {{ border: 1px solid rgba(255,255,255,.35); border-radius: 999px; padding: 8px 12px; text-decoration: none; color: #fff; background: rgba(0,0,0,.45); font-size: 14px; }}
        .btn.disabled {{ opacity: .45; pointer-events: none; }}
        .side {{ position: fixed; top: 0; bottom: 0; width: 22%; z-index: 5; }}
        .side.left {{ left: 0; }}
        .side.right {{ right: 0; }}
    </style>
</head>
<body>
    <div class="top">
        <a class="btn" href="{back_href}">← Dettaglio</a>
        <div style="display:flex; gap:8px;">
            <button id="fullscreen-btn" class="btn" type="button">⛶</button>
            <a class="btn {'disabled' if not first_href else ''}" href="{first_href or '#'}">⏮</a>
            <a class="btn {'disabled' if not prev_href else ''}" href="{prev_href or '#'}">←</a>
            <a class="btn {'disabled' if not next_href else ''}" href="{next_href or '#'}">→</a>
            <a class="btn {'disabled' if not last_href else ''}" href="{last_href or '#'}">⏭</a>
        </div>
    </div>

    <a class="side left" href="{prev_href or '#'}" {'style="pointer-events:none"' if not prev_href else ''}></a>
    <a class="side right" href="{next_href or '#'}" {'style="pointer-events:none"' if not next_href else ''}></a>

    <div class="wrap" id="viewer-wrap">
        {media_html}
    </div>

    <script>
        (function () {{
            const prevHref = {('"' + prev_href + '"') if prev_href else 'null'};
            const nextHref = {('"' + next_href + '"') if next_href else 'null'};
            const firstHref = {('"' + first_href + '"') if first_href else 'null'};
            const lastHref = {('"' + last_href + '"') if last_href else 'null'};
            const backHref = "{back_href}";
            const fullscreenBtn = document.getElementById("fullscreen-btn");

            let startX = 0;
            let startY = 0;
            let tracking = false;
            let suppressTapUntil = 0;

            function goPrev() {{ if (prevHref) window.location.href = prevHref; }}
            function goNext() {{ if (nextHref) window.location.href = nextHref; }}
            function goFirst() {{ if (firstHref) window.location.href = firstHref; }}
            function goLast() {{ if (lastHref) window.location.href = lastHref; }}
            async function toggleFullscreen() {{
                const doc = document;
                const el = document.documentElement;
                const isFs = !!(doc.fullscreenElement || doc.webkitFullscreenElement || doc.msFullscreenElement);
                try {{
                    if (isFs) {{
                        if (doc.exitFullscreen) await doc.exitFullscreen();
                        else if (doc.webkitExitFullscreen) doc.webkitExitFullscreen();
                        else if (doc.msExitFullscreen) doc.msExitFullscreen();
                    }} else {{
                        if (el.requestFullscreen) await el.requestFullscreen();
                        else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
                        else if (el.msRequestFullscreen) el.msRequestFullscreen();
                    }}
                }} catch (_err) {{
                    // ignore: browser may refuse fullscreen or not support it
                }}
            }}

            document.addEventListener("keydown", (ev) => {{
                if (ev.key === "ArrowLeft") goPrev();
                if (ev.key === "ArrowRight") goNext();
                if (ev.key === "Home") goFirst();
                if (ev.key === "End") goLast();
                if (ev.key.toLowerCase() === "f") toggleFullscreen();
            }});

            const wrap = document.getElementById("viewer-wrap");
            if (!wrap) return;

            if (fullscreenBtn) {{
                fullscreenBtn.addEventListener("click", (ev) => {{
                    ev.preventDefault();
                    toggleFullscreen();
                }});
            }}

            wrap.addEventListener("touchstart", (ev) => {{
                const t = ev.changedTouches && ev.changedTouches[0];
                if (!t) return;
                tracking = true;
                startX = t.clientX;
                startY = t.clientY;
            }}, {{ passive: true }});

            wrap.addEventListener("touchend", (ev) => {{
                if (!tracking) return;
                const t = ev.changedTouches && ev.changedTouches[0];
                tracking = false;
                if (!t) return;

                const dx = t.clientX - startX;
                const dy = t.clientY - startY;
                if (Math.abs(dx) < 40 || Math.abs(dx) < Math.abs(dy)) return;
                suppressTapUntil = Date.now() + 400;
                if (dx < 0) goNext();
                else goPrev();
            }}, {{ passive: true }});

            wrap.addEventListener("click", (ev) => {{
                if (Date.now() < suppressTapUntil) {{
                    ev.preventDefault();
                    return;
                }}
                if (ev.target && ev.target.closest && ev.target.closest("video")) {{
                    return;
                }}
                window.location.href = backHref;
            }});
        }})();
    </script>
</body>
</html>
"""


@app.post("/api/search_by_image_html", response_class=HTMLResponse)
def search_by_image_html(
    file: UploadFile = File(...),
    k: int = Query(default=DEFAULT_K, ge=1, le=MAX_K),
):
    try:
        content = file.file.read()
        from io import BytesIO
        im = Image.open(BytesIO(content))
        im = orient_image(im)
    except Exception:
        raise HTTPException(400, "Immagine non valida")

    vec = embed_image(MODEL, PREPROCESS, DEVICE, im)

    col = chroma_collection()
    res = col.query(
        query_embeddings=[vec],
        n_results=int(min(k, MAX_K)),
        include=["metadatas", "distances"],
    )
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return render_cards(ids, dists, metas)


@app.get("/thumb/{sha1}.jpg")
def thumb(sha1: str):
    p = THUMB_DIR / f"{sha1}.jpg"
    if not p.exists():
        raise HTTPException(404, "thumb not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/img/{sha1}")
def img(sha1: str):
    p = None
    db_mime: Optional[str] = None
    if SQLITE_DB_PATH.exists():
        con = sqlite_connect()
        try:
            row = con.execute("SELECT path, mime FROM images WHERE sha1 = ? LIMIT 1", (sha1,)).fetchone()
        finally:
            con.close()
        if row:
            p = resolve_image_path_from_rel(row["path"])
            db_mime = str(row["mime"] or "").strip() or None

    if p is None:
        col = chroma_collection()
        got = col.get(ids=[sha1], include=["metadatas"])
        if not got.get("ids"):
            raise HTTPException(404, "id not found")
        meta = (got.get("metadatas") or [None])[0] or {}
        p = pick_image_path(meta)

    if not p or not p.exists():
        raise HTTPException(404, "file missing")

    if RESTRICT_TO_PHOTOS_DIR and not within_photos_dir(p):
        raise HTTPException(403, "file outside PHOTOS_DIR")

    suffix = p.suffix.lower()
    if suffix in {".heic", ".heif"}:
        try:
            from io import BytesIO

            im = load_and_orient_image(p).convert("RGB")
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=90)
            return Response(content=buf.getvalue(), media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(500, f"cannot convert HEIC/HEIF: {e}")

    # This serves the original media file. Keep the server bound to localhost.
    if db_mime:
        return FileResponse(str(p), media_type=db_mime)
    return FileResponse(str(p))

