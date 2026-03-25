#!/usr/bin/env python3
import mimetypes
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional


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


def sqlite_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def detect_mime(path: Path) -> Optional[str]:
    # mimetypes.guess_type is case-sensitive on Linux; normalise the extension
    path_for_guess = path.with_suffix(path.suffix.lower())
    guessed, _enc = mimetypes.guess_type(str(path_for_guess), strict=False)
    if guessed:
        return guessed

    try:
        from PIL import Image
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except Exception:
            pass

        with Image.open(path) as img:
            fmt = (img.format or "").upper()
        if fmt:
            return Image.MIME.get(fmt)
    except Exception:
        return None

    return None


@dataclass
class Stats:
    scanned: int = 0
    updated: int = 0
    skipped_has_mime: int = 0
    missing_file: int = 0
    invalid_path: int = 0
    undetected: int = 0



