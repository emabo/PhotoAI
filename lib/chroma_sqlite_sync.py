#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path, PurePosixPath
from typing import Dict, Any, List, Tuple, Optional

import chromadb
from chromadb.config import Settings


def must_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set (define it in .env or environment variables).")
    return v


def re_drive_letter(p: str) -> bool:
    return len(p) >= 2 and p[1] == ":" and p[0].isalpha()


def normalize_relpath(p: str) -> Optional[str]:
    """
    Normalize a relative path in POSIX style and reject unsafe paths.
    """
    if not p:
        return None

    p = p.strip().replace("\\", "/")

    # reject absolute paths
    if p.startswith("/") or re_drive_letter(p):
        return None

    pp = PurePosixPath(p)

    parts = []
    for part in pp.parts:
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


def chroma_collection(chroma_dir: Path, collection: str):
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(collection)


def sqlite_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def upsert_images(con: sqlite3.Connection, rows: List[Tuple]):
    con.executemany(
        """
        INSERT INTO images (sha1, path, mtime, w, h, file_size, mime)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sha1) DO UPDATE SET
          path=excluded.path,
          mtime=excluded.mtime,
          w=excluded.w,
          h=excluded.h,
          file_size=excluded.file_size,
          mime=excluded.mime
        """,
        rows,
    )


def ensure_jobs(con: sqlite3.Connection, sha1_list: List[str], steps: List[str], status: str = "queued"):
    if not steps:
        return
    payload = [(sha1, step, status) for sha1 in sha1_list for step in steps]
    con.executemany(
        """
        INSERT OR IGNORE INTO jobs (sha1, step, status)
        VALUES (?, ?, ?)
        """,
        payload,
    )




