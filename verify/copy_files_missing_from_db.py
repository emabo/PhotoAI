#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv


SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/heic",
    "image/heif",
    "video/mp4",
    "video/x-msvideo",
    "video/avi",
    "video/mpeg",
}


def must_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set (define it in .env or environment variables).")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visita ricorsivamente tutti i file di una directory sorgente, calcola la SHA1 e copia "
            "solo quelli non presenti nel DB SQLite mantenendo il path relativo originale."
        )
    )
    parser.add_argument(
        "--from",
        dest="src_dir",
        required=True,
        help="Directory sorgente da scandire ricorsivamente.",
    )
    parser.add_argument(
        "--to",
        dest="dst_dir",
        required=True,
        help="Directory destinazione dove copiare i file mancanti dal DB.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Non copia nulla, stampa solo cosa verrebbe copiato.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Numero massimo di file da visitare (0 = nessun limite).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive i file già esistenti in destinazione. Default: skip.",
    )
    return parser.parse_args()


def connect_db() -> sqlite3.Connection:
    sqlite_dir = Path(must_env("PHOTOAI_SQLITE_DIR")).expanduser()
    db_path = sqlite_dir / "photo_ai.sqlite"
    if not db_path.exists():
        raise RuntimeError(f"DB not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys=ON;")
    con.row_factory = sqlite3.Row
    return con


def load_db_sha1_to_path(con: sqlite3.Connection) -> Dict[str, str]:
    rows = con.execute(
        """
        SELECT sha1, path
        FROM images
        WHERE sha1 IS NOT NULL
          AND trim(sha1) <> ''
          AND path IS NOT NULL
          AND trim(path) <> ''
        """
    )
    out: Dict[str, str] = {}
    for row in rows:
        sha1 = str(row[0]).strip()
        rel_path = str(row[1]).replace("\\", "/").strip().strip("/")
        if sha1 and rel_path and sha1 not in out:
            out[sha1] = rel_path
    return out


def db_file_exists_and_nonempty(sha1: str, sha1_to_db_path: Dict[str, str], photos_root: Optional[Path]) -> bool:
    if photos_root is None:
        return True

    rel_path = sha1_to_db_path.get(sha1)
    if not rel_path:
        return False

    db_fs_path = photos_root / rel_path
    if not db_fs_path.exists() or not db_fs_path.is_file():
        return False

    try:
        return db_fs_path.stat().st_size > 0
    except Exception:
        return False


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_mime(path: Path) -> str:
    try:
        from lib import mime_sqlite as mimemod

        mime = mimemod.detect_mime(path)
        if mime:
            return str(mime).strip()
    except Exception:
        pass

    mime, _enc = mimetypes.guess_type(str(path), strict=False)
    return (mime or "(unknown)").strip()


def build_filename_index(root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        index.setdefault(path.name, []).append(path)
    return index


def find_same_name_same_sha1(
    src_file: Path,
    src_sha1: str,
    filename_index: Dict[str, List[Path]],
    sha1_cache: Dict[Path, str],
) -> Optional[Path]:
    candidates = filename_index.get(src_file.name, [])
    if not candidates:
        return None

    src_resolved = src_file.resolve()
    for candidate in candidates:
        try:
            if candidate.resolve() == src_resolved:
                continue
        except Exception:
            pass

        try:
            candidate_sha1 = sha1_cache.get(candidate)
            if candidate_sha1 is None:
                candidate_sha1 = sha1_file(candidate)
                sha1_cache[candidate] = candidate_sha1
            if candidate_sha1 == src_sha1:
                return candidate
        except Exception:
            continue

    return None


def copy_if_needed(src_path: Path, src_root: Path, dst_root: Path, overwrite: bool, dry_run: bool) -> Tuple[str, Path]:
    rel_path = src_path.relative_to(src_root)
    dst_path = dst_root / rel_path

    if dst_path.exists() and not overwrite:
        return "exists", dst_path

    if not dry_run:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    return "copied", dst_path


def main() -> int:
    load_dotenv()
    args = parse_args()

    src_root = Path(args.src_dir).expanduser().resolve()
    dst_root = Path(args.dst_dir).expanduser().resolve()
    photos_dir_env = os.environ.get("PHOTOAI_PHOTOS_DIR", "").strip()
    photos_root = Path(photos_dir_env).expanduser().resolve() if photos_dir_env else None

    if not src_root.exists() or not src_root.is_dir():
        print(f"ERROR: invalid source directory: {src_root}", file=sys.stderr)
        return 2

    if src_root == dst_root:
        print("ERROR: source and destination directories must be different", file=sys.stderr)
        return 2

    filename_index: Dict[str, List[Path]] = {}
    sha1_cache: Dict[Path, str] = {}
    if photos_root is not None and photos_root.exists() and photos_root.is_dir():
        filename_index = build_filename_index(photos_root)

    try:
        con = connect_db()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        sha1_to_db_path = load_db_sha1_to_path(con)
    finally:
        con.close()

    known_sha1s: Set[str] = set(sha1_to_db_path.keys())

    files = list(iter_files(src_root))
    files.sort()
    if args.limit > 0:
        files = files[: args.limit]

    scanned = 0
    skipped_known = 0
    db_known_but_missing_or_empty = 0
    unsupported_seen = 0
    skipped_unsupported_same_name_sha1 = 0
    copied = 0
    skipped_existing_dest = 0
    errors = 0

    print(f"Source      : {src_root}")
    print(f"Destination : {dst_root}")
    print(f"DB sha1     : {len(known_sha1s)}")
    print(f"Files found : {len(files)}")
    if photos_root is not None:
        print(f"PHOTOAI_PHOTOS_DIR: {photos_root}")
        print(f"Indexed names     : {len(filename_index)}")
    if args.dry_run:
        print("Mode        : dry-run")

    for file_path in files:
        scanned += 1
        if scanned == 1 or scanned % 250 == 0 or scanned == len(files):
            print(f"[scan] {scanned}/{len(files)}")

        rel_path = file_path.relative_to(src_root)
        try:
            mime = detect_mime(file_path)
            file_sha1 = sha1_file(file_path)
            if file_sha1 in known_sha1s:
                if db_file_exists_and_nonempty(file_sha1, sha1_to_db_path, photos_root):
                    skipped_known += 1
                    continue
                db_known_but_missing_or_empty += 1
                db_rel = sha1_to_db_path.get(file_sha1, "(path_db_mancante)")
                print(f"[DB-MISSING-OR-EMPTY] {rel_path} -> {db_rel}")

            if mime not in SUPPORTED_MIME_TYPES:
                unsupported_seen += 1
                if filename_index:
                    matched_path = find_same_name_same_sha1(
                        src_file=file_path,
                        src_sha1=file_sha1,
                        filename_index=filename_index,
                        sha1_cache=sha1_cache,
                    )
                    if matched_path is not None:
                        skipped_unsupported_same_name_sha1 += 1
                        print(f"[SKIP-UNSUPPORTED-SAME-SHA1] {rel_path} -> {matched_path}")
                        continue

            result, dst_path = copy_if_needed(
                src_path=file_path,
                src_root=src_root,
                dst_root=dst_root,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if result == "exists":
                skipped_existing_dest += 1
                print(f"[SKIP-DEST] {rel_path} -> {dst_path}")
                continue

            copied += 1
            tag = "DRY-COPY" if args.dry_run else "COPY"
            print(f"[{tag}] {rel_path} -> {dst_path}")
        except Exception as exc:
            errors += 1
            print(f"[ERROR] {rel_path} -> {exc}", file=sys.stderr)

    print("\nSummary:")
    print(f"  scanned              : {scanned}")
    print(f"  skipped_known_sha1   : {skipped_known}")
    print(f"  db_known_but_missing_or_empty : {db_known_but_missing_or_empty}")
    print(f"  unsupported_seen     : {unsupported_seen}")
    print(f"  skipped_unsupported_same_name_sha1 : {skipped_unsupported_same_name_sha1}")
    print(f"  copied               : {copied}")
    print(f"  skipped_dest_exists  : {skipped_existing_dest}")
    print(f"  errors               : {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
