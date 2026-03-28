#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv


@dataclass
class ErrorEntry:
    step: str
    detail: str
    updated_at: float


@dataclass
class ErrorCandidate:
    sha1: str
    relpath: str
    mime: str
    file_exists: bool
    thumb_exists: bool
    errors: List[ErrorEntry]


def must_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set (define it in .env or environment variables).")
    return value


def connect_db() -> Tuple[sqlite3.Connection, Path]:
    sqlite_dir = Path(must_env("PHOTOAI_SQLITE_DIR")).expanduser()
    db_path = sqlite_dir / "photo_ai.sqlite"
    if not db_path.exists():
        raise RuntimeError(f"DB not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys=ON;")
    con.row_factory = sqlite3.Row
    return con, db_path


def format_ts(value: float) -> str:
    try:
        from datetime import datetime

        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def delete_from_chroma(sha1: str) -> Optional[str]:
    chroma_dir_raw = os.environ.get("PHOTOAI_CHROMA_DIR", "").strip()
    if not chroma_dir_raw:
        return None

    try:
        import chromadb
        from chromadb.config import Settings

        image_collection = os.environ.get("PHOTOAI_COLLECTION", "images_openclip_vitl14_336").strip()
        captions_collection = os.environ.get("PHOTOAI_CAPTIONS_COLLECTION", "captions_openclip_vitl14_336").strip()

        client = chromadb.PersistentClient(
            path=str(Path(chroma_dir_raw).expanduser()),
            settings=Settings(anonymized_telemetry=False),
        )

        for collection_name in (image_collection, captions_collection):
            if not collection_name:
                continue
            try:
                client.get_collection(collection_name).delete(ids=[sha1])
            except Exception:
                # Missing collection or missing id should not block the cleanup.
                continue
        return None
    except Exception as exc:
        return str(exc)


def collect_error_candidates(
    con: sqlite3.Connection,
    photos_dir: Path,
    thumb_dir: Path,
    limit: int,
) -> List[ErrorCandidate]:
    rows = con.execute(
        """
        SELECT
          j.sha1,
          i.path,
          i.mime,
          j.step,
          j.detail,
          j.updated_at
        FROM jobs j
        LEFT JOIN images i ON i.sha1 = j.sha1
        WHERE j.status = 'error'
        ORDER BY j.updated_at DESC, j.sha1 ASC, j.step ASC
        """
    ).fetchall()

    grouped: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        sha1 = str(row["sha1"] or "").strip()
        if not sha1:
            continue
        grouped.setdefault(sha1, []).append(row)

    out: List[ErrorCandidate] = []
    for sha1, group_rows in grouped.items():
        first = group_rows[0]
        relpath = str(first["path"] or "").strip()
        mime = str(first["mime"] or "").strip()
        abs_path = (photos_dir / relpath).resolve() if relpath else None
        thumb_path = thumb_dir / f"{sha1}.jpg"
        out.append(
            ErrorCandidate(
                sha1=sha1,
                relpath=relpath,
                mime=mime,
                file_exists=bool(abs_path and abs_path.exists()),
                thumb_exists=thumb_path.exists(),
                errors=[
                    ErrorEntry(
                        step=str(r["step"] or ""),
                        detail=str(r["detail"] or "")[:500],
                        updated_at=float(r["updated_at"] or 0),
                    )
                    for r in group_rows
                ],
            )
        )

    out.sort(key=lambda c: max((e.updated_at for e in c.errors), default=0), reverse=True)
    if limit > 0:
        out = out[:limit]
    return out


def remove_candidate(
    con: sqlite3.Connection,
    candidate: ErrorCandidate,
    photos_dir: Path,
    thumb_dir: Path,
    dry_run: bool,
) -> Tuple[List[str], Optional[str]]:
    actions: List[str] = []
    relpath = candidate.relpath
    abs_path = (photos_dir / relpath).resolve() if relpath else None
    thumb_path = thumb_dir / f"{candidate.sha1}.jpg"

    if abs_path and abs_path.exists():
        actions.append(f"fs:{abs_path}")
        if not dry_run:
            abs_path.unlink()
    else:
        actions.append("fs:(missing)")

    if thumb_path.exists():
        actions.append(f"thumb:{thumb_path}")
        if not dry_run:
            thumb_path.unlink()
    else:
        actions.append("thumb:(missing)")

    actions.append("sqlite:captions_fts")
    actions.append("sqlite:captions")
    actions.append("sqlite:tags")
    actions.append("sqlite:jobs")
    actions.append("sqlite:images")
    if not dry_run:
        con.execute("DELETE FROM captions_fts WHERE sha1 = ?", (candidate.sha1,))
        con.execute("DELETE FROM captions WHERE sha1 = ?", (candidate.sha1,))
        con.execute("DELETE FROM tags WHERE sha1 = ?", (candidate.sha1,))
        con.execute("DELETE FROM jobs WHERE sha1 = ?", (candidate.sha1,))
        con.execute("DELETE FROM images WHERE sha1 = ?", (candidate.sha1,))
        con.commit()

    chroma_error: Optional[str] = None
    actions.append("chroma:image+captions")
    if not dry_run:
        chroma_error = delete_from_chroma(candidate.sha1)

    return actions, chroma_error


def print_candidate(index: int, total: int, candidate: ErrorCandidate) -> None:
    print("\n" + "=" * 80)
    print(f"[{index}/{total}] {candidate.relpath or '(no image path)'}")
    print("=" * 80)
    print(f"sha1       : {candidate.sha1}")
    print(f"mime       : {candidate.mime or '(null)'}")
    print(f"file_exists: {candidate.file_exists}")
    print(f"thumb_exists: {candidate.thumb_exists}")
    print("errori:")
    for err in candidate.errors:
        print(f"  - {format_ts(err.updated_at)} | step={err.step} | detail={err.detail}")


def ask_action() -> str:
    while True:
        answer = input("Rimuovere da DB/fs/chroma/thumb? [y]es / [n]o / [a]ll / [q]uit: ").strip().lower()
        if answer in {"y", "n", "a", "q"}:
            return answer
        print("Risposta non valida. Usa y, n, a oppure q.")


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Per ogni file con job in errore, chiede se rimuoverlo da filesystem, SQLite, Chroma e thumbs.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limita il numero di file in errore esaminati (0 = tutti)")
    ap.add_argument("--dry-run", action="store_true", help="Mostra le azioni senza eseguirle")
    args = ap.parse_args()

    photos_dir = Path(must_env("PHOTOAI_PHOTOS_DIR")).expanduser().resolve()
    thumb_dir = Path(os.environ.get("PHOTOAI_THUMB_DIR", "cache/thumbs")).expanduser().resolve()

    if not photos_dir.exists() or not photos_dir.is_dir():
        raise RuntimeError(f"PHOTOAI_PHOTOS_DIR is not a valid directory: {photos_dir}")

    con, db_path = connect_db()
    removed = 0
    skipped = 0
    chroma_warnings = 0

    try:
        candidates = collect_error_candidates(con, photos_dir, thumb_dir, args.limit)
        if not candidates:
            print("Nessun file con stato error trovato.")
            return 0

        print(f"DB        : {db_path}")
        print(f"PHOTOS_DIR: {photos_dir}")
        print(f"THUMB_DIR : {thumb_dir}")
        print(f"Candidati : {len(candidates)}")
        if args.dry_run:
            print("Modo      : dry-run")

        yes_to_all = False
        for idx, candidate in enumerate(candidates, start=1):
            print_candidate(idx, len(candidates), candidate)
            if yes_to_all:
                answer = "y"
            else:
                answer = ask_action()

            if answer == "q":
                break
            if answer == "n":
                skipped += 1
                continue
            if answer == "a":
                yes_to_all = True
                answer = "y"

            actions, chroma_error = remove_candidate(
                con=con,
                candidate=candidate,
                photos_dir=photos_dir,
                thumb_dir=thumb_dir,
                dry_run=args.dry_run,
            )
            removed += 1
            print("Azioni:")
            for action in actions:
                print(f"  - {action}")
            if chroma_error:
                chroma_warnings += 1
                print(f"WARN Chroma: {chroma_error}", file=sys.stderr)

        print("\nSummary")
        print(f"  removed={removed}")
        print(f"  skipped={skipped}")
        print(f"  chroma_warnings={chroma_warnings}")
        if args.dry_run:
            print("  mode=dry-run")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
