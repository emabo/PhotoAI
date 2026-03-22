#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv


class Candidate:
    def __init__(self, sha1: str, db_relpath: str, dup_relpaths: List[str]) -> None:
        self.sha1 = sha1
        self.db_relpath = db_relpath
        self.dup_relpaths = dup_relpaths


def must_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set (define it in .env or environment variables).")
    return value


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def connect_db() -> Tuple[sqlite3.Connection, Path]:
    sqlite_dir = Path(must_env("PHOTOAI_SQLITE_DIR")).expanduser()
    db_path = sqlite_dir / "photo_ai.sqlite"
    if not db_path.exists():
        raise RuntimeError(f"DB not found: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys=ON;")
    con.row_factory = sqlite3.Row
    return con, db_path


def normalize_relpath(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def update_image_technical_fields(con: sqlite3.Connection, sha1: str, abs_path: Path) -> None:
    st = abs_path.stat()
    con.execute(
        "UPDATE images SET mtime = ?, file_size = ? WHERE sha1 = ?",
        (float(st.st_mtime), int(st.st_size), sha1),
    )


def update_related_path_fields(con: sqlite3.Connection, sha1: str, new_relpath: str) -> None:
    con.execute("UPDATE images SET path = ? WHERE sha1 = ?", (new_relpath, sha1))

    row = con.execute("SELECT params_json FROM captions WHERE sha1 = ?", (sha1,)).fetchone()
    if row is None:
        return

    params_json = row["params_json"]
    if not params_json:
        return

    try:
        params = json.loads(params_json)
    except Exception:
        return

    if not isinstance(params, dict):
        return

    params["relpath"] = new_relpath
    con.execute(
        "UPDATE captions SET params_json = ? WHERE sha1 = ?",
        (json.dumps(params, ensure_ascii=False, separators=(",", ":")), sha1),
    )


def update_chroma_path_fields(sha1: str, new_relpath: str) -> Optional[str]:
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
            col = client.get_collection(collection_name)
            got = col.get(ids=[sha1], include=["metadatas"])
            ids = got.get("ids", []) or []
            metadatas = got.get("metadatas", []) or []
            if not ids:
                continue
            meta = dict(metadatas[0] or {})
            meta["path"] = new_relpath
            if "relpath" in meta:
                meta["relpath"] = new_relpath
            col.update(ids=[sha1], metadatas=[meta])
        return None
    except Exception as exc:
        return str(exc)


def collect_candidates(con: sqlite3.Connection, photos_dir: Path) -> List[Candidate]:
    image_rows = con.execute("SELECT sha1, path FROM images WHERE sha1 IS NOT NULL AND trim(sha1) <> ''").fetchall()

    db_by_path: Dict[str, sqlite3.Row] = {}
    db_by_sha1: Dict[str, sqlite3.Row] = {}
    for row in image_rows:
        relpath = normalize_relpath(str(row["path"] or ""))
        sha1 = str(row["sha1"] or "").strip()
        if relpath:
            db_by_path[relpath] = row
        if sha1 and sha1 not in db_by_sha1:
            db_by_sha1[sha1] = row

    fs_relpaths: List[str] = []
    for path in photos_dir.rglob("*"):
        if path.is_file():
            fs_relpaths.append(normalize_relpath(str(path.relative_to(photos_dir))))

    unmapped_relpaths = sorted(set(fs_relpaths) - set(db_by_path.keys()))

    dup_by_sha1: Dict[str, List[str]] = {}
    for relpath in unmapped_relpaths:
        abs_path = photos_dir / relpath
        try:
            sha1 = sha1_file(abs_path)
        except Exception as exc:
            print(f"WARN: sha1 non calcolabile per {relpath}: {exc}", file=sys.stderr)
            continue

        db_row = db_by_sha1.get(sha1)
        if db_row is None:
            continue

        dup_by_sha1.setdefault(sha1, []).append(relpath)

    out: List[Candidate] = []
    for sha1 in sorted(dup_by_sha1.keys()):
        db_row = db_by_sha1[sha1]
        db_relpath = normalize_relpath(str(db_row["path"] or ""))
        dup_relpaths = sorted(dup_by_sha1[sha1], key=lambda p: (Path(p).parent, Path(p).name))
        if db_relpath and dup_relpaths:
            out.append(Candidate(sha1=sha1, db_relpath=db_relpath, dup_relpaths=dup_relpaths))
    out.sort(key=lambda c: (Path(c.db_relpath).parent.as_posix(), Path(c.db_relpath).name, c.sha1))
    return out


def print_grouped_dup_relpaths(dup_relpaths: List[str]) -> None:
    by_dir: defaultdict = defaultdict(list)
    idx_map: Dict[str, int] = {}
    idx = 1
    for relpath in dup_relpaths:
        parent = Path(relpath).parent
        by_dir[parent].append(relpath)
        idx_map[relpath] = idx
        idx += 1

    for dir_key in sorted(by_dir.keys()):
        print(f"\n  {dir_key}/")
        for relpath in by_dir[dir_key]:
            print(f"   {idx_map[relpath]}) {Path(relpath).name}")


def prompt_choice(candidate: Candidate, photos_dir: Path) -> Optional[str]:
    db_abs = photos_dir / candidate.db_relpath
    db_exists = db_abs.exists()
    print("\n" + "=" * 90)
    print(f"SHA1: {candidate.sha1}")
    print(f" k) DB : {candidate.db_relpath} {'[OK]' if db_exists else '[MANCANTE]'}")
    
    print_grouped_dup_relpaths(candidate.dup_relpaths)

    print("\nScelte:")
    print("  k   = tieni il file gia presente al path DB e rimuovi solo i duplicati non mappati")
    print("  d   = tieni la entry DB ma ricollegala a uno dei file FS sopra; la scelta del file avviene dopo")
    print("  s = salta")
    print("  q = esci")

    while True:
        answer = input("Selezione: ").strip().lower()
        if answer in {"k", "d", "s", "q"}:
            return answer
        print("Valore non valido.")


def remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def resolve_keep_existing_db_file(candidate: Candidate, photos_dir: Path) -> Tuple[int, int]:
    db_abs = photos_dir / candidate.db_relpath
    if not db_abs.exists():
        raise RuntimeError(f"file DB non trovato: {candidate.db_relpath}")

    removed = 0
    errors = 0
    for relpath in candidate.dup_relpaths:
        abs_path = photos_dir / relpath
        try:
            remove_file(abs_path)
            removed += 1
        except Exception as exc:
            errors += 1
            print(f"WARN: impossibile rimuovere {relpath}: {exc}", file=sys.stderr)
    return removed, errors


def are_files_in_same_dir(candidate: Candidate) -> bool:
    """Check if all duplicate unmapped files are in the same directory as the mapped DB file."""
    db_parent = Path(candidate.db_relpath).parent
    for dup_relpath in candidate.dup_relpaths:
        dup_parent = Path(dup_relpath).parent
        if dup_parent != db_parent:
            return False
    return True


def choose_existing_relpath(candidate: Candidate, photos_dir: Path) -> str:
    existing = [relpath for relpath in candidate.dup_relpaths if (photos_dir / relpath).exists()]
    if not existing:
        raise RuntimeError("nessun file alternativo esistente trovato sul filesystem")
    if len(existing) == 1:
        return existing[0]

    print("\nSono presenti piu file candidati. Scegli quale path associare alla entry DB:")
    for idx, relpath in enumerate(existing, start=1):
        print(f"  {idx}) {relpath}")

    while True:
        answer = input("Path da associare al DB: ").strip()
        if answer.isdigit():
            idx = int(answer)
            if 1 <= idx <= len(existing):
                return existing[idx - 1]
        print("Valore non valido.")


def resolve_keep_db(
    con: sqlite3.Connection,
    candidate: Candidate,
    selected_relpath: str,
    photos_dir: Path,
) -> Tuple[int, int, Optional[str]]:
    db_abs = photos_dir / candidate.db_relpath
    selected_abs = photos_dir / selected_relpath
    if not selected_abs.exists():
        raise RuntimeError(f"file selezionato non trovato: {selected_relpath}")

    removed = 0
    errors = 0

    if candidate.db_relpath != selected_relpath and db_abs.exists():
        try:
            remove_file(db_abs)
            removed += 1
        except Exception as exc:
            errors += 1
            print(f"WARN: impossibile rimuovere {candidate.db_relpath}: {exc}", file=sys.stderr)

    for relpath in candidate.dup_relpaths:
        if relpath == selected_relpath:
            continue
        abs_path = photos_dir / relpath
        try:
            remove_file(abs_path)
            removed += 1
        except Exception as exc:
            errors += 1
            print(f"WARN: impossibile rimuovere {relpath}: {exc}", file=sys.stderr)

    update_related_path_fields(con, candidate.sha1, selected_relpath)
    update_image_technical_fields(con, candidate.sha1, selected_abs)
    chroma_error = update_chroma_path_fields(candidate.sha1, selected_relpath)
    return removed, errors, chroma_error


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trova file con stesso SHA1 di una entry già presente nel DB e chiede quale tenere."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        help="Directory foto. Se omessa usa PHOTOAI_PHOTOS_DIR.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Numero massimo di gruppi da processare (0 = nessun limite).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    photos_dir_raw = (args.root or os.environ.get("PHOTOAI_PHOTOS_DIR") or "").strip()
    if not photos_dir_raw:
        print("ERRORE: specifica una directory oppure imposta PHOTOAI_PHOTOS_DIR.", file=sys.stderr)
        return 2

    photos_dir = Path(photos_dir_raw).expanduser().resolve()
    if not photos_dir.exists() or not photos_dir.is_dir():
        print(f"ERRORE: directory non valida: {photos_dir}", file=sys.stderr)
        return 2

    try:
        con, db_path = connect_db()
    except Exception as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 2

    try:
        print(f"DB: {db_path}")
        print(f"PHOTOS_DIR: {photos_dir}")

        candidates = collect_candidates(con, photos_dir)
        if args.limit > 0:
            candidates = candidates[: args.limit]

        print(f"Gruppi trovati: {len(candidates)}")
        if not candidates:
            return 0

        groups_done = 0
        removed_files = 0
        errors = 0
        skipped = 0

        for candidate in candidates:
            db_abs = photos_dir / candidate.db_relpath
            db_exists = db_abs.exists()

            if are_files_in_same_dir(candidate) and db_exists:
                choice = "k"
                print("\n" + "=" * 90)
                print(f"SHA1: {candidate.sha1}")
                print(f" k) DB : {candidate.db_relpath} [OK]")
                print_grouped_dup_relpaths(candidate.dup_relpaths)
                print("\n[AUTO] Tutti i duplicati nella stessa dir del file DB: scelgo automaticamente 'k'")
            else:
                choice = prompt_choice(candidate, photos_dir)

            if choice == "q":
                break
            if choice == "s" or choice is None:
                skipped += 1
                continue

            try:
                if choice == "k":
                    removed, op_errors = resolve_keep_existing_db_file(candidate, photos_dir)
                    con.commit()
                    print(
                        "OK: tenuto il file gia presente nel DB, path invariato e rimossi "
                        f"{removed} file duplicati"
                    )
                elif choice == "d":
                    selected_relpath = choose_existing_relpath(candidate, photos_dir)
                    removed, op_errors, chroma_error = resolve_keep_db(
                        con,
                        candidate,
                        selected_relpath,
                        photos_dir,
                    )
                    con.commit()
                    print(
                        "OK: tenuta entry DB, aggiornato path DB a "
                        f"{selected_relpath} e rimossi {removed} file"
                    )
                    if chroma_error:
                        print(f"WARN: aggiornamento path su Chroma fallito: {chroma_error}", file=sys.stderr)
                removed_files += removed
                errors += op_errors
                groups_done += 1
            except Exception as exc:
                con.rollback()
                errors += 1
                print(f"ERRORE: {exc}", file=sys.stderr)

        print("\nRiepilogo:")
        print(f"  gruppi_processati={groups_done}")
        print(f"  gruppi_saltati={skipped}")
        print(f"  file_rimossi={removed_files}")
        print(f"  errori={errors}")
        return 0 if errors == 0 else 1
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
