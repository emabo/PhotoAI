#!/usr/bin/env python3
import os
import sys
import sqlite3
import argparse
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from lib.media_types import is_supported_mime


def must_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set (define it in .env or environment variables).")
    return v


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def section(title: str) -> None:
    print("\n\n" + "-" * 80)
    print(title)
    print("-" * 80 + "\n")


def fetch_all(con: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
    cur = con.execute(sql, params)
    return cur.fetchall()


def fetch_one(con: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Tuple[Any, ...]]:
    cur = con.execute(sql, params)
    return cur.fetchone()


def _format_timestamp(field_name: str, value: Any) -> str:
    if value is None:
        return ""

    name = (field_name or "").lower()
    is_time_field = (
        name.endswith("_at")
        or name in {"mtime", "updated", "created", "taken"}
    ) and not name.endswith("_utc")

    if not is_time_field:
        return str(value)

    try:
        ts = float(value)
    except (TypeError, ValueError):
        return str(value)

    try:
        dt = datetime.fromtimestamp(ts)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (OverflowError, OSError, ValueError):
        return str(value)


def print_rows(rows: List[Tuple[Any, ...]], max_rows: int = 30, headers: Optional[List[str]] = None) -> None:
    if not rows:
        print("  (no rows)")
        return

    shown_rows = rows[:max_rows]
    sample = shown_rows[0]
    n_cols = len(sample)

    if headers is None:
        headers = [f"col{i+1}" for i in range(n_cols)]
    else:
        headers = list(headers)
        if len(headers) < n_cols:
            headers.extend(f"col{i+1}" for i in range(len(headers), n_cols))
        elif len(headers) > n_cols:
            headers = headers[:n_cols]

    string_rows = [
        [_format_timestamp(headers[i], r[i]) if i < len(r) else "" for i in range(n_cols)]
        for r in shown_rows
    ]
    widths = [len(headers[i]) for i in range(n_cols)]
    for row in string_rows:
        for i, value in enumerate(row):
            if len(value) > widths[i]:
                widths[i] = len(value)

    def fmt(values: List[str]) -> str:
        return " | ".join(values[i].ljust(widths[i]) for i in range(n_cols))

    separator = "-+-".join("-" * w for w in widths)
    print("  " + fmt(headers))
    print("  " + separator)
    for row in string_rows:
        print("  " + fmt(row))

    if len(rows) > max_rows:
        print(f"  ... ({len(rows) - max_rows} more rows)")


def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = fetch_one(con, "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=? LIMIT 1", (name,))
    return row is not None


def _chunked(items: List[str], size: int = 500) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _count_distinct_sha1_in(con: sqlite3.Connection, table_name: str, sha1s: List[str]) -> int:
    if not sha1s:
        return 0

    total = 0
    for chunk in _chunked(sha1s):
        placeholders = ",".join("?" for _ in chunk)
        row = fetch_one(
            con,
            f"SELECT COUNT(DISTINCT sha1) AS n FROM {table_name} WHERE sha1 IN ({placeholders})",
            tuple(chunk),
        )
        if row is not None:
            total += int(row["n"])
    return total


def _count_chroma_ids_for_sha1s(sha1s: List[str]) -> Optional[int]:
    chroma_dir_raw = os.environ.get("PHOTOAI_CHROMA_DIR", "").strip()
    if not chroma_dir_raw or not sha1s:
        return 0

    try:
        import chromadb
        from chromadb.config import Settings
    except Exception:
        return None

    image_collection = os.environ.get("PHOTOAI_COLLECTION", "images_openclip_vitl14_336")
    try:
        client = chromadb.PersistentClient(
            path=str(Path(chroma_dir_raw).expanduser()),
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection(image_collection)
    except Exception:
        return None

    found: Set[str] = set()
    for chunk in _chunked(sha1s):
        got = col.get(ids=chunk, include=[])
        found.update(got.get("ids", []))
    return len(found)


def _sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _mime_for_unmapped_file(path: Path) -> str:
    try:
        from lib import mime_sqlite as mimemod

        mime = mimemod.detect_mime(path)
        if mime:
            return mime
    except Exception:
        pass

    mime, _enc = mimetypes.guess_type(str(path), strict=False)
    return mime or "(unknown)"


def _extension_for_report(path: Path) -> str:
    suffix = path.suffix.lower().strip()
    return suffix if suffix else "(no_ext)"


def run_filesystem_mapping_prospect(con: sqlite3.Connection) -> None:
    section("Prospetto filesystem / SQLite / Chroma")

    photos_dir_raw = os.environ.get("PHOTOAI_PHOTOS_DIR", "").strip()
    if not photos_dir_raw:
        print("  WARN: PHOTOAI_PHOTOS_DIR non impostata, prospetto saltato")
        return

    photos_dir = Path(photos_dir_raw).expanduser()
    if not photos_dir.exists() or not photos_dir.is_dir():
        print(f"  WARN: PHOTOAI_PHOTOS_DIR non valida: {photos_dir}")
        return

    print(f"  PHOTOAI_PHOTOS_DIR: {photos_dir.resolve()}")

    fs_relpaths: Set[str] = set()
    for file_path in photos_dir.rglob("*"):
        if file_path.is_file():
            rel = file_path.relative_to(photos_dir)
            fs_relpaths.add(str(rel).replace("\\", "/"))

    image_rows = fetch_all(
        con,
        """
        SELECT path, sha1, mime, gps_lat, gps_lon
        FROM images
        """,
    )

    db_by_path: Dict[str, sqlite3.Row] = {}
    db_by_sha1: Dict[str, sqlite3.Row] = {}
    for row in image_rows:
        path = str(row["path"] or "").replace("\\", "/").strip("/")
        sha1 = str(row["sha1"] or "").strip()
        if path:
            db_by_path[path] = row
        if sha1 and sha1 not in db_by_sha1:
            db_by_sha1[sha1] = row

    db_paths = set(db_by_path.keys())
    mapped_paths = fs_relpaths & db_paths
    unmapped_paths = fs_relpaths - db_paths

    mapped_rows = [db_by_path[p] for p in mapped_paths]
    mapped_sha1s = [str(r["sha1"]) for r in mapped_rows if r["sha1"]]

    captions_count = _count_distinct_sha1_in(con, "captions", mapped_sha1s)
    tags_count = _count_distinct_sha1_in(con, "tags", mapped_sha1s)
    georef_count = sum(1 for r in mapped_rows if r["gps_lat"] is not None and r["gps_lon"] is not None)
    chroma_count = _count_chroma_ids_for_sha1s(mapped_sha1s)

    summary_rows = [
        ("file_in_photos_dir", len(fs_relpaths)),
        ("mappati_su_sqlite", len(mapped_paths)),
        ("presenti_in_chroma", chroma_count if chroma_count is not None else "N/A"),
        ("con_caption", captions_count),
        ("con_tags", tags_count),
        ("con_georef(gps)", georef_count),
    ]
    print_rows(summary_rows, max_rows=20, headers=["metrica", "conteggio"])

    section("Estensioni filesystem non accettate o non riconosciute")

    unsupported_ext_counts: Dict[str, int] = {}
    unknown_ext_counts: Dict[str, int] = {}
    unsupported_examples: List[Tuple[Any, ...]] = []
    unknown_examples: List[Tuple[Any, ...]] = []

    for rel in sorted(fs_relpaths):
        abs_path = photos_dir / rel
        mime = _mime_for_unmapped_file(abs_path)
        ext = _extension_for_report(abs_path)

        if mime == "(unknown)":
            unknown_ext_counts[ext] = unknown_ext_counts.get(ext, 0) + 1
            if len(unknown_examples) < 30:
                unknown_examples.append((ext, rel))
            continue

        if not is_supported_mime(mime):
            unsupported_ext_counts[ext] = unsupported_ext_counts.get(ext, 0) + 1
            if len(unsupported_examples) < 30:
                unsupported_examples.append((ext, mime, rel))

    if unsupported_ext_counts:
        print_rows(
            sorted(((ext, count) for ext, count in unsupported_ext_counts.items()), key=lambda x: (-x[1], x[0])),
            max_rows=max(30, len(unsupported_ext_counts)),
            headers=["ext_non_accettata", "count"],
        )
    else:
        print("  Nessuna estensione non accettata trovata.")

    if unknown_ext_counts:
        print_rows(
            sorted(((ext, count) for ext, count in unknown_ext_counts.items()), key=lambda x: (-x[1], x[0])),
            max_rows=max(30, len(unknown_ext_counts)),
            headers=["ext_non_riconosciuta", "count"],
        )
    else:
        print("  Nessuna estensione non riconosciuta trovata.")

    if unsupported_examples:
        section("Esempi: estensioni non accettate")
        print_rows(
            unsupported_examples,
            max_rows=30,
            headers=["ext", "mime", "path"],
        )

    if unknown_examples:
        section("Esempi: estensioni non riconosciute")
        print_rows(
            unknown_examples,
            max_rows=30,
            headers=["ext", "path"],
        )

    section("Prospetto MIME: DB vs file non ancora mappati")

    db_mime_counts: Dict[str, int] = {}
    for row in mapped_rows:
        mime = (row["mime"] or "").strip() if row["mime"] is not None else ""
        key = mime if mime else "(null)"
        db_mime_counts[key] = db_mime_counts.get(key, 0) + 1

    unmapped_mime_counts: Dict[str, int] = {}
    supported_unmapped_count = 0
    unsupported_unmapped_count = 0
    same_sha1_other_path_count = 0
    supported_without_sha1_match_count = 0
    same_sha1_other_path_rows: List[Tuple[Any, ...]] = []
    supported_without_sha1_match_rows: List[Tuple[Any, ...]] = []

    for rel in sorted(unmapped_paths):
        abs_path = photos_dir / rel
        mime = _mime_for_unmapped_file(abs_path)
        unmapped_mime_counts[mime] = unmapped_mime_counts.get(mime, 0) + 1

        if not is_supported_mime(mime):
            unsupported_unmapped_count += 1
            continue

        supported_unmapped_count += 1

        try:
            sha1 = _sha1_file(abs_path)
        except Exception:
            sha1 = ""

        db_row = db_by_sha1.get(sha1) if sha1 else None
        if db_row is not None:
            same_sha1_other_path_count += 1
            if len(same_sha1_other_path_rows) < 30:
                same_sha1_other_path_rows.append(
                    (
                        rel,
                        mime,
                        str(db_row["path"] or ""),
                        sha1,
                    )
                )
        else:
            supported_without_sha1_match_count += 1
            if len(supported_without_sha1_match_rows) < 30:
                supported_without_sha1_match_rows.append(
                    (
                        rel,
                        mime,
                        "da_ingestire_o_job_fallito",
                    )
                )

    mime_keys = sorted(set(db_mime_counts.keys()) | set(unmapped_mime_counts.keys()))
    mime_rows = [
        (mime, db_mime_counts.get(mime, 0), unmapped_mime_counts.get(mime, 0))
        for mime in mime_keys
    ]

    print_rows(
        mime_rows,
        max_rows=max(50, len(mime_rows)),
        headers=["mime", "db_count(mapped)", "non_mappati_count"],
    )

    print(
        "  "
        f"totale_mapped={len(mapped_paths)} "
        f"totale_non_mappati={len(unmapped_paths)} "
        f"totale_db_images={len(db_paths)}"
    )

    section("Diagnostica non mappati")
    print_rows(
        [
            ("non_mappati_supportati", supported_unmapped_count),
            ("non_mappati_non_supportati", unsupported_unmapped_count),
            ("non_mappati_stesso_sha1_altro_path", same_sha1_other_path_count),
            ("non_mappati_supportati_senza_match_sha1", supported_without_sha1_match_count),
        ],
        max_rows=20,
        headers=["metrica", "conteggio"],
    )

    if same_sha1_other_path_rows:
        section("Esempi: file non mappati con stesso sha1 ma path diverso")
        print_rows(
            same_sha1_other_path_rows,
            max_rows=30,
            headers=["path_fs", "mime", "path_db", "sha1"],
        )
        
        # Group by sha1 to show duplicates/renames
        sha1_groups = {}
        for rel_path, mime, db_path, sha1 in same_sha1_other_path_rows:
            if sha1 not in sha1_groups:
                sha1_groups[sha1] = {"fs": [], "db": []}
            sha1_groups[sha1]["fs"].append(rel_path)
            sha1_groups[sha1]["db"].append(db_path)
        
        if sha1_groups:
            section("Raggruppamento per SHA1 (file con stesso contenuto)")
            for sha1, paths in sorted(sha1_groups.items()):
                print(f"  SHA1: {sha1}")
                for fs_p in paths["fs"]:
                    print(f"    FS:  {fs_p}")
                for db_p in paths["db"]:
                    print(f"    DB:  {db_p}")

    if supported_without_sha1_match_rows:
        section("Esempi: file supportati non mappati senza match sha1")
        print_rows(
            supported_without_sha1_match_rows,
            max_rows=30,
            headers=["path_fs", "mime", "stato"],
        )


def _detect_target(path_arg: str, photos_dir: Optional[Path]) -> Tuple[str, bool, bool]:
    raw = path_arg.strip()
    p = Path(raw).expanduser()
    if p.exists():
        resolved = p.resolve()
        is_dir = p.is_dir()
        if photos_dir is not None:
            try:
                rel = resolved.relative_to(photos_dir)
                return str(rel).replace("\\", "/"), is_dir, True
            except ValueError:
                pass
        return str(resolved).replace("\\", "/"), is_dir, True

    normalized = raw.replace("\\", "/").strip()
    is_dir = normalized.endswith("/") or Path(normalized).suffix == ""
    return normalized.strip("/"), is_dir, False


def _fetch_image_rows_for_target(con: sqlite3.Connection, target_key: str, is_dir: bool) -> List[sqlite3.Row]:
    if is_dir:
        prefix = target_key.strip("/")
        if not prefix:
            return fetch_all(con, "SELECT * FROM images ORDER BY path")
        return fetch_all(
            con,
            """
            SELECT *
            FROM images
            WHERE path = ? OR path LIKE ?
            ORDER BY path
            """,
            (prefix, f"{prefix}/%"),
        )

    return fetch_all(
        con,
        """
        SELECT *
        FROM images
        WHERE path = ?
        ORDER BY path
        """,
        (target_key,),
    )


def _print_row_kv(row: sqlite3.Row) -> None:
    keys = list(row.keys())
    width = max(len(str(k)) for k in keys) if keys else 0
    for key in keys:
        value = row[key]
        print(f"  {str(key).ljust(width)} : {_format_timestamp(str(key), value)}")


def run_target_verify(con: sqlite3.Connection, path_arg: str) -> int:
    photos_dir_raw = os.environ.get("PHOTOAI_PHOTOS_DIR", "").strip()
    photos_dir = Path(photos_dir_raw).expanduser().resolve() if photos_dir_raw else None

    target_key, is_dir, from_fs = _detect_target(path_arg, photos_dir)
    mode = "directory" if is_dir else "file"

    header("PhotoAI target verify")
    print(f"Input: {path_arg}")
    print(f"Interpreted as: {mode}")
    print(f"Lookup key: {target_key}")
    if photos_dir is not None:
        print(f"PHOTOAI_PHOTOS_DIR: {photos_dir}")
    if not from_fs:
        print("Note: path not found on filesystem, using DB-relative interpretation.")

    image_rows = _fetch_image_rows_for_target(con, target_key, is_dir)
    if not image_rows:
        print("\nNo matching images found in SQLite.")
        return 1

    sha1s = [str(r["sha1"]) for r in image_rows]
    section(f"Images matched: {len(image_rows)}")
    print_rows(
        [
            (
                r["sha1"],
                r["path"],
                r["mime"],
                r["w"],
                r["h"],
                r["duration"],
                r["file_size"],
                r["taken_at"],
                r["place_name"],
            )
            for r in image_rows
        ],
        max_rows=max(30, len(image_rows)),
        headers=["sha1", "path", "mime", "w", "h", "duration", "file_size", "taken_at", "place_name"],
    )

    section("Detailed dump per image (all SQLite fields)")
    for i, image_row in enumerate(image_rows, start=1):
        sha1 = str(image_row["sha1"])
        print(f"\n[{i}/{len(image_rows)}] sha1={sha1}")
        print("IMAGE")
        _print_row_kv(image_row)

        caption_rows = fetch_all(
            con,
            """
            SELECT
              sha1,
              lang,
              model,
              substr(caption, 1, 140) AS caption_it,
              substr(json_extract(params_json, '$.caption_en'), 1, 140) AS caption_en,
              updated_at
            FROM captions
            WHERE sha1=?
            ORDER BY updated_at DESC
            """,
            (sha1,),
        )
        print("\nCAPTIONS")
        if caption_rows:
            print_rows(
                [
                    (r["sha1"], r["lang"], r["model"], r["caption_it"], r["caption_en"], r["updated_at"])
                    for r in caption_rows
                ],
                max_rows=max(30, len(caption_rows)),
                headers=["sha1", "lang", "model", "caption_it", "caption_en", "updated_at"],
            )
        else:
            print("  (no rows)")

        tags_en_rows = fetch_all(
            con,
            """
            SELECT c.sha1, je.value AS tag_en
            FROM captions c, json_each(c.params_json, '$.tags_en') je
            WHERE c.sha1=?
            ORDER BY je.value
            """,
            (sha1,),
        )
        print("\nTAGS_EN")
        if tags_en_rows:
            print_rows(
                [(r["sha1"], r["tag_en"]) for r in tags_en_rows],
                max_rows=max(30, len(tags_en_rows)),
                headers=["sha1", "tag_en"],
            )
        else:
            print("  (no rows)")

        tag_rows = fetch_all(
            con,
            "SELECT * FROM tags WHERE sha1=? ORDER BY score DESC, tag",
            (sha1,),
        )
        print("\nTAGS")
        if tag_rows:
            tag_headers = list(tag_rows[0].keys())
            print_rows(
                [tuple(r[k] for k in tag_headers) for r in tag_rows],
                max_rows=max(30, len(tag_rows)),
                headers=tag_headers,
            )
        else:
            print("  (no rows)")

        job_rows = fetch_all(
            con,
            "SELECT * FROM jobs WHERE sha1=? ORDER BY updated_at DESC",
            (sha1,),
        )
        print("\nJOBS")
        if job_rows:
            job_headers = list(job_rows[0].keys())
            print_rows(
                [tuple(r[k] for k in job_headers) for r in job_rows],
                max_rows=max(30, len(job_rows)),
                headers=job_headers,
            )
        else:
            print("  (no rows)")

    section("Chroma presence for matched sha1")
    chroma_dir_raw = os.environ.get("PHOTOAI_CHROMA_DIR", "").strip()
    img_ids: set = set()
    cap_ids: set = set()
    _chroma_ok = False
    if not chroma_dir_raw:
        print("  WARN: PHOTOAI_CHROMA_DIR not set, skipping Chroma lookup")
    else:
        try:
            import chromadb
            from chromadb.config import Settings

            image_collection = os.environ.get("PHOTOAI_COLLECTION", "images_openclip_vitl14_336")
            captions_collection = os.environ.get("PHOTOAI_CAPTIONS_COLLECTION", "captions_openclip_vitl14_336")
            client = chromadb.PersistentClient(
                path=str(Path(chroma_dir_raw).expanduser()),
                settings=Settings(anonymized_telemetry=False),
            )
            img_col = client.get_collection(image_collection)
            cap_col = client.get_collection(captions_collection)
            for chunk in _chunked(sha1s):
                got_img = img_col.get(ids=chunk, include=[])
                got_cap = cap_col.get(ids=chunk, include=[])
                img_ids.update(got_img.get("ids", []))
                cap_ids.update(got_cap.get("ids", []))
            _chroma_ok = True
        except Exception as exc:
            print(f"  WARN: Chroma lookup failed ({exc})")

    if _chroma_ok:
        print_rows(
            [(s, (s in img_ids), (s in cap_ids)) for s in sha1s],
            max_rows=max(30, len(sha1s)),
            headers=["sha1", "in_chroma_images", "in_chroma_captions"],
        )

    section("Thumbnail presence")
    thumb_dir_raw = os.environ.get("PHOTOAI_THUMB_DIR", "").strip()
    if not thumb_dir_raw:
        print("  WARN: PHOTOAI_THUMB_DIR not set, skipping thumb check")
    else:
        thumb_dir = Path(thumb_dir_raw).expanduser().resolve()
        print(f"  Thumb dir: {thumb_dir}\n")
        thumb_rows = []
        for s in sha1s:
            thumb_path = thumb_dir / f"{s}.jpg"
            exists = thumb_path.exists()
            size_str = str(thumb_path.stat().st_size) if exists else "-"
            thumb_rows.append((s, exists, size_str, str(thumb_path)))
        print_rows(
            thumb_rows,
            max_rows=max(30, len(thumb_rows)),
            headers=["sha1", "thumb_exists", "size_bytes", "thumb_path"],
        )
        missing = [r for r in thumb_rows if not r[1]]
        if missing:
            print(f"\n  WARN: {len(missing)} thumbnail(s) missing")

    return 0


def run_chroma_verify(con: sqlite3.Connection, missing_limit: int, problems: int, warnings: int) -> Tuple[int, int]:
    section("Chroma verify")

    chroma_dir_raw = os.environ.get("PHOTOAI_CHROMA_DIR", "").strip()
    if not chroma_dir_raw:
        warnings += 1
        print("  WARN: PHOTOAI_CHROMA_DIR is not set, skipping Chroma checks")
        return problems, warnings

    captions_collection = os.environ.get("PHOTOAI_CAPTIONS_COLLECTION", "captions_openclip_vitl14_336")
    chroma_dir = Path(chroma_dir_raw).expanduser()

    print(f"  Chroma dir: {chroma_dir}")
    print(f"  Captions collection: {captions_collection}")

    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as exc:
        warnings += 1
        print(f"  WARN: cannot import chromadb ({exc}), skipping Chroma checks")
        return problems, warnings

    try:
        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_collection(captions_collection)
    except Exception as exc:
        warnings += 1
        print(f"  WARN: cannot open Chroma collection ({exc})")
        return problems, warnings

    chroma_count = col.count()
    row = fetch_one(con, "SELECT COUNT(*) AS n FROM captions")
    sqlite_captions = int(row["n"]) if row is not None else 0
    print(f"  Chroma count: {chroma_count}")
    print(f"  SQLite captions count: {sqlite_captions}")

    sha_rows = fetch_all(
        con,
        "SELECT sha1 FROM captions ORDER BY updated_at DESC LIMIT ?",
        (200,),
    )
    sha1s = [r["sha1"] for r in sha_rows]

    if not sha1s:
        warnings += 1
        print("  WARN: no captions in SQLite, skipping Chroma id sample and semantic query")
        return problems, warnings

    got = col.get(ids=sha1s, include=["metadatas", "documents"])
    got_ids = set(got.get("ids", []))
    missing = [s for s in sha1s if s not in got_ids]
    print(f"  Sample check (last {len(sha1s)}): present={len(got_ids)} missing={len(missing)}")
    if missing:
        problems += 1
        print(f"  PROBLEM: missing in Chroma for {len(missing)} sampled ids")
        print_rows(
            [(m,) for m in missing[: min(20, missing_limit)]],
            max_rows=min(20, missing_limit),
            headers=["missing_sha1"],
        )

    try:
        import open_clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        model = model.to(device).eval()

        query_text = os.environ.get("PHOTOAI_TEST_QUERY_IT", "una foto di un gatto")
        tokens = tokenizer([query_text]).to(device)
        feats = model.encode_text(tokens)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        q_emb = feats.detach().cpu().float().tolist()[0]

        res = col.query(query_embeddings=[q_emb], n_results=5, include=["metadatas", "documents", "distances"])
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        mets = res.get("metadatas", [[]])[0]
        docs = res.get("documents", [[]])[0]

        print(f"  Query test (IT): {query_text}")
        for i, _id in enumerate(ids):
            dist = dists[i] if i < len(dists) else None
            meta = mets[i] if i < len(mets) else {}
            doc = docs[i] if i < len(docs) else ""
            path = meta.get("path", "")
            print(f"    #{i+1} id={_id} distance={dist} path={path} caption_it={doc[:120]}")
    except Exception as exc:
        warnings += 1
        print(f"  WARN: semantic Chroma query test failed ({exc})")

    return problems, warnings


def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Verify PhotoAI data quality and completeness (SQLite + Chroma).")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run the full verification suite (default behavior).",
    )
    ap.add_argument(
        "--path",
        default="",
        help="File path or directory to inspect related records (SQLite + Chroma) for that target.",
    )
    ap.add_argument(
        "--missing-limit",
        type=int,
        default=30,
        help="Max rows to print for each missing-data sample (default: 30)",
    )
    ap.add_argument(
        "--missing-only",
        action="store_true",
        help="Print only completeness/missing-data sections (skip other checks)",
    )
    args = ap.parse_args()

    if args.all and args.path:
        print("ERROR: use either --all or --path, not both.")
        return 2

    run_all = args.all or not args.path

    missing_limit = max(1, args.missing_limit)

    sqlite_dir = Path(must_env("PHOTOAI_SQLITE_DIR")).expanduser()
    db_path = sqlite_dir / "photo_ai.sqlite"
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}")
        return 2

    header("PhotoAI verify report")
    print(f"DB: {db_path}")

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys=ON;")
    con.row_factory = sqlite3.Row

    problems = 0
    warnings = 0

    try:
        if not run_all:
            return run_target_verify(con, args.path)

        run_filesystem_mapping_prospect(con)

        # 1) counts
        section("Counts: images / captions / tagged")
        row = fetch_one(
            con,
            """
            SELECT
              (SELECT COUNT(*) FROM images) AS images,
              (SELECT COUNT(*) FROM captions) AS captions,
              (SELECT COUNT(DISTINCT sha1) FROM tags) AS images_with_tags
            """
        )
        print(f"  images={row['images']} captions={row['captions']} images_with_tags={row['images_with_tags']}")

        section("Completeness: missing info overview")
        row = fetch_one(
            con,
            """
            SELECT
              (SELECT COUNT(*) FROM images i LEFT JOIN captions c ON c.sha1=i.sha1 WHERE c.sha1 IS NULL) AS images_without_caption,
              (SELECT COUNT(*) FROM images i LEFT JOIN (SELECT DISTINCT sha1 FROM tags) t ON t.sha1=i.sha1 WHERE t.sha1 IS NULL) AS images_without_tags,
              (SELECT COUNT(*) FROM images WHERE taken_at IS NULL) AS missing_taken_at,
              (SELECT COUNT(*) FROM images WHERE gps_lat IS NULL OR gps_lon IS NULL) AS missing_gps,
              (SELECT COUNT(*) FROM images WHERE country_code IS NULL OR trim(country_code)='') AS missing_country_code,
              (SELECT COUNT(*) FROM images WHERE city IS NULL OR trim(city)='') AS missing_city,
              (SELECT COUNT(*) FROM images WHERE place_name IS NULL OR trim(place_name)='') AS missing_place_name,
              (SELECT COUNT(*) FROM captions WHERE json_extract(params_json,'$.caption_en') IS NULL) AS captions_missing_caption_en,
                            (SELECT COUNT(*) FROM jobs WHERE status='done_base') AS job_done_base,
              (SELECT COUNT(*) FROM jobs WHERE status='error') AS job_errors
            """
        )
        print(
            "  "
            f"images_without_caption={row['images_without_caption']} "
            f"images_without_tags={row['images_without_tags']} "
            f"missing_taken_at={row['missing_taken_at']} "
            f"missing_gps={row['missing_gps']} "
            f"missing_country_code={row['missing_country_code']} "
            f"missing_city={row['missing_city']} "
            f"missing_place_name={row['missing_place_name']} "
            f"captions_missing_caption_en={row['captions_missing_caption_en']} "
            f"job_done_base={row['job_done_base']} "
            f"job_errors={row['job_errors']}"
        )

        section(f"Missing sample: images without captions (max {missing_limit})")
        rows = fetch_all(
            con,
            """
            SELECT i.sha1, i.path
            FROM images i
            LEFT JOIN captions c ON c.sha1=i.sha1
            WHERE c.sha1 IS NULL
            ORDER BY i.added_at DESC
            LIMIT ?
            """,
            (missing_limit,),
        )
        print_rows([(r["sha1"], r["path"]) for r in rows], max_rows=missing_limit, headers=["sha1", "path"])

        section(f"Random sample: images without captions (max {missing_limit})")
        rows = fetch_all(
            con,
            """
            SELECT i.sha1, i.path
            FROM images i
            LEFT JOIN captions c ON c.sha1=i.sha1
            WHERE c.sha1 IS NULL
            ORDER BY random()
            LIMIT ?
            """,
            (missing_limit,),
        )
        print_rows([(r["sha1"], r["path"]) for r in rows], max_rows=missing_limit, headers=["sha1", "path"])

        section(f"Missing sample: images without tags (max {missing_limit})")
        rows = fetch_all(
            con,
            """
            SELECT i.sha1, i.path
            FROM images i
            LEFT JOIN (SELECT DISTINCT sha1 FROM tags) t ON t.sha1=i.sha1
            WHERE t.sha1 IS NULL
            ORDER BY i.added_at DESC
            LIMIT ?
            """,
            (missing_limit,),
        )
        print_rows([(r["sha1"], r["path"]) for r in rows], max_rows=missing_limit, headers=["sha1", "path"])

        section(f"Random sample: images without tags (max {missing_limit})")
        rows = fetch_all(
            con,
            """
            SELECT i.sha1, i.path
            FROM images i
            LEFT JOIN (SELECT DISTINCT sha1 FROM tags) t ON t.sha1=i.sha1
            WHERE t.sha1 IS NULL
            ORDER BY random()
            LIMIT ?
            """,
            (missing_limit,),
        )
        print_rows([(r["sha1"], r["path"]) for r in rows], max_rows=missing_limit, headers=["sha1", "path"])

        section(f"Missing sample: images with missing geotime fields (max {missing_limit})")
        rows = fetch_all(
            con,
            """
            SELECT
              sha1,
              path,
              taken_at,
              gps_lat,
              gps_lon,
              country_code,
              city,
              place_name
            FROM images
            WHERE taken_at IS NULL
               OR gps_lat IS NULL
               OR gps_lon IS NULL
               OR country_code IS NULL
               OR trim(country_code)=''
               OR city IS NULL
               OR trim(city)=''
               OR place_name IS NULL
               OR trim(place_name)=''
            ORDER BY added_at DESC
            LIMIT ?
            """,
            (missing_limit,),
        )
        print_rows(
            [
                (
                    r["sha1"],
                    r["path"],
                    r["taken_at"],
                    r["gps_lat"],
                    r["gps_lon"],
                    r["country_code"],
                    r["city"],
                    r["place_name"],
                )
                for r in rows
            ],
            max_rows=missing_limit,
            headers=["sha1", "path", "taken_at", "gps_lat", "gps_lon", "country_code", "city", "place_name"],
        )

        section(f"Random sample: images with missing geotime fields (max {missing_limit})")
        rows = fetch_all(
            con,
            """
            SELECT
              sha1,
              path,
              taken_at,
              gps_lat,
              gps_lon,
              country_code,
              city,
              place_name
            FROM images
            WHERE taken_at IS NULL
               OR gps_lat IS NULL
               OR gps_lon IS NULL
               OR country_code IS NULL
               OR trim(country_code)=''
               OR city IS NULL
               OR trim(city)=''
               OR place_name IS NULL
               OR trim(place_name)=''
            ORDER BY random()
            LIMIT ?
            """,
            (missing_limit,),
        )
        print_rows(
            [
                (
                    r["sha1"],
                    r["path"],
                    r["taken_at"],
                    r["gps_lat"],
                    r["gps_lon"],
                    r["country_code"],
                    r["city"],
                    r["place_name"],
                )
                for r in rows
            ],
            max_rows=missing_limit,
            headers=["sha1", "path", "taken_at", "gps_lat", "gps_lon", "country_code", "city", "place_name"],
        )

        section(f"Random sample: complete images (geo + locality + taken_at) (max {missing_limit})")
        rows = fetch_all(
            con,
            """
            SELECT
              sha1,
              path,
              datetime(taken_at, 'unixepoch') AS taken_at_utc,
              gps_lat,
              gps_lon,
              city,
              region,
              country,
              country_code,
              place_name,
              location_source
            FROM images
            WHERE taken_at IS NOT NULL
              AND gps_lat IS NOT NULL
              AND gps_lon IS NOT NULL
              AND country_code IS NOT NULL
              AND trim(country_code) <> ''
              AND city IS NOT NULL
              AND trim(city) <> ''
              AND place_name IS NOT NULL
              AND trim(place_name) <> ''
            ORDER BY random()
            LIMIT ?
            """,
            (missing_limit,),
        )
        print_rows(
            [
                (
                    r["sha1"],
                    r["path"],
                    r["taken_at_utc"],
                    r["gps_lat"],
                    r["gps_lon"],
                    r["city"],
                    r["region"],
                    r["country"],
                    r["country_code"],
                    r["place_name"],
                    r["location_source"],
                )
                for r in rows
            ],
            max_rows=missing_limit,
            headers=[
                "sha1",
                "path",
                "taken_at_utc",
                "gps_lat",
                "gps_lon",
                "city",
                "region",
                "country",
                "country_code",
                "place_name",
                "location_source",
            ],
        )

        if args.missing_only:
            header("Summary")
            print(f"Problems: {problems}")
            print(f"Warnings: {warnings}")
            print("Result: OK")
            return 0

        # 2) jobs breakdown
        section("Jobs status breakdown")
        rows = fetch_all(
            con,
            """
            SELECT step, status, COUNT(*) AS n
            FROM jobs
            GROUP BY step, status
            ORDER BY step, status
            """
        )
        print_rows([(r["step"], r["status"], r["n"]) for r in rows], max_rows=200, headers=["step", "status", "count"])

        # quick add_all job view
        section("Jobs: add_all quick view (queued/processing/done/done_base/error)")
        row = fetch_one(
            con,
            """
            SELECT
              (SELECT COUNT(*) FROM jobs WHERE step='add_all' AND status='queued') AS queued,
              (SELECT COUNT(*) FROM jobs WHERE step='add_all' AND status='processing') AS processing,
              (SELECT COUNT(*) FROM jobs WHERE step='add_all' AND status='done') AS done,
              (SELECT COUNT(*) FROM jobs WHERE step='add_all' AND status='done_base') AS done_base,
              (SELECT COUNT(*) FROM jobs WHERE step='add_all' AND status='error') AS error
            """
        )
        print(
            "  "
            f"queued={row['queued']} "
            f"processing={row['processing']} "
            f"done={row['done']} "
            f"done_base={row['done_base']} "
            f"error={row['error']}"
        )
        if row["error"] and row["error"] > 0:
            warnings += 1

        # last errors
        section("Last 20 errors (if any)")
        rows = fetch_all(
            con,
            """
            SELECT
              j.sha1,
              i.path,
              j.step,
              j.status,
              substr(j.detail,1,140) AS detail
            FROM jobs j
            LEFT JOIN images i ON i.sha1 = j.sha1
            WHERE j.status='error'
            ORDER BY updated_at DESC
            LIMIT 20
            """
        )
        print_rows(
            [(r["sha1"], r["path"] or "(no image path)", r["step"], r["status"], r["detail"]) for r in rows],
            max_rows=30,
            headers=["sha1", "path", "step", "status", "detail"],
        )

        # 3) captions sanity
        section("Captions by lang")
        rows = fetch_all(con, "SELECT lang, COUNT(*) AS n FROM captions GROUP BY lang ORDER BY n DESC")
        print_rows([(r["lang"], r["n"]) for r in rows], max_rows=50, headers=["lang", "count"])

        section("Last 10 captions: IT + EN (from params_json)")
        rows = fetch_all(
            con,
            """
            SELECT
              sha1,
              lang,
              substr(caption,1,90) AS caption_it,
              substr(json_extract(params_json,'$.caption_en'),1,90) AS caption_en
            FROM captions
            ORDER BY updated_at DESC
            LIMIT 10
            """
        )
        print_rows(
            [(r["sha1"], r["lang"], r["caption_it"], r["caption_en"]) for r in rows],
            max_rows=15,
            headers=["sha1", "lang", "caption_it", "caption_en"],
        )

        section("Captions missing caption_en in params_json (should be 0)")
        rows = fetch_all(
            con,
            """
            SELECT sha1
            FROM captions
            WHERE json_extract(params_json,'$.caption_en') IS NULL
            LIMIT 50
            """
        )
        if rows:
            problems += 1
            print("  PROBLEM: found captions without caption_en in params_json")
        print_rows([(r["sha1"],) for r in rows], max_rows=50, headers=["sha1"])

        section("Last 10 captions: tags_en (from params_json)")
        rows = fetch_all(
            con,
            """
            SELECT c.sha1, group_concat(je.value, ', ') AS tags_en
            FROM captions c
            LEFT JOIN json_each(c.params_json, '$.tags_en') je
            GROUP BY c.sha1
            ORDER BY c.updated_at DESC
            LIMIT 10
            """,
        )
        print_rows([(r["sha1"], r["tags_en"]) for r in rows], max_rows=15, headers=["sha1", "tags_en"])

        section("Top 30 tags_en by frequency")
        rows = fetch_all(
            con,
            """
            SELECT je.value AS tag_en, COUNT(*) AS n
            FROM captions c, json_each(c.params_json, '$.tags_en') je
            GROUP BY je.value
            ORDER BY n DESC
            LIMIT 30
            """,
        )
        print_rows([(r["tag_en"], r["n"]) for r in rows], max_rows=40, headers=["tag_en", "count"])

        # 4) tags sanity
        section("Tags for last 10 captioned images")
        rows = fetch_all(
            con,
            """
            SELECT t.sha1, group_concat(t.tag, ', ') AS tags_it
            FROM tags t
            WHERE t.sha1 IN (SELECT sha1 FROM captions ORDER BY updated_at DESC LIMIT 10)
            GROUP BY t.sha1
            """
        )
        print_rows([(r["sha1"], r["tags_it"]) for r in rows], max_rows=15, headers=["sha1", "tags_it"])

        section("Top 30 tags by frequency")
        rows = fetch_all(
            con,
            """
            SELECT tag, COUNT(*) AS n
            FROM tags
            GROUP BY tag
            ORDER BY n DESC
            LIMIT 30
            """
        )
        print_rows([(r["tag"], r["n"]) for r in rows], max_rows=40, headers=["tag", "count"])

        section("Heuristic: tags without accented vowels (often EN-ish) - top 30")
        rows = fetch_all(
            con,
            """
            SELECT tag, COUNT(*) AS n
            FROM tags
            WHERE tag GLOB '*[a-z]*'
              AND tag NOT GLOB '*[àèéìòóù]*'
            GROUP BY tag
            ORDER BY n DESC
            LIMIT 30
            """
        )
        print_rows([(r["tag"], r["n"]) for r in rows], max_rows=40, headers=["tag", "count"])

        # 5) integrity checks
        section("Integrity: captions without images (should be 0)")
        row = fetch_one(
            con,
            """
            SELECT COUNT(*) AS n
            FROM captions c
            LEFT JOIN images i ON i.sha1 = c.sha1
            WHERE i.sha1 IS NULL
            """
        )
        print(f"  orphans={row['n']}")
        if row["n"] and row["n"] > 0:
            problems += 1

        section("Integrity: tags without images (should be 0)")
        row = fetch_one(
            con,
            """
            SELECT COUNT(*) AS n
            FROM tags t
            LEFT JOIN images i ON i.sha1 = t.sha1
            WHERE i.sha1 IS NULL
            """
        )
        print(f"  orphans={row['n']}")
        if row["n"] and row["n"] > 0:
            problems += 1

        section("Integrity: jobs without images (should be 0)")
        row = fetch_one(
            con,
            """
            SELECT COUNT(*) AS n
            FROM jobs j
            LEFT JOIN images i ON i.sha1 = j.sha1
            WHERE i.sha1 IS NULL
            """
        )
        print(f"  orphans={row['n']}")
        if row["n"] and row["n"] > 0:
            problems += 1

        # 6) end-to-end join
        section("End-to-end (path + caption_it + caption_en + tags_it) last 10")
        rows = fetch_all(
            con,
            """
            SELECT
              i.path,
              substr(c.caption,1,90) AS caption_it,
              substr(json_extract(c.params_json,'$.caption_en'),1,90) AS caption_en,
              (SELECT group_concat(tag, ', ') FROM tags t WHERE t.sha1=i.sha1) AS tags_it
            FROM images i
            JOIN captions c ON c.sha1=i.sha1
            ORDER BY c.updated_at DESC
            LIMIT 10
            """
        )
        print_rows(
            [(r["path"], r["caption_it"], r["caption_en"], r["tags_it"]) for r in rows],
            max_rows=15,
            headers=["path", "caption_it", "caption_en", "tags_it"],
        )

        # 7) random sample
        section("Random sample (10): path + caption_it + tags_it")
        rows = fetch_all(
            con,
            """
            SELECT
              i.path,
              substr(c.caption,1,90) AS caption_it,
              substr((SELECT group_concat(tag, ', ') FROM tags t WHERE t.sha1=i.sha1),1,90) AS tags_it
            FROM images i
            JOIN captions c ON c.sha1=i.sha1
            ORDER BY random()
            LIMIT 10
            """
        )
        print_rows(
            [(r["path"], r["caption_it"], r["tags_it"]) for r in rows],
            max_rows=15,
            headers=["path", "caption_it", "tags_it"],
        )

        # 8) FTS smoke
        section("FTS smoke test (captions_fts)")
        if table_exists(con, "captions_fts"):
            rows = fetch_all(
                con,
                """
                SELECT sha1, snippet(captions_fts, 1, '[', ']', '…', 10) AS snip
                FROM captions_fts
                WHERE captions_fts MATCH 'documento'
                LIMIT 10
                """
            )
            print_rows([(r["sha1"], r["snip"]) for r in rows], max_rows=15, headers=["sha1", "snippet"])
        else:
            warnings += 1
            print("  WARN: captions_fts not found (schema not created or different DB)")

        # 9) Chroma checks
        problems, warnings = run_chroma_verify(con, missing_limit, problems, warnings)

        # Summary
        header("Summary")
        print(f"Problems: {problems}")
        print(f"Warnings: {warnings}")
        if problems > 0:
            print("Result: FAIL (fix the problems above)")
            return 1
        print("Result: OK")
        return 0

    finally:
        con.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

