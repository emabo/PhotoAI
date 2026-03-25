#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from lib.filename_date_parser import (
    parse_date_from_stem,
)

# Video MIME types that should be handled as "base-only" with video metadata extraction
BASE_ONLY_MIME_TYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/avi",
    "video/mpeg",
}


@dataclass
class ExtensionCount:
    counts: Dict[str, int]

    @staticmethod
    def new() -> "ExtensionCount":
        return ExtensionCount(counts={})

    def add(self, ext: str) -> None:
        self.counts[ext] = self.counts.get(ext, 0) + 1

    def print(self) -> None:
        if not self.counts:
            print("No files found by extension.")
            return
        print("\n\nFile count by extension:")
        sorted_items = sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)
        for ext, count in sorted_items:
            print(f"  {(ext if ext else '(no extension)')}: {count}")


@dataclass
class Stats:
    tot: int = 0
    copied: int = 0
    moved: int = 0
    renamed: int = 0
    already_present: int = 0
    skipped: int = 0

    def inc_tot(self) -> None:
        self.tot += 1

    def inc_copied(self) -> None:
        self.copied += 1

    def inc_moved(self) -> None:
        self.moved += 1

    def inc_renamed(self) -> None:
        self.renamed += 1

    def inc_already_present(self) -> None:
        self.already_present += 1

    def inc_skipped(self) -> None:
        self.skipped += 1

    def print_all(self) -> None:
        print(f"\n\nTotal number of files: {self.tot}")
        print(f"Skipped files: {self.skipped}")
        print(f"Already present files: {self.already_present}")
        print(f"Copied files: {self.copied}")
        print(f"Moved files: {self.moved}")
        print(f"Requiring renaming: {self.renamed}")


@dataclass
class Options:
    dir_from: str
    dir_to: str
    copy: bool
    dry_run: bool
    recursive: bool
    max_depth: int
    verbose: bool
    prefer_metadata_on_conflict: bool
    count_extensions: bool
    on_saved: Optional[Callable[[Path], None]] = None
    sha1_index: Optional[Dict[str, Path]] = None
    dest_files_by_size: Optional[Dict[int, List[Path]]] = None
    dest_sha1_cache: Optional[Dict[Path, str]] = None


def find_existing_date_dir(base_path: Path, year_path: str, date_prefix: str) -> Optional[str]:
    year_dir = base_path / year_path
    if not year_dir.exists() or not year_dir.is_dir():
        return None
    try:
        for entry in year_dir.iterdir():
            if entry.is_dir():
                dir_name = entry.name
                if dir_name == date_prefix:
                    return dir_name
                if dir_name.startswith(f"{date_prefix}."):
                    return dir_name
    except OSError:
        return None
    return None


def compute_hash(filename: str) -> str:
    hasher = hashlib.sha1()
    with open(filename, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_size_index(destination_root: Path, verbose: bool) -> Dict[int, List[Path]]:
    index: Dict[int, List[Path]] = {}
    if not destination_root.exists() or not destination_root.is_dir():
        return index

    for p in destination_root.rglob("*"):
        if not p.is_file():
            continue
        try:
            size = int(p.stat().st_size)
        except OSError:
            continue
        if size < 0:
            continue
        index.setdefault(size, []).append(p)

    if verbose:
        total_files = sum(len(v) for v in index.values())
        print(f"Destination size index loaded: {total_files} files")
    return index


def _find_existing_sha1_in_destination(opts: Options, destination_root: Path, src_sha1: str, src_size: int) -> Optional[Path]:
    if opts.sha1_index is None:
        opts.sha1_index = {}
    if opts.dest_sha1_cache is None:
        opts.dest_sha1_cache = {}
    if opts.dest_files_by_size is None:
        opts.dest_files_by_size = _build_size_index(destination_root, opts.verbose)

    cached = opts.sha1_index.get(src_sha1)
    if cached is not None:
        if cached.exists() and cached.is_file():
            return cached
        opts.sha1_index.pop(src_sha1, None)

    candidates = opts.dest_files_by_size.get(src_size, [])
    if not candidates:
        return None

    alive_candidates: List[Path] = []
    for p in candidates:
        if not p.exists() or not p.is_file():
            opts.dest_sha1_cache.pop(p, None)
            continue

        alive_candidates.append(p)

        dst_sha1 = opts.dest_sha1_cache.get(p)
        if not dst_sha1:
            try:
                dst_sha1 = compute_hash(str(p))
            except OSError:
                continue
            opts.dest_sha1_cache[p] = dst_sha1

        if dst_sha1 == src_sha1:
            opts.sha1_index[src_sha1] = p
            return p

    opts.dest_files_by_size[src_size] = alive_candidates
    return None


def _exif_tag_value(tags: Dict[str, object], key: str) -> Optional[str]:
    value = tags.get(key)
    if value is None:
        return None
    return str(value)


def detect_mime(path: Path) -> Optional[str]:
    """Detect MIME type of a file (simple heuristic)."""
    try:
        from lib import mime_sqlite as mimemod
        return mimemod.detect_mime(path)
    except Exception:
        pass
    
    import mimetypes
    mime, _ = mimetypes.guess_type(str(path), strict=False)
    return mime


def _parse_video_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse datetime strings from video metadata (ISO 8601 or EXIF-like formats)."""
    if not dt_str:
        return None
    
    dt_str = str(dt_str).strip()
    if not dt_str:
        return None
    
    # Try ISO 8601 format first
    for fmt in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y:%m:%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(dt_str.replace("Z", "+0000"), fmt)
        except ValueError:
            continue
    
    return None


def extract_date_from_video(filename: str, verbose: bool) -> Optional[datetime]:
    """Extract creation date from video file using ffprobe and exiftool fallback."""
    try:
        # Try ffprobe first
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-print_format", "json",
                "-show_format",
                str(filename),
            ],
            capture_output=True,
            timeout=10,
        )
        
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            tags = fmt.get("tags", {})
            
            # Try common keys
            for key in ("creation_time", "com.apple.quicktime.creationdate", "CreationTime", 
                       "MediaCreateDate", "TrackCreateDate", "CreateDate", "DATE"):
                dt_str = tags.get(key)
                if dt_str:
                    dt = _parse_video_datetime(dt_str)
                    if dt:
                        if verbose:
                            print(f"Video date from ffprobe [{key}]: {dt}")
                        return dt
    except Exception:
        pass
    
    # Fallback: try exiftool
    try:
        result = subprocess.run(
            ["exiftool", "-n", "-j", str(filename)],
            capture_output=True,
            timeout=10,
        )
        
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            if isinstance(data, list) and data:
                tags = data[0]
                
                # Try common exiftool keys
                for key in ("CreationTime", "MediaCreateDate", "TrackCreateDate", "CreateDate", 
                           "DateTimeOriginal", "ModifyDate"):
                    value = tags.get(key)
                    if value:
                        dt = _parse_video_datetime(str(value))
                        if dt:
                            if verbose:
                                print(f"Video date from exiftool [{key}]: {dt}")
                            return dt
    except Exception:
        pass
    
    return None


def extract_date(filename: str, verbose: bool, filename_for_fallback: Optional[str] = None) -> datetime:
    """Extract date from file metadata (video or photo), with filename fallback.
    
    Args:
        filename: Full path to the file
        verbose: Print debug info
        filename_for_fallback: Filename stem to use as fallback (e.g., 20240315_143045)
    
    Returns:
        datetime object extracted from metadata or filename
    
    Raises:
        RuntimeError if no date can be extracted
    """
    path = Path(filename)
    mime = detect_mime(path)
    
    # Try video extraction if it looks like a video
    if mime and mime in BASE_ONLY_MIME_TYPES:
        video_date = extract_date_from_video(filename, verbose)
        if video_date:
            return video_date
        if verbose:
            print(f"Could not extract date from video metadata, trying filename")
    
    # Try EXIF extraction for photos (or as fallback)
    try:
        import exifread  # type: ignore
        with open(filename, "rb") as handle:
            tags = exifread.process_file(handle, details=verbose)

        if verbose:
            tag_keys = [
                "EXIF ExifVersion",
                "EXIF PixelXDimension",
                "Image XResolution",
                "Image ImageDescription",
                "Image DateTime",
            ]
            for key in tag_keys:
                value = _exif_tag_value(tags, key)
                if value is not None:
                    print(f"{key.split(' ', 1)[-1]}: {value}")

        date_value = _exif_tag_value(tags, "Image DateTime")
        if date_value:
            return datetime.strptime(date_value, "%Y:%m:%d %H:%M:%S")
    except (ValueError, IsADirectoryError):
        pass
    except ImportError:
        pass
    
    # Fallback: try filename
    if filename_for_fallback:
        parsed_date, pattern = parse_date_from_stem(filename_for_fallback)
        if parsed_date is not None:
            if verbose:
                print(f"Date extracted from filename using pattern: {pattern}")
            return datetime(parsed_date.year, parsed_date.month, parsed_date.day, 0, 0, 0)
    
    raise RuntimeError("Date from file not found")


def extract_date_from_filename(filename: str, verbose: bool) -> date:
    parsed_date, pattern = parse_date_from_stem(filename)
    if parsed_date is not None:
        if verbose and pattern:
            print(f"Format found: {pattern}")
        return parsed_date
    raise RuntimeError("Date from filename not found")


def visit_dirs(
    directory: Path,
    callback: Callable[[os.DirEntry, Options, Stats, ExtensionCount], None],
    options: Options,
    file_stats: Stats,
    ext_count: ExtensionCount,
    depth: int,
) -> None:
    if not directory.is_dir():
        return
    with os.scandir(directory) as it:
        for entry in it:
            path = Path(entry.path)
            if path.is_dir() and (depth < options.max_depth or options.recursive):
                visit_dirs(path, callback, options, file_stats, ext_count, depth + 1)
            else:
                callback(entry, options, file_stats, ext_count)


def compute_file(entry: os.DirEntry, opts: Options, stats: Stats, ext_count: ExtensionCount) -> None:
    path_tmp = Path(entry.path)
    full_filename_from = str(path_tmp)

    if entry.is_dir():
        return

    path_from = Path(full_filename_from)

    if opts.count_extensions:
        ext = path_from.suffix[1:] if path_from.suffix else ""
        ext_count.add(ext)
        stats.inc_tot()
        return

    print(f"\nFilename: {full_filename_from}")

    stats.inc_tot()

    filename_to = path_from.stem
    original_filename_base = filename_to
    renamed_due_to_conflict = False
    if opts.verbose:
        print(f"Filename prefix: {filename_to}")

    date1_present = False
    date2_present = False
    date1 = datetime(2000, 1, 1, 0, 0, 0)
    date2 = date(2000, 1, 1)

    try:
        date1 = extract_date(full_filename_from, opts.verbose, filename_for_fallback=filename_to)
        date1_present = True
    except RuntimeError as exc:
        print(f"WARN: {exc}")

    try:
        date2 = extract_date_from_filename(filename_to, opts.verbose)
        date2_present = True
    except RuntimeError as exc:
        print(f"WARN: {exc}")

    chosen_date: Optional[datetime]
    if date1_present and date2_present and date1.date() != date2:
        if opts.prefer_metadata_on_conflict:
            if opts.verbose:
                print("Date conflict: using metadata date and renaming with %Y%m%d_%H%M%S")
            filename_to = (
                f"{date1.year:04}{date1.month:02}{date1.day:02}_"
                f"{date1.hour:02}{date1.minute:02}{date1.second:02}"
            )
            if filename_to != original_filename_base:
                renamed_due_to_conflict = True
            chosen_date = date1
        else:
            stats.inc_skipped()
            print("ERROR: Date from file and from filename are different, skipping image")
            chosen_date = None
    elif not date1_present and not date2_present:
        stats.inc_skipped()
        print("ERROR: Cannot extract date from file or filename")
        chosen_date = None
    else:
        if date1_present:
            chosen_date = date1
        else:
            chosen_date = datetime(date2.year, date2.month, date2.day, 0, 0, 0)

    if chosen_date is None:
        return

    src_sha1 = compute_hash(full_filename_from)
    try:
        src_size = int(path_from.stat().st_size)
    except OSError:
        src_size = -1

    date_only = chosen_date.date()

    orig_path = Path(opts.dir_to)

    existing_same_sha1 = _find_existing_sha1_in_destination(opts, orig_path, src_sha1, src_size)
    if existing_same_sha1 is not None:
        try:
            same_file = existing_same_sha1.resolve() == path_from.resolve()
        except Exception:
            same_file = False

        if not same_file:
            print(f"SKIP duplicate SHA1: {full_filename_from} (already in destination as {existing_same_sha1})")
            if opts.verbose:
                print(f"Duplicate by SHA1 found in destination: {existing_same_sha1}")
            if not opts.copy:
                if opts.verbose:
                    print(f"Deleting {full_filename_from}")
                if not opts.dry_run:
                    os.remove(full_filename_from)
            stats.inc_already_present()
            return

    year_dir = f"{date_only.year:04}"
    date_prefix = f"{date_only.year:04}_{date_only.month:02}_{date_only.day:02}"

    dir_name = find_existing_date_dir(orig_path, year_dir, date_prefix)
    if dir_name:
        if opts.verbose:
            print(f"Found existing directory: {dir_name}")
    else:
        dir_name = date_prefix

    extension = path_from.suffix[1:] if path_from.suffix else None
    base_dir = orig_path / year_dir / dir_name

    if not base_dir.exists():
        if not opts.dry_run:
            if opts.verbose:
                print(f"Create new directory: {base_dir}")
            base_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    done = False
    while not done:
        if counter > 0:
            candidate_name = f"{filename_to}_{counter:02}"
            if opts.verbose:
                print(f"New filename: {candidate_name}")
        else:
            candidate_name = filename_to

        if extension:
            final_name = f"{candidate_name}.{extension}"
        else:
            final_name = candidate_name

        full_filename_to = base_dir / final_name
        print(f"Destination path: {full_filename_to}")

        if full_filename_to.exists():
            if opts.verbose:
                print(f"File {full_filename_to} already exists")

            hash_dst = compute_hash(str(full_filename_to))
            if src_sha1 == hash_dst:
                if opts.verbose:
                    print("The two files are equal")
                if not opts.copy:
                    if opts.verbose:
                        print(f"Deleting {full_filename_from}")
                    if not opts.dry_run:
                        os.remove(full_filename_from)
                stats.inc_already_present()
                done = True
            else:
                if opts.verbose:
                    print("The two files are different")
                counter += 1
        else:
            if counter > 0:
                if opts.verbose:
                    print(f"Renaming file from {full_filename_from} to {full_filename_to}")
                stats.inc_renamed()
            elif renamed_due_to_conflict and full_filename_to.name != path_from.name:
                if opts.verbose:
                    print(f"Renaming file (date conflict) from {full_filename_from} to {full_filename_to}")
                stats.inc_renamed()
                renamed_due_to_conflict = False

            if opts.copy:
                if opts.verbose:
                    print(f"Copy {full_filename_from} to {full_filename_to}")
                if not opts.dry_run:
                    shutil.copy2(full_filename_from, full_filename_to)
                    stats.inc_copied()
                    if opts.on_saved is not None:
                        opts.on_saved(full_filename_to)
                    if opts.sha1_index is not None and src_sha1 not in opts.sha1_index:
                        opts.sha1_index[src_sha1] = full_filename_to
                    if opts.dest_sha1_cache is not None:
                        opts.dest_sha1_cache[full_filename_to] = src_sha1
                    if opts.dest_files_by_size is not None and src_size >= 0:
                        opts.dest_files_by_size.setdefault(src_size, []).append(full_filename_to)
            else:
                if opts.verbose:
                    print(f"Move {full_filename_from} to {full_filename_to}")
                if not opts.dry_run:
                    try:
                        os.rename(full_filename_from, full_filename_to)
                    except OSError as exc:
                        print(f"ERROR: Cannot rename file: {exc}, trying to move it")
                        print("Try to move it")
                        shutil.copy2(full_filename_from, full_filename_to)
                        os.remove(full_filename_from)
                    stats.inc_moved()
                    if opts.on_saved is not None:
                        opts.on_saved(full_filename_to)
                    if opts.sha1_index is not None and src_sha1 not in opts.sha1_index:
                        opts.sha1_index[src_sha1] = full_filename_to
                    if opts.dest_sha1_cache is not None:
                        opts.dest_sha1_cache[full_filename_to] = src_sha1
                    if opts.dest_files_by_size is not None and src_size >= 0:
                        opts.dest_files_by_size.setdefault(src_size, []).append(full_filename_to)
            done = True





