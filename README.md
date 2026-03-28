# PhotoAI

PhotoAI is a local photo archive system with an ingest pipeline, vector indexing, metadata extraction, and a web interface.

The project is designed to:

- import photos and videos into an organized library
- extract technical and temporal metadata
- store information in SQLite
- create embeddings in ChromaDB for semantic search
- generate captions and tags
- enrich geographic location using GeoNames
- serve everything through a FastAPI web UI

## Main features

- ingest images and videos from a source directory
- support for JPEG, PNG, GIF, WebP, BMP, TIFF, HEIC/HEIF, and MP4
- extraction of `sha1`, dimensions, video duration, MIME, `taken_at`, and GPS
- GPS parsing from MP4 QuickTime / ISO6709 metadata
- locality enrichment (`country`, `region`, `city`, `place_name`) via GeoNames
- automatic caption generation and EN -> IT translation
- automatic Italian tag generation
- text and image-based search
- thumbnail generation for images and videos
- verification, deduplication, and SHA1-based copy utilities

---

## Project architecture

```text
photoai.py                  # main ingest/sync pipeline
server/app.py               # FastAPI server + HTML UI
run_uvicorn_external.sh     # start server exposed on LAN

init/
  init_sqlite.py            # initialize SQLite schema
  import_geonames.py        # import GeoNames dataset into SQLite

lib/
  caption_pipeline.py       # captioning, translation, tags, text embeddings
  chroma_image_index.py     # image embeddings and Chroma indexing
  chroma_sqlite_sync.py     # sync utilities between Chroma and SQLite
  exif_sqlite.py            # EXIF reading and date/GPS parsing
  geonames_location.py      # nearest city lookup and location enrichment
  media_sorter.py           # date/path-based copy/move organizer
  mime_sqlite.py            # MIME detection
  thumbnail_precompute.py   # thumbnails for images/videos

verify/
  verify.py                         # full DB/filesystem/Chroma audit
  resolve_same_sha1_files.py        # interactive duplicate resolution
  copy_files_missing_from_db.py     # copy files missing from DB while preserving relpath
  remove_files.py                   # interactive removal from filesystem/DB/Chroma/thumbs
```

---

## Requirements

### System

Recommended:

- Linux
- Python 3.11 or 3.12
- `ffmpeg` and `ffprobe` available in `PATH`
- optional: `exiftool` available in `PATH` (extra fallback metadata for uncommon or legacy media files)
- optional CUDA GPU for embeddings and captioning

### Python dependencies

The [requirements.txt](requirements.txt) file contains a minimal base. For the full project, some optional dependencies used by advanced features are also recommended.

Recommended installation:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install python-dotenv transformers accelerate sentencepiece scipy
```

Notes:

- `python-dotenv` is used by multiple scripts.
- `transformers`, `accelerate`, and `sentencepiece` are required for captioning/translation.
- `scipy` is recommended for fast geographic lookup using KDTree.
- `ffmpeg`/`ffprobe` are required for video thumbnails and primary video metadata parsing.
- `exiftool` is optional but recommended as fallback metadata reader for uncommon or legacy files.

---

## Environment configuration

The project uses a `.env` file or exported shell variables.

### Create your `.env` from template

```bash
cp .env.example .env
```

Then edit `.env` with your local paths and preferred runtime settings.

### Core variables

```env
PHOTOAI_PHOTOS_DIR=/path/to/photo/library
PHOTOAI_SQLITE_DIR=/path/to/sqlite/db
PHOTOAI_CHROMA_DIR=/path/to/chroma/db
```

### Pipeline / model variables

```env
PHOTOAI_DEVICE=cuda
PHOTOAI_COLLECTION=images_openclip_vitl14_336
PHOTOAI_CAPTIONS_COLLECTION=captions_openclip_vitl14_336
PHOTOAI_DTYPE=bf16
PHOTOAI_CAPTION_MODEL=Salesforce/blip2-flan-t5-xxl
PHOTOAI_TRANSLATE_MODEL=facebook/nllb-200-distilled-600M
PHOTOAI_CAPTION_STEP=caption
PHOTOAI_CAPTION_PASS_LIMIT=16
PHOTOAI_THUMB_DIR=/path/to/thumb/cache
PHOTOAI_THUMB_SIZE=256
```

### Web server variables

```env
PHOTOAI_DEFAULT_K=60
PHOTOAI_MAX_K=200
PHOTOAI_MAX_CANDIDATES=1000
PHOTOAI_RESTRICT_TO_PHOTOS_DIR=1
```

### Verification helper variables

```env
PHOTOAI_TEST_QUERY_IT=a photo of a cat
```

---

## Quick start

### 1. Initialize the SQLite database

```bash
python init/init_sqlite.py
```

This creates:

- `images`
- `captions`
- `tags`
- `jobs`
- `captions_fts`
- primary indexes and triggers

### 2. Import GeoNames (optional but recommended)

Download the data:

```bash
wget https://download.geonames.org/export/dump/cities500.zip
wget https://download.geonames.org/export/dump/countryInfo.txt
wget https://download.geonames.org/export/dump/admin1CodesASCII.txt
unzip cities500.zip -d data/geonames
mv countryInfo.txt data/geonames/
mv admin1CodesASCII.txt data/geonames/
```

Import into SQLite:

```bash
python init/import_geonames.py --geonames-dir data/geonames
```

### 3. Ingest new media

```bash
python photoai.py --from /path/to/new_media --recursive --prefer-metadata
```

### 4. Start the web UI

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Or, to expose it on the local network:

```bash
./run_uvicorn_external.sh
```

---

## Main tools

## 1) `photoai.py`

Main ingest pipeline.

### What it does

- reads files from a source directory
- copies or moves them into the final library
- computes SHA1
- stores/updates metadata in SQLite
- generates image embeddings in ChromaDB
- extracts EXIF and video metadata
- generates thumbnails
- generates captions and tags
- enriches geographic location using GeoNames

### Typical command

```bash
python photoai.py --from /path/to/new_images --recursive --prefer-metadata
```

### Main options

- `--from`: source directory
- `--to`: override final library path, otherwise uses `PHOTOAI_PHOTOS_DIR`
- `--move`: move instead of copy
- `--recursive`: scan subdirectories
- `--max-depth`: maximum depth for `media_sorter` if not using `--recursive`
- `--prefer-metadata`: prefer EXIF metadata when resolving date/name conflicts
- `--skip-location`: disable GeoNames enrichment
- `--skip-thumbs`: skip thumbnail generation
- `--skip-captions`: skip caption/tag generation
- `--dry-run`: simulate without writing
- `--sync-missing`: rescan `PHOTOAI_PHOTOS_DIR` and complete missing DB/Chroma/thumb/caption/tag data
- `--quiet`: reduce output in `--sync-missing`
- `--limit N`: limit visited files in `--sync-missing`
- `--sync-subdir RELPATH`: with `--sync-missing`, scan only one subfolder under `PHOTOAI_PHOTOS_DIR`
- `--sync-mime MIME`: with `--sync-missing`, scan only one MIME type (for example `image/jpeg`)

### Sync-missing filtered examples

```bash
# scan only one subfolder
python photoai.py --sync-missing --sync-subdir 2024/08

# scan only JPEG files
python photoai.py --sync-missing --sync-mime image/jpeg

# combine both filters
python photoai.py --sync-missing --sync-subdir 2024/08 --sync-mime image/jpeg --quiet
```

### Important notes

- MP4 files are treated as `base-only mime`: they receive metadata, thumbnails, GPS, and locality enrichment, but not full photo-style captioning
- MP4 videos can also provide GPS via QuickTime/ISO6709 tags when available
- `--sync-missing` uses the same GPS/locality enrichment logic as the normal ingest flow

---

## 2) `server/app.py`

FastAPI server exposing the HTML UI and the main APIs.

### What it does

- serves the web interface
- runs semantic search on the catalog
- serves thumbnails and original files
- supports image-to-image search
- shows media details and viewer pages
- supports image deletion through API endpoints

### Run locally

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

### Main endpoints

- `GET /` — home / main UI
- `GET /api/search_html` — HTML search results
- `POST /api/search_by_image_html` — search by uploaded image
- `POST /api/delete_images` — delete images
- `GET /photo/{sha1}` — media detail page
- `GET /viewer/{sha1}` — viewer page
- `GET /thumb/{sha1}.jpg` — thumbnail
- `GET /img/{sha1}` — original file

### Important variables

- `PHOTOAI_CHROMA_DIR`
- `PHOTOAI_SQLITE_DIR`
- `PHOTOAI_PHOTOS_DIR`
- `PHOTOAI_COLLECTION`
- `PHOTOAI_DEVICE`
- `PHOTOAI_THUMB_DIR`
- `PHOTOAI_THUMB_SIZE`
- `PHOTOAI_DEFAULT_K`
- `PHOTOAI_MAX_K`
- `PHOTOAI_MAX_CANDIDATES`
- `PHOTOAI_RESTRICT_TO_PHOTOS_DIR`

---

## 3) `run_uvicorn_external.sh`

Shell wrapper to expose the server on the local network.

### What it does

- automatically detects the local server IP
- starts `uvicorn` bound to that IP
- prints the URL to access the UI from other devices on the LAN

### Examples

```bash
./run_uvicorn_external.sh
PORT=9000 ./run_uvicorn_external.sh
HOST_IP=192.168.1.50 ./run_uvicorn_external.sh
```

---

## 4) `init/init_sqlite.py`

Initializes the project's SQLite schema.

### What it creates

- `images`
- `captions`
- `captions_fts`
- `tags`
- `jobs`
- indexes on path, GPS, city, country, timestamps
- triggers to sync `captions` and `captions_fts`

### Run

```bash
python init/init_sqlite.py
```

### Notes on the `images` schema

The table already includes:

- technical fields: `sha1`, `path`, `mtime`, `w`, `h`, `duration`, `file_size`, `mime`
- time field: `taken_at`
- geolocation fields: `gps_lat`, `gps_lon`, `gps_alt`
- locality fields: `country`, `country_code`, `region`, `city`, `place_name`, `location_source`

---

## 5) `init/import_geonames.py`

Imports GeoNames datasets into the SQLite database.

### What it does

- creates and populates `geonames_cities`
- creates and populates `geonames_countries`
- creates and populates `geonames_admin1`

### Example

```bash
python init/import_geonames.py --geonames-dir data/geonames
```

### Options

- `--truncate-cities`: clear `geonames_cities` before re-importing
- `--skip-admin1`: skip `admin1CodesASCII.txt`

### When it is needed

Use it if you want `photoai.py` and `--sync-missing` to transform raw GPS coordinates into human-readable places.

---

## 6) `verify/verify.py`

Audit and diagnostic tool for the project.

### What it checks

- filesystem state versus the database
- presence of data in SQLite and Chroma
- missing captions and tags
- missing GPS and locality data
- missing thumbnails
- failed or incomplete jobs
- MIME mismatches between unmapped files and DB entries
- unsupported and unrecognized file extensions in the filesystem

### Useful commands

```bash
python verify/verify.py
python verify/verify.py --all
python verify/verify.py --missing-only
python verify/verify.py --path "2025/2025_11_30/20251130_151233.mp4"
python verify/verify.py --path "2025/2025_11_30"
```

### Notable output

The report also includes:

- georeferenced file count (`con_georef(gps)`)
- count of images missing geo/time fields
- list of unsupported and unrecognized extensions
- examples of unmapped files with the same SHA1 but a different path

---

## 7) `verify/resolve_same_sha1_files.py`

Interactive tool to resolve filesystem duplicates that share the same content as an existing DB entry.

### What it does

- finds files with the same `sha1` as media already indexed in the DB
- asks which file should be kept
- can update the `path` stored in SQLite
- can update path metadata in Chroma
- can remove unmapped duplicates

### Examples

```bash
python verify/resolve_same_sha1_files.py
python verify/resolve_same_sha1_files.py /path/to/photos
python verify/resolve_same_sha1_files.py --limit 20
```

### Main modes

- `k`: keep the file already referenced by the DB
- `d`: reconnect the DB entry to another existing filesystem file
- `s`: skip
- `q`: quit

---

## 8) `verify/copy_files_missing_from_db.py`

SHA1-based selective copy tool.

### What it does

- recursively visits all files under `--from`
- computes `sha1`
- compares against `images.sha1`
- if the SHA1 already exists and the DB-linked file really exists under `PHOTOAI_PHOTOS_DIR` and is non-empty, it skips the file
- if the SHA1 exists in DB but the DB-linked filesystem file is missing or empty, the file is treated as recoverable and can be copied
- if the MIME is unsupported, it searches the photo library for a file with the same name and same SHA1; if found, it skips copying
- if the file is new, it copies it into `--to` while preserving the original relative path

### Examples

```bash
python verify/copy_files_missing_from_db.py --from /source/dir --to /destination/dir
python verify/copy_files_missing_from_db.py --from /source/dir --to /destination/dir --dry-run
python verify/copy_files_missing_from_db.py --from /source/dir --to /destination/dir --limit 500
```

### Options

- `--dry-run`: simulate without copying
- `--limit`: limit visited files
- `--overwrite`: overwrite files already present in the destination

### Final counters

The script also reports:

- `skipped_known_sha1`
- `db_known_but_missing_or_empty`
- `unsupported_seen`
- `skipped_unsupported_same_name_sha1`
- `copied`
- `skipped_dest_exists`

---

## 9) `verify/remove_files.py`

Interactive removal utility to delete media consistently across filesystem, SQLite, Chroma, and thumbnails.

### What it does

- removes original files from `PHOTOAI_PHOTOS_DIR`
- removes corresponding thumbnail cache entries
- removes linked rows from SQLite (`images`, `jobs`, `tags`, `captions`, `captions_fts`)
- removes entries from Chroma image/captions collections

### Main modes

- `--error`: iterates files with `jobs.status='error'` and asks confirmation per file
- without `--error`: removes files selected by `--path` (supports wildcard selectors)

### Examples

```bash
# interactive cleanup of error files
python verify/remove_files.py --error

# cleanup only error files under a path selector
python verify/remove_files.py --error --path '2024/**/*.mp4'

# remove selected files (wildcard) with confirmation per file
python verify/remove_files.py --path '2024/**/*.jpg'

# dry-run preview
python verify/remove_files.py --error --dry-run
```

---

## Internal modules (`lib/`)

These files are not intended as public CLI tools, but they are the internal engine of the project.

| Module | Role |
|---|---|
| `lib/media_sorter.py` | organizes copied/moved files in the final library |
| `lib/exif_sqlite.py` | reads EXIF, timestamps, and coordinates from images |
| `lib/mime_sqlite.py` | detects file MIME type |
| `lib/chroma_image_index.py` | generates image embeddings and manages indexing |
| `lib/caption_pipeline.py` | captioning, Italian translation, tags, and text embeddings |
| `lib/geonames_location.py` | nearest-city lookup and locality enrichment |
| `lib/thumbnail_precompute.py` | creates thumbnails for images and videos |
| `lib/chroma_sqlite_sync.py` | sync utilities between Chroma and SQLite |

---

## Supported media

The currently supported MIME types in code are:

- `image/jpeg`
- `image/png`
- `image/gif`
- `image/webp`
- `image/bmp`
- `image/tiff`
- `image/heic`
- `image/heif`
- `video/mp4`

---

## Recommended workflows

## Workflow A — Fresh setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install python-dotenv transformers accelerate sentencepiece scipy
python init/init_sqlite.py
python init/import_geonames.py --geonames-dir data/geonames
```

## Workflow B — Import new media

```bash
python photoai.py --from /path/to/new_media --recursive --prefer-metadata
```

## Workflow C — Recover missing derived data for an existing library

```bash
python photoai.py --sync-missing --quiet
```

## Workflow D — Full audit

```bash
python verify/verify.py --all
```

## Workflow E — Duplicate cleanup

```bash
python verify/resolve_same_sha1_files.py
```

## Workflow F — Differential copy from an external source

```bash
python verify/copy_files_missing_from_db.py --from /mnt/source --to /mnt/recovery --dry-run
```

## Workflow G — Remove broken/error files

```bash
python verify/remove_files.py --error
```

---

## Notes on video geolocation

PhotoAI can extract the following from MP4 files:

- video dimensions
- duration
- `taken_at` from `creation_time` (or equivalent tag names when available)
- GPS from `location` or `com.apple.quicktime.location.ISO6709` (or ExifTool GPS tags, when available)

If GPS is present in the video and GeoNames has been loaded, the system can also populate:

- `country`
- `country_code`
- `region`
- `city`
- `place_name`
- `location_source`

This works both in the normal ingest flow and in `--sync-missing` mode.

---

## Troubleshooting

### `ffmpeg` or `ffprobe` not found

Install FFmpeg and make sure the binaries are available in `PATH`.

### limited metadata on legacy files

Some legacy files may not expose rich metadata through FFmpeg tags alone.

- install `exiftool` and keep it in `PATH` to enable fallback extraction for duration/date/GPS when present

### location enrichment disabled

If you see a warning about GeoNames or SciPy:

- import GeoNames with `init/import_geonames.py`
- install `scipy`

### captioning unavailable

Install:

```bash
pip install transformers accelerate sentencepiece
```

### files exist in DB but are missing in the filesystem

Use:

```bash
python verify/copy_files_missing_from_db.py --from /source --to /destination --dry-run
```

or:

```bash
python verify/resolve_same_sha1_files.py
```

### remove files across DB + Chroma + thumbs

Use:

```bash
python verify/remove_files.py --error
```

or target specific path selectors:

```bash
python verify/remove_files.py --path '2024/**/*.mp4'
```

---

## Repository status

This repository contains both production tools and diagnostic utilities for advanced local management of a personal photo library.

For GitHub, this `README.md` is the main landing page for the project.
