#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SQLITE_DIR = os.environ.get("PHOTOAI_SQLITE_DIR")
if not SQLITE_DIR:
    raise RuntimeError(
    "PHOTOAI_SQLITE_DIR is not set. "
    "Define the SQLite DB directory in .env or as an environment variable."
    )

SQLITE_DIR = Path(SQLITE_DIR).expanduser()
DB_PATH = SQLITE_DIR / "photo_ai.sqlite"

SCHEMA = r"""
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- =========================
-- IMAGES (RELATIVE PATH)
-- =========================
CREATE TABLE IF NOT EXISTS images (
  sha1        TEXT PRIMARY KEY,
  path        TEXT NOT NULL,        -- ALWAYS relative to PHOTOS_DIR
  mtime       REAL,                 -- technical only (NOT for photo time search)
  w           INTEGER,
  h           INTEGER,
  duration    REAL NOT NULL DEFAULT 0,  -- video duration in seconds; photos stay at 0
  file_size   INTEGER,
  mime        TEXT,
  added_at    REAL NOT NULL DEFAULT (unixepoch()),

  taken_at        INTEGER,  -- UTC epoch: shot date/time (use ONLY this for time-based search)
  gps_lat         REAL,     -- WGS84
  gps_lon         REAL,     -- WGS84
  gps_alt         REAL,     -- meters (optional)
  gps_lat_round   REAL,     -- for geocode caching (e.g. round(lat,3))
  gps_lon_round   REAL,

  country         TEXT,
  country_code    TEXT,     -- "IT"
  region          TEXT,     -- e.g. "Lazio"
  city            TEXT,     -- e.g. "Rome"
  place_name      TEXT,     -- e.g. "Rome, Lazio, Italy"
  location_source TEXT      -- 'nominatim' | 'geonames' | 'manual'
);

CREATE INDEX IF NOT EXISTS idx_images_path     ON images(path);
CREATE INDEX IF NOT EXISTS idx_images_mtime    ON images(mtime);
CREATE INDEX IF NOT EXISTS idx_images_taken_at ON images(taken_at);
CREATE INDEX IF NOT EXISTS idx_images_gps      ON images(gps_lat, gps_lon);
CREATE INDEX IF NOT EXISTS idx_images_city     ON images(city);
CREATE INDEX IF NOT EXISTS idx_images_country  ON images(country);

-- =========================
-- CAPTIONS (1:1)
-- =========================
CREATE TABLE IF NOT EXISTS captions (
  sha1        TEXT PRIMARY KEY,
  caption     TEXT NOT NULL,
  lang        TEXT,
  model       TEXT,
  params_json TEXT,
  created_at  REAL NOT NULL DEFAULT (unixepoch()),
  updated_at  REAL,
  FOREIGN KEY (sha1) REFERENCES images(sha1) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_captions_model   ON captions(model);
CREATE INDEX IF NOT EXISTS idx_captions_created ON captions(created_at);

-- =========================
-- FULL TEXT SEARCH (CAPTION)
-- =========================
CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts
USING fts5(
  sha1 UNINDEXED,
  caption,
  tokenize = 'unicode61'
);

-- =========================
-- TAGS (N:1)
-- =========================
CREATE TABLE IF NOT EXISTS tags (
  sha1        TEXT NOT NULL,
  tag         TEXT NOT NULL,
  score       REAL,
  source      TEXT,
  created_at  REAL NOT NULL DEFAULT (unixepoch()),
  PRIMARY KEY (sha1, tag),
  FOREIGN KEY (sha1) REFERENCES images(sha1) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_tag   ON tags(tag);
CREATE INDEX IF NOT EXISTS idx_tags_sha1  ON tags(sha1);
CREATE INDEX IF NOT EXISTS idx_tags_score ON tags(score);

-- =========================
-- JOBS / PIPELINE STATUS
-- =========================
CREATE TABLE IF NOT EXISTS jobs (
  sha1        TEXT NOT NULL,
  step        TEXT NOT NULL,
  status      TEXT NOT NULL,        -- "queued", "processing", "done", "done_base", "error"
  detail      TEXT,
  updated_at  REAL NOT NULL DEFAULT (unixepoch()),
  PRIMARY KEY (sha1, step),
  FOREIGN KEY (sha1) REFERENCES images(sha1) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_jobs_step_status ON jobs(step, status);
"""

TRIGGERS = r"""
-- =========================
-- SYNC captions <-> FTS
-- =========================
CREATE TRIGGER IF NOT EXISTS captions_ai
AFTER INSERT ON captions
BEGIN
  INSERT INTO captions_fts (sha1, caption)
  VALUES (new.sha1, new.caption);
END;

CREATE TRIGGER IF NOT EXISTS captions_ad
AFTER DELETE ON captions
BEGIN
  DELETE FROM captions_fts WHERE sha1 = old.sha1;
END;

CREATE TRIGGER IF NOT EXISTS captions_au
AFTER UPDATE OF caption ON captions
BEGIN
  UPDATE captions_fts
  SET caption = new.caption
  WHERE sha1 = new.sha1;
END;
"""

def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(SCHEMA)
        con.executescript(TRIGGERS)
        con.commit()
    finally:
        con.close()

def main():
    init_db(DB_PATH)
    print(f"OK: SQLite DB ready at {DB_PATH}")

if __name__ == "__main__":
    main()

