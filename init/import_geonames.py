#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

def must_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set. Define it in .env or environment variables.")
    return v

def table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None

def init_tables(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS geonames_cities (
          geonameid     INTEGER PRIMARY KEY,
          name          TEXT,
          asciiname     TEXT,
          latitude      REAL NOT NULL,
          longitude     REAL NOT NULL,
          country_code  TEXT,
          admin1_code   TEXT,
          population    INTEGER,
          timezone      TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_geonames_cities_latlon ON geonames_cities(latitude, longitude);
        CREATE INDEX IF NOT EXISTS idx_geonames_cities_cc     ON geonames_cities(country_code);
        CREATE INDEX IF NOT EXISTS idx_geonames_cities_pop    ON geonames_cities(population);

        CREATE TABLE IF NOT EXISTS geonames_countries (
          country_code TEXT PRIMARY KEY,
          country_name TEXT
        );

        CREATE TABLE IF NOT EXISTS geonames_admin1 (
          code       TEXT PRIMARY KEY,   -- es: IT.07
          name       TEXT,
          asciiname  TEXT,
          geonameid  INTEGER
        );
        """
    )

def import_countries(con: sqlite3.Connection, country_info_path: Path) -> int:
    # countryInfo.txt includes comment lines starting with '#'
    n = 0
    with country_info_path.open("r", encoding="utf-8", errors="replace") as f:
        rows = []
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            cc = parts[0].strip()
            name = parts[4].strip()  # Country name
            if not cc or not name:
                continue
            rows.append((cc.upper(), name))
        con.executemany(
            "INSERT INTO geonames_countries(country_code, country_name) VALUES (?, ?) "
            "ON CONFLICT(country_code) DO UPDATE SET country_name=excluded.country_name",
            rows,
        )
        n = len(rows)
    return n

def import_admin1(con: sqlite3.Connection, admin1_path: Path) -> int:
    # admin1CodesASCII.txt format: code \t name \t asciiname \t geonameid
    n = 0
    with admin1_path.open("r", encoding="utf-8", errors="replace") as f:
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            code = parts[0].strip()
            name = parts[1].strip()
            asciiname = parts[2].strip()
            try:
                geonameid = int(parts[3])
            except Exception:
                geonameid = None
            if not code:
                continue
            rows.append((code, name, asciiname, geonameid))
        con.executemany(
            "INSERT INTO geonames_admin1(code, name, asciiname, geonameid) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(code) DO UPDATE SET name=excluded.name, asciiname=excluded.asciiname, geonameid=excluded.geonameid",
            rows,
        )
        n = len(rows)
    return n

def import_cities(con: sqlite3.Connection, cities_path: Path, truncate: bool) -> int:
    if truncate:
        con.execute("DELETE FROM geonames_cities;")

    n = 0
    batch = []
    BATCH_SIZE = 5000

    # cities500.txt is TSV with many fields; we use:
    # 0 geonameid
    # 1 name
    # 2 asciiname
    # 4 latitude
    # 5 longitude
    # 8 country_code
    # 10 admin1_code
    # 14 population
    # 17 timezone
    with cities_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 18:
                continue
            try:
                geonameid = int(parts[0])
                name = parts[1]
                asciiname = parts[2]
                lat = float(parts[4])
                lon = float(parts[5])
                country_code = (parts[8] or "").upper()
                admin1_code = parts[10] or ""
                pop = int(parts[14]) if parts[14] else 0
                tz = parts[17] or ""
            except Exception:
                continue

            batch.append((geonameid, name, asciiname, lat, lon, country_code, admin1_code, pop, tz))

            if len(batch) >= BATCH_SIZE:
                con.executemany(
                    """
                    INSERT INTO geonames_cities
                      (geonameid, name, asciiname, latitude, longitude, country_code, admin1_code, population, timezone)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(geonameid) DO UPDATE SET
                      name=excluded.name,
                      asciiname=excluded.asciiname,
                      latitude=excluded.latitude,
                      longitude=excluded.longitude,
                      country_code=excluded.country_code,
                      admin1_code=excluded.admin1_code,
                      population=excluded.population,
                      timezone=excluded.timezone
                    """,
                    batch,
                )
                n += len(batch)
                batch.clear()

        if batch:
            con.executemany(
                """
                INSERT INTO geonames_cities
                  (geonameid, name, asciiname, latitude, longitude, country_code, admin1_code, population, timezone)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(geonameid) DO UPDATE SET
                  name=excluded.name,
                  asciiname=excluded.asciiname,
                  latitude=excluded.latitude,
                  longitude=excluded.longitude,
                  country_code=excluded.country_code,
                  admin1_code=excluded.admin1_code,
                  population=excluded.population,
                  timezone=excluded.timezone
                """,
                batch,
            )
            n += len(batch)

    return n

def main():
    import argparse

    load_dotenv()

    ap = argparse.ArgumentParser(description="Import GeoNames cities500 + countryInfo (+admin1) into SQLite.")
    ap.add_argument("--geonames-dir", required=True, help="Directory containing cities500.txt and countryInfo.txt")
    ap.add_argument("--truncate-cities", action="store_true", help="Delete geonames_cities before import")
    ap.add_argument("--skip-admin1", action="store_true", help="Skip admin1CodesASCII.txt even if present")
    args = ap.parse_args()

    geodir = Path(args.geonames_dir).expanduser()
    cities = geodir / "cities500.txt"
    countries = geodir / "countryInfo.txt"
    admin1 = geodir / "admin1CodesASCII.txt"

    if not cities.exists():
        raise SystemExit(f"Missing: {cities}")
    if not countries.exists():
        raise SystemExit(f"Missing: {countries}")

    sqlite_dir = Path(must_env("PHOTOAI_SQLITE_DIR")).expanduser()
    db_path = sqlite_dir / "photo_ai.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(db_path))
    try:
        init_tables(con)

        n_countries = import_countries(con, countries)
        print(f"Imported countries: {n_countries}")

        if admin1.exists() and not args.skip_admin1:
            n_admin1 = import_admin1(con, admin1)
            print(f"Imported admin1: {n_admin1}")
        else:
            print("Admin1 not imported (file missing or --skip-admin1).")

        n_cities = import_cities(con, cities, truncate=args.truncate_cities)
        print(f"Imported cities: {n_cities}")

        con.commit()
        print(f"OK: GeoNames imported into {db_path}")
    finally:
        con.close()

if __name__ == "__main__":
    main()
