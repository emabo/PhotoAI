#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv

import math
import numpy as np

def must_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"{name} is not set. Define it in .env or environment variables.")
    return v

def ensure_images_columns(con: sqlite3.Connection) -> None:
    # Adds missing columns safely (idempotent)
    cols = {r[1] for r in con.execute("PRAGMA table_info(images);").fetchall()}

    def add(col: str, decl: str):
        nonlocal cols
        if col in cols:
            return
        con.execute(f"ALTER TABLE images ADD COLUMN {col} {decl};")
        cols.add(col)

    add("gps_lat", "REAL")
    add("gps_lon", "REAL")
    add("gps_alt", "REAL")
    add("gps_lat_round", "REAL")
    add("gps_lon_round", "REAL")

    add("country", "TEXT")
    add("country_code", "TEXT")
    add("region", "TEXT")
    add("city", "TEXT")
    add("place_name", "TEXT")
    add("location_source", "TEXT")

    con.executescript("""
    CREATE INDEX IF NOT EXISTS idx_images_gps      ON images(gps_lat, gps_lon);
    CREATE INDEX IF NOT EXISTS idx_images_city     ON images(city);
    CREATE INDEX IF NOT EXISTS idx_images_country  ON images(country);
    """)

def ensure_cache_table(con: sqlite3.Connection) -> None:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS geocode_cache (
      lat_round REAL NOT NULL,
      lon_round REAL NOT NULL,
      country_code TEXT,
      country TEXT,
      region TEXT,
      city TEXT,
      place_name TEXT,
      geonameid INTEGER,
      dist_km REAL,
      updated_at REAL NOT NULL DEFAULT (unixepoch()),
      PRIMARY KEY (lat_round, lon_round)
    );
    CREATE INDEX IF NOT EXISTS idx_geocode_cache_updated ON geocode_cache(updated_at);
    """)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    # scalar haversine
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)

def load_geonames(con: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    # Load cities into arrays
    rows = con.execute("""
      SELECT geonameid, name, asciiname, latitude, longitude, country_code, admin1_code, population
      FROM geonames_cities
    """).fetchall()
    if not rows:
        raise RuntimeError("geonames_cities is empty. Run import_geonames.py first")

    geonameid = np.array([r[0] for r in rows], dtype=np.int64)
    lat = np.array([r[3] for r in rows], dtype=np.float64)
    lon = np.array([r[4] for r in rows], dtype=np.float64)

    meta: List[Dict[str, Any]] = []
    for r in rows:
        meta.append({
            "geonameid": r[0],
            "name": r[1],
            "asciiname": r[2],
            "lat": r[3],
            "lon": r[4],
            "country_code": r[5],
            "admin1_code": r[6],
            "population": r[7],
        })
    return geonameid, lat, lon, meta

def load_country_map(con: sqlite3.Connection) -> Dict[str, str]:
    rows = con.execute("SELECT country_code, country_name FROM geonames_countries").fetchall()
    return {r[0]: r[1] for r in rows}

def load_admin1_map(con: sqlite3.Connection) -> Dict[str, str]:
    # code like IT.07 -> name
    if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='geonames_admin1'").fetchone() is None:
        return {}
    rows = con.execute("SELECT code, name FROM geonames_admin1").fetchall()
    return {r[0]: r[1] for r in rows if r[0] and r[1]}

def build_tree(lat: np.ndarray, lon: np.ndarray):
    # Prefer scipy cKDTree if available for speed
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        return None

    xyz = to_unit_xyz(lat, lon)
    return cKDTree(xyz)

def nearest_city(lat0: float, lon0: float,
                 lat_arr: np.ndarray, lon_arr: np.ndarray, meta: List[Dict[str, Any]],
                 tree, k: int = 1) -> Tuple[Dict[str, Any], float]:
    if tree is None:
        raise RuntimeError(
            "scipy is not installed: it is required for fast KDTree lookup.\n"
            "Install with: pip install scipy"
        )
    xyz0 = to_unit_xyz(np.array([lat0]), np.array([lon0]))[0]
    dist_eu, idx = tree.query(xyz0, k=k)
    # idx can be scalar
    if np.isscalar(idx):
        idx0 = int(idx)
        city = meta[idx0]
        dist_km = haversine_km(lat0, lon0, city["lat"], city["lon"])
        return city, dist_km

    # If k>1: choose best by true haversine (rarely needed)
    best_city = None
    best_d = 1e18
    for i in np.atleast_1d(idx):
        c = meta[int(i)]
        d = haversine_km(lat0, lon0, c["lat"], c["lon"])
        if d < best_d:
            best_d = d
            best_city = c
    assert best_city is not None
    return best_city, best_d

def format_place(city: str, region: str, country: str) -> str:
    parts = [p for p in [city, region, country] if p]
    return ", ".join(parts)


