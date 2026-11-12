"""Convert TransLink GTFS stop records into a WGS 84 GeoJSON point layer."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_DIR / "data" / "metadata"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FIELDS = [
    "stop_id",
    "stop_code",
    "stop_name",
    "stop_desc",
    "stop_lat",
    "stop_lon",
    "zone_id",
    "location_type",
    "parent_station",
    "wheelchair_boarding",
]


def stops_path() -> Path:
    """Locate stops.txt in the snapshot recorded by source metadata."""
    source_file = METADATA_DIR / "source.json"
    if not source_file.exists():
        raise FileNotFoundError("Run scripts/01_download_gtfs.py first.")
    source = json.loads(source_file.read_text(encoding="utf-8"))
    path = (
        PROJECT_DIR
        / "data"
        / "extracted"
        / str(source["download_date"])
        / "stops.txt"
    )
    if not path.exists():
        raise FileNotFoundError(f"Required GTFS table is missing: {path}")
    return path


def build_stops(path: Path) -> gpd.GeoDataFrame:
    """Preserve source stop types and create points from valid GTFS coordinates."""
    stops = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = required.difference(stops.columns)
    if missing:
        raise ValueError(f"stops.txt lacks fields: {', '.join(sorted(missing))}")
    for field in OUTPUT_FIELDS:
        if field not in stops.columns:
            stops[field] = ""
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    if stops[["stop_lat", "stop_lon"]].isna().any().any():
        invalid = int(stops[["stop_lat", "stop_lon"]].isna().any(axis=1).sum())
        raise ValueError(f"stops.txt contains {invalid} records without valid coordinates.")
    geometry = gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"], crs="EPSG:4326")
    return gpd.GeoDataFrame(stops[OUTPUT_FIELDS], geometry=geometry, crs="EPSG:4326")


def main() -> None:
    """Write all GTFS stop and station records as GeoJSON."""
    stops = build_stops(stops_path())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / "transit_stops.geojson"
    stops.to_file(output, driver="GeoJSON", index=False)
    print(f"Transit stops: {len(stops):,} features -> {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Stop construction failed: {exc}") from exc
