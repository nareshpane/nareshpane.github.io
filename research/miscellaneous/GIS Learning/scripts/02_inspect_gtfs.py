"""Inspect the downloaded TransLink GTFS archive and save an inventory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_DIR / "data" / "metadata"
CHUNK_SIZE = 250_000

IMPORTANT_FIELDS = {
    "agency.txt": ["agency_id", "agency_name", "agency_url", "agency_timezone"],
    "routes.txt": [
        "route_id",
        "agency_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_color",
        "route_text_color",
    ],
    "trips.txt": ["route_id", "service_id", "trip_id", "shape_id", "direction_id"],
    "stops.txt": ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"],
    "stop_times.txt": [
        "trip_id",
        "arrival_time",
        "departure_time",
        "stop_id",
        "stop_sequence",
        "shape_dist_traveled",
    ],
    "shapes.txt": [
        "shape_id",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_pt_sequence",
        "shape_dist_traveled",
    ],
}


def source_details() -> tuple[Path, dict[str, object]]:
    """Locate the exact archive recorded by the downloader."""
    source_file = METADATA_DIR / "source.json"
    if not source_file.exists():
        raise FileNotFoundError("Run scripts/01_download_gtfs.py before inspection.")
    metadata = json.loads(source_file.read_text(encoding="utf-8"))
    archive = PROJECT_DIR / "data" / "raw" / str(metadata["filename"])
    if not archive.exists():
        raise FileNotFoundError(f"Recorded GTFS archive is missing: {archive}")
    return archive, metadata


def table_members(gtfs_zip: ZipFile) -> dict[str, str]:
    """Map GTFS text filenames to their archive member paths."""
    members: dict[str, str] = {}
    for name in gtfs_zip.namelist():
        if not name.endswith("/") and Path(name).suffix.lower() == ".txt":
            filename = Path(name).name
            if filename in members:
                raise RuntimeError(f"Archive contains duplicate GTFS filename: {filename}")
            members[filename] = name
    return members


def read_table(gtfs_zip: ZipFile, member: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read one GTFS CSV table as strings while retaining empty values."""
    with gtfs_zip.open(member) as stream:
        return pd.read_csv(stream, dtype=str, keep_default_na=False, usecols=usecols)


def inspect_table(gtfs_zip: ZipFile, member: str, filename: str) -> dict[str, object]:
    """Count rows, columns, and selected missing values without assuming table size."""
    row_count = 0
    columns: list[str] = []
    missing = {field: 0 for field in IMPORTANT_FIELDS.get(filename, [])}
    with gtfs_zip.open(member) as stream:
        for chunk in pd.read_csv(
            stream, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE
        ):
            if not columns:
                columns = list(chunk.columns)
                missing = {field: 0 for field in missing if field in columns}
            row_count += len(chunk)
            for field in missing:
                missing[field] += int(chunk[field].str.strip().eq("").sum())
    return {
        "filename": filename,
        "row_count": row_count,
        "column_count": len(columns),
        "column_names": columns,
        "missing_values": missing,
    }


def numeric_bounds(frame: pd.DataFrame) -> dict[str, float | None]:
    """Calculate stop-coordinate bounds from valid numeric values."""
    latitudes = pd.to_numeric(frame["stop_lat"], errors="coerce")
    longitudes = pd.to_numeric(frame["stop_lon"], errors="coerce")
    return {
        "minimum_latitude": float(latitudes.min()) if latitudes.notna().any() else None,
        "maximum_latitude": float(latitudes.max()) if latitudes.notna().any() else None,
        "minimum_longitude": float(longitudes.min()) if longitudes.notna().any() else None,
        "maximum_longitude": float(longitudes.max()) if longitudes.notna().any() else None,
    }


def inspect_archive(archive: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build feed-level statistics and a per-file inventory."""
    with ZipFile(archive) as gtfs_zip:
        members = table_members(gtfs_zip)
        inventory = [
            inspect_table(gtfs_zip, members[name], name) for name in sorted(members)
        ]
        inventory_by_name = {str(item["filename"]): item for item in inventory}

        required = ["agency.txt", "routes.txt", "trips.txt", "stops.txt", "stop_times.txt", "shapes.txt"]
        absent = [name for name in required if name not in members]
        if absent:
            raise RuntimeError(f"Required GTFS tables are missing: {', '.join(absent)}")

        agency = read_table(gtfs_zip, members["agency.txt"])
        routes = read_table(gtfs_zip, members["routes.txt"])
        trips = read_table(gtfs_zip, members["trips.txt"])
        stops = read_table(gtfs_zip, members["stops.txt"])
        shapes = read_table(gtfs_zip, members["shapes.txt"], ["shape_id"])

        service_ids = set(trips["service_id"].loc[trips["service_id"].str.strip().ne("")])
        if "calendar.txt" in members:
            calendar = read_table(gtfs_zip, members["calendar.txt"], ["service_id"])
            service_ids.update(calendar["service_id"].loc[calendar["service_id"].str.strip().ne("")])
        if "calendar_dates.txt" in members:
            exceptions = read_table(gtfs_zip, members["calendar_dates.txt"], ["service_id"])
            service_ids.update(exceptions["service_id"].loc[exceptions["service_id"].str.strip().ne("")])

        route_types = sorted(
            routes["route_type"].loc[routes["route_type"].str.strip().ne("")].unique().tolist(),
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )
        missing_values = {
            name: item["missing_values"]
            for name, item in inventory_by_name.items()
            if item["missing_values"]
        }
        summary: dict[str, object] = {
            "archive_filename": archive.name,
            "files_in_archive": sorted(gtfs_zip.namelist()),
            "gtfs_tables": sorted(members),
            "counts": {
                "agencies": len(agency),
                "routes": len(routes),
                "trips": len(trips),
                "unique_shapes": int(shapes["shape_id"].nunique()),
                "stops": len(stops),
                "stop_time_records": inventory_by_name["stop_times.txt"]["row_count"],
                "service_ids": len(service_ids),
            },
            "route_types": route_types,
            "stop_coordinate_bounds": numeric_bounds(stops),
            "missing_values": missing_values,
            "field_population": {
                "route_color_present": "route_color" in routes.columns,
                "route_color_populated_rows": int(routes.get("route_color", pd.Series(dtype=str)).str.strip().ne("").sum()),
                "route_text_color_present": "route_text_color" in routes.columns,
                "route_text_color_populated_rows": int(routes.get("route_text_color", pd.Series(dtype=str)).str.strip().ne("").sum()),
                "direction_id_present": "direction_id" in trips.columns,
                "direction_id_populated_rows": int(trips.get("direction_id", pd.Series(dtype=str)).str.strip().ne("").sum()),
                "shape_dist_traveled_in_shapes": "shape_dist_traveled" in inventory_by_name["shapes.txt"]["column_names"],
                "shape_dist_traveled_in_stop_times": "shape_dist_traveled" in inventory_by_name["stop_times.txt"]["column_names"],
                "feed_info_exists": "feed_info.txt" in members,
            },
        }
        return summary, inventory


def write_reports(summary: dict[str, object], inventory: list[dict[str, object]]) -> None:
    """Save JSON statistics and a compact CSV table inventory."""
    (METADATA_DIR / "gtfs_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (METADATA_DIR / "gtfs_files.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["filename", "row_count", "column_count", "column_names"]
        )
        writer.writeheader()
        for item in inventory:
            writer.writerow(
                {
                    "filename": item["filename"],
                    "row_count": item["row_count"],
                    "column_count": item["column_count"],
                    "column_names": "|".join(item["column_names"]),
                }
            )


def main() -> None:
    """Inspect the recorded archive and report its principal counts."""
    archive, metadata = source_details()
    summary, inventory = inspect_archive(archive)
    write_reports(summary, inventory)
    counts = summary["counts"]
    print(f"Snapshot: {metadata['download_date']} ({archive.name})")
    print(f"GTFS tables: {len(summary['gtfs_tables'])}")
    print(
        "Counts: "
        f"{counts['agencies']:,} agencies, {counts['routes']:,} routes, "
        f"{counts['trips']:,} trips, {counts['unique_shapes']:,} shapes, "
        f"{counts['stops']:,} stops, {counts['stop_time_records']:,} stop times"
    )
    print(f"Route types: {', '.join(summary['route_types'])}")
    print(f"Reports written to {METADATA_DIR}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, BadZipFile, KeyError, pd.errors.ParserError, RuntimeError) as exc:
        raise SystemExit(f"GTFS inspection failed: {exc}") from exc
