"""Validate Phase 1 GIS outputs against the source TransLink GTFS snapshot."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_DIR / "data" / "metadata"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
METRO_BOUNDS = (-124.5, 48.5, -121.0, 50.5)
CHUNK_SIZE = 250_000


def source_details() -> tuple[dict[str, object], Path, Path]:
    """Locate the recorded archive and its date-specific extraction."""
    source_file = METADATA_DIR / "source.json"
    if not source_file.exists():
        raise FileNotFoundError("Missing source.json; run the pipeline from step 1.")
    source = json.loads(source_file.read_text(encoding="utf-8"))
    archive = PROJECT_DIR / "data" / "raw" / str(source["filename"])
    extracted = PROJECT_DIR / "data" / "extracted" / str(source["download_date"])
    return source, archive, extracted


def check(condition: bool, label: str, failures: list[str]) -> None:
    """Print one check result and retain failed labels for the exit status."""
    result = "PASS" if condition else "FAIL"
    print(f"[{result}] {label}")
    if not condition:
        failures.append(label)


def sha256_file(path: Path) -> str:
    """Calculate a file checksum without loading the complete file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extraction_matches_archive(archive: Path, extracted: Path) -> bool:
    """Verify that every extracted file is byte-identical to its ZIP member."""
    with ZipFile(archive) as gtfs_zip:
        members = [member for member in gtfs_zip.infolist() if not member.is_dir()]
        expected_paths = {member.filename for member in members}
        actual_paths = {
            path.relative_to(extracted).as_posix()
            for path in extracted.rglob("*")
            if path.is_file()
        }
        if actual_paths != expected_paths:
            return False
        for member in members:
            path = extracted / member.filename
            digest = hashlib.sha256()
            with gtfs_zip.open(member) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != member.file_size or sha256_file(path) != digest.hexdigest():
                return False
    return True


def inspection_reports_match(
    archive: Path,
    source: dict[str, object],
    extracted: Path,
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    shapes: pd.DataFrame,
    stops: pd.DataFrame,
) -> bool:
    """Recompute the table inventory and principal inspection statistics."""
    summary_path = METADATA_DIR / "gtfs_summary.json"
    inventory_path = METADATA_DIR / "gtfs_files.csv"
    checksum_path = METADATA_DIR / "sha256.txt"
    if not all(path.exists() for path in [summary_path, inventory_path, checksum_path]):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inventory = pd.read_csv(inventory_path, dtype={"filename": str})
    expected_checksum = f"{source['sha256']}  data/raw/{source['filename']}\n"
    if checksum_path.read_text(encoding="utf-8") != expected_checksum:
        return False

    actual_rows: list[dict[str, object]] = []
    with ZipFile(archive) as gtfs_zip:
        members = {
            Path(name).name: name
            for name in gtfs_zip.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() == ".txt"
        }
        for filename, member in sorted(members.items()):
            with gtfs_zip.open(member) as stream:
                columns = list(pd.read_csv(stream, dtype=str, keep_default_na=False, nrows=0).columns)
            row_count = 0
            with gtfs_zip.open(member) as stream:
                for chunk in pd.read_csv(
                    stream, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE
                ):
                    row_count += len(chunk)
            actual_rows.append(
                {
                    "filename": filename,
                    "row_count": row_count,
                    "column_count": len(columns),
                    "column_names": "|".join(columns),
                }
            )
    actual_inventory = pd.DataFrame(actual_rows)
    fields = ["filename", "row_count", "column_count", "column_names"]
    if not inventory[fields].sort_values("filename").reset_index(drop=True).equals(
        actual_inventory[fields].sort_values("filename").reset_index(drop=True)
    ):
        return False

    inventory_counts = actual_inventory.set_index("filename")["row_count"].to_dict()
    service_ids = set(trips["service_id"].loc[trips["service_id"].str.strip().ne("")])
    for filename in ["calendar.txt", "calendar_dates.txt"]:
        path = extracted / filename
        if path.exists():
            table = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["service_id"])
            service_ids.update(table["service_id"].loc[table["service_id"].str.strip().ne("")])
    latitudes = pd.to_numeric(stops["stop_lat"], errors="coerce")
    longitudes = pd.to_numeric(stops["stop_lon"], errors="coerce")
    expected_counts = {
        "agencies": inventory_counts.get("agency.txt"),
        "routes": len(routes),
        "trips": len(trips),
        "unique_shapes": int(shapes["shape_id"].nunique()),
        "stops": len(stops),
        "stop_time_records": inventory_counts.get("stop_times.txt"),
        "service_ids": len(service_ids),
    }
    expected_bounds = {
        "minimum_latitude": float(latitudes.min()),
        "maximum_latitude": float(latitudes.max()),
        "minimum_longitude": float(longitudes.min()),
        "maximum_longitude": float(longitudes.max()),
    }
    route_types = sorted(
        routes["route_type"].loc[routes["route_type"].str.strip().ne("")].unique().tolist(),
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    return bool(
        summary.get("archive_filename") == source["filename"]
        and summary.get("gtfs_tables") == sorted(actual_inventory["filename"].tolist())
        and summary.get("counts") == expected_counts
        and summary.get("route_types") == route_types
        and summary.get("stop_coordinate_bounds") == expected_bounds
    )


def is_epsg_4326(layer: gpd.GeoDataFrame) -> bool:
    """Return whether a layer has an EPSG:4326 coordinate reference system."""
    return layer.crs is not None and layer.crs.to_epsg() == 4326


def coordinates_are_global(layer: gpd.GeoDataFrame) -> bool:
    """Check all geometry bounds against valid longitude and latitude ranges."""
    if layer.empty:
        return False
    min_x, min_y, max_x, max_y = layer.total_bounds
    return -180 <= min_x <= max_x <= 180 and -90 <= min_y <= max_y <= 90


def normalized_text(series: pd.Series) -> pd.Series:
    """Normalize null and textual values for source/output comparisons."""
    return series.fillna("").astype(str)


def add_direction_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized direction used in identifiers."""
    output = frame.copy()
    output["_direction"] = normalized_text(output["direction_id"])
    return output


def pattern_keys(layer: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Create normalized route/shape/direction identifiers."""
    keyed = add_direction_key(layer)
    return set(
        zip(
            normalized_text(keyed["route_id"]),
            normalized_text(keyed["shape_id"]),
            keyed["_direction"],
        )
    )


def expected_pattern_counts(
    trips: pd.DataFrame, excluded_shape_ids: set[str]
) -> pd.DataFrame:
    """Calculate source trip counts for every constructable route pattern."""
    work = trips.loc[
        trips["shape_id"].str.strip().ne("")
        & ~trips["shape_id"].isin(excluded_shape_ids)
    ].copy()
    if "direction_id" not in work.columns:
        work["direction_id"] = ""
    return (
        work.groupby(["route_id", "shape_id", "direction_id"], dropna=False)
        .size()
        .rename("trip_count")
        .reset_index()
    )


def exclusions_are_justified(
    shapes: pd.DataFrame, trips: pd.DataFrame, exclusion_report: dict[str, object]
) -> bool:
    """Confirm every omitted source shape is genuinely unable to form a line."""
    reported = exclusion_report.get("shapes", [])
    if exclusion_report.get("excluded_shape_count") != len(reported):
        return False
    shape_counts = shapes.groupby("shape_id").size().to_dict()
    for item in reported:
        shape_id = str(item.get("shape_id", ""))
        matching_trips = trips.loc[trips["shape_id"].eq(shape_id)]
        points = shapes.loc[shapes["shape_id"].eq(shape_id)]
        coordinates = points[["shape_pt_lon", "shape_pt_lat"]].drop_duplicates()
        cannot_form_line = len(points) < 2 or len(coordinates) < 2
        if not cannot_form_line:
            return False
        if item.get("source_point_count") != shape_counts.get(shape_id, 0):
            return False
        if item.get("trip_count") != len(matching_trips):
            return False
        if item.get("route_ids") != sorted(matching_trips["route_id"].unique().tolist()):
            return False
    return exclusion_report.get("trips_without_shape_id", 0) == int(
        trips["shape_id"].str.strip().eq("").sum()
    )


def pattern_counts_match(patterns: gpd.GeoDataFrame, expected: pd.DataFrame) -> bool:
    """Compare detailed pattern identifiers and trip counts to source trips."""
    actual = add_direction_key(patterns)[
        ["route_id", "shape_id", "_direction", "trip_count"]
    ].copy()
    target = add_direction_key(expected)[
        ["route_id", "shape_id", "_direction", "trip_count"]
    ].copy()
    keys = ["route_id", "shape_id", "_direction"]
    merged = actual.merge(
        target,
        on=keys,
        how="outer",
        suffixes=("_output", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    return bool(
        merged["_merge"].eq("both").all()
        and (
            pd.to_numeric(merged["trip_count_output"], errors="coerce")
            == pd.to_numeric(merged["trip_count_source"], errors="coerce")
        ).all()
    )


def pattern_geometries_match(patterns: gpd.GeoDataFrame, shapes: pd.DataFrame) -> bool:
    """Compare every output line to its exact ordered source shape points."""
    points = shapes.copy()
    for field in ["shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"]:
        points[field] = pd.to_numeric(points[field], errors="coerce")
    if points[["shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"]].isna().any().any():
        return False
    expected_coordinates = {
        str(shape_id): list(
            zip(
                group.sort_values("shape_pt_sequence", kind="stable")["shape_pt_lon"],
                group.sort_values("shape_pt_sequence", kind="stable")["shape_pt_lat"],
            )
        )
        for shape_id, group in points.groupby("shape_id", sort=False)
    }
    for row in patterns.itertuples():
        coordinates = expected_coordinates.get(str(row.shape_id))
        if coordinates is None or len(coordinates) != int(row.shape_point_count):
            return False
        actual = list(row.geometry.coords)
        if len(actual) != len(coordinates):
            return False
        if any(
            abs(actual_x - expected_x) > 1e-9 or abs(actual_y - expected_y) > 1e-9
            for (actual_x, actual_y), (expected_x, expected_y) in zip(actual, coordinates)
        ):
            return False
    return True


def route_attributes_match(layer: pd.DataFrame, routes: pd.DataFrame) -> bool:
    """Confirm retained route fields agree with routes.txt."""
    fields = [
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_color",
        "route_text_color",
    ]
    source = routes.copy()
    for field in fields:
        if field not in source.columns:
            source[field] = ""
    merged = layer[["route_id", *fields]].merge(
        source[["route_id", *fields]],
        on="route_id",
        how="left",
        suffixes=("_output", "_source"),
        validate="many_to_one",
    )
    return all(
        normalized_text(merged[f"{field}_output"]).equals(
            normalized_text(merged[f"{field}_source"])
        )
        for field in fields
    )


def expected_representatives(patterns: gpd.GeoDataFrame) -> pd.DataFrame:
    """Apply the documented route/direction shape ranking rule."""
    ranked = add_direction_key(patterns)
    ranked = ranked.sort_values(
        ["route_id", "_direction", "trip_count", "shape_point_count", "shape_id"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    return ranked.drop_duplicates(["route_id", "_direction"], keep="first")


def representatives_match(
    simplified: gpd.GeoDataFrame, patterns: gpd.GeoDataFrame
) -> bool:
    """Check selected IDs, counts, and geometry against the detailed layer."""
    expected = expected_representatives(patterns)
    actual = add_direction_key(simplified)
    fields = ["route_id", "_direction", "shape_id", "trip_count"]
    if not actual[fields].sort_values(fields).reset_index(drop=True).equals(
        expected[fields].sort_values(fields).reset_index(drop=True)
    ):
        return False
    detailed_geometry = dict(zip(pattern_keys_in_order(patterns), patterns.geometry))
    return all(
        geometry.equals_exact(detailed_geometry[key], tolerance=1e-9)
        for key, geometry in zip(pattern_keys_in_order(actual), actual.geometry)
    )


def pattern_keys_in_order(layer: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Create normalized identifiers while preserving feature order."""
    keyed = add_direction_key(layer)
    return list(
        zip(
            normalized_text(keyed["route_id"]),
            normalized_text(keyed["shape_id"]),
            keyed["_direction"],
        )
    )


def stops_match_source(stops: gpd.GeoDataFrame, source_stops: pd.DataFrame) -> bool:
    """Compare stop coverage, attributes, and geometry coordinates to stops.txt."""
    fields = [
        "stop_code",
        "stop_name",
        "stop_desc",
        "zone_id",
        "location_type",
        "parent_station",
        "wheelchair_boarding",
    ]
    source = source_stops.copy()
    for field in fields:
        if field not in source.columns:
            source[field] = ""
    actual = stops.copy()
    actual["_geometry_lon"] = actual.geometry.x
    actual["_geometry_lat"] = actual.geometry.y
    merged = actual.merge(
        source[["stop_id", "stop_lat", "stop_lon", *fields]],
        on="stop_id",
        how="outer",
        suffixes=("_output", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        return False
    source_lat = pd.to_numeric(merged["stop_lat_source"], errors="coerce")
    source_lon = pd.to_numeric(merged["stop_lon_source"], errors="coerce")
    output_lat = pd.to_numeric(merged["stop_lat_output"], errors="coerce")
    output_lon = pd.to_numeric(merged["stop_lon_output"], errors="coerce")
    numeric_match = (
        (source_lat - output_lat).abs().le(1e-9)
        & (source_lon - output_lon).abs().le(1e-9)
        & (source_lat - merged["_geometry_lat"]).abs().le(1e-9)
        & (source_lon - merged["_geometry_lon"]).abs().le(1e-9)
    ).all()
    text_match = all(
        normalized_text(merged[f"{field}_output"]).equals(
            normalized_text(merged[f"{field}_source"])
        )
        for field in fields
    )
    return bool(numeric_match and text_match)


def unique_stops_by_route(directory: Path, trips: pd.DataFrame) -> dict[str, int]:
    """Recalculate unique served stops from source stop-time relationships."""
    trip_routes = trips.set_index("trip_id")["route_id"].to_dict()
    route_stops: defaultdict[str, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(
        directory / "stop_times.txt",
        dtype=str,
        keep_default_na=False,
        usecols=["trip_id", "stop_id"],
        chunksize=CHUNK_SIZE,
    ):
        chunk["route_id"] = chunk["trip_id"].map(trip_routes)
        if chunk["route_id"].isna().any():
            return {}
        for route_id, group in chunk.groupby("route_id"):
            route_stops[str(route_id)].update(group["stop_id"].loc[group["stop_id"].ne("")])
    return {route_id: len(stop_ids) for route_id, stop_ids in route_stops.items()}


def stop_time_references_are_valid(
    directory: Path, trips: pd.DataFrame, source_stops: pd.DataFrame
) -> bool:
    """Check all stop-time trip and stop foreign keys against source tables."""
    trip_ids = set(trips["trip_id"])
    stop_ids = set(source_stops["stop_id"])
    for chunk in pd.read_csv(
        directory / "stop_times.txt",
        dtype=str,
        keep_default_na=False,
        usecols=["trip_id", "stop_id"],
        chunksize=CHUNK_SIZE,
    ):
        if not set(chunk["trip_id"]) <= trip_ids or not set(chunk["stop_id"]) <= stop_ids:
            return False
    return True


def expected_route_summary(
    directory: Path,
    routes: pd.DataFrame,
    trips: pd.DataFrame,
    simplified: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Recalculate every route-summary source and spatial statistic."""
    source = routes.copy()
    for field in ["route_short_name", "route_long_name", "route_color"]:
        if field not in source.columns:
            source[field] = ""
    base = source[
        ["route_id", "route_short_name", "route_long_name", "route_type", "route_color"]
    ].copy()
    if "direction_id" not in trips.columns:
        trips = trips.assign(direction_id="")
    statistics = trips.groupby("route_id").agg(
        trip_count=("trip_id", "size"),
        unique_shape_count=("shape_id", lambda values: values.loc[values.str.strip().ne("")].nunique()),
        direction_count=("direction_id", lambda values: values.loc[values.str.strip().ne("")].nunique()),
        service_id_count=("service_id", lambda values: values.loc[values.str.strip().ne("")].nunique()),
    ).reset_index()
    stop_counts = unique_stops_by_route(directory, trips)
    statistics["unique_stop_count"] = statistics["route_id"].map(stop_counts).fillna(0).astype(int)
    expected = base.merge(statistics, on="route_id", how="left", validate="one_to_one")
    count_fields = [
        "trip_count",
        "unique_shape_count",
        "direction_count",
        "unique_stop_count",
        "service_id_count",
    ]
    expected[count_fields] = expected[count_fields].fillna(0).astype(int)

    projected = simplified.to_crs("EPSG:26910").copy()
    projected["_length_km"] = projected.geometry.length / 1000
    representatives = []
    for route_id, group in projected.groupby("route_id", sort=False):
        representatives.append(
            {
                "route_id": str(route_id),
                "representative_shape_ids": "|".join(sorted(group["shape_id"].astype(str).unique())),
                "representative_length_km": round(float(group["_length_km"].mean()), 3),
            }
        )
    expected = expected.merge(
        pd.DataFrame(representatives), on="route_id", how="left", validate="one_to_one"
    )
    expected["representative_shape_ids"] = expected["representative_shape_ids"].fillna("")
    return expected


def route_summary_matches(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    """Compare source fields, all counts, selected IDs, and projected lengths."""
    actual = actual.sort_values("route_id").reset_index(drop=True)
    expected = expected.sort_values("route_id").reset_index(drop=True)
    text_fields = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_color",
        "representative_shape_ids",
    ]
    count_fields = [
        "trip_count",
        "unique_shape_count",
        "direction_count",
        "unique_stop_count",
        "service_id_count",
    ]
    if len(actual) != len(expected):
        return False
    if not all(
        normalized_text(actual[field]).equals(normalized_text(expected[field]))
        for field in text_fields
    ):
        return False
    if not all(
        pd.to_numeric(actual[field], errors="coerce").equals(
            pd.to_numeric(expected[field], errors="coerce")
        )
        for field in count_fields
    ):
        return False
    actual_length = pd.to_numeric(actual["representative_length_km"], errors="coerce")
    expected_length = pd.to_numeric(expected["representative_length_km"], errors="coerce")
    return bool(
        (actual_length.isna() == expected_length.isna()).all()
        and (actual_length.fillna(0) - expected_length.fillna(0)).abs().le(0.0005).all()
    )


def main() -> None:
    """Run provenance, geometry, coordinate, and GTFS relationship checks."""
    failures: list[str] = []
    outputs = {
        "route patterns": PROCESSED_DIR / "transit_route_patterns.geojson",
        "simplified routes": PROCESSED_DIR / "transit_routes_simplified.geojson",
        "stops": PROCESSED_DIR / "transit_stops.geojson",
        "route summary": PROCESSED_DIR / "route_summary.csv",
    }
    for label, path in outputs.items():
        check(path.exists() and path.stat().st_size > 0, f"{label} output exists and is non-empty", failures)
    if failures:
        raise SystemExit(f"Validation stopped: {len(failures)} required outputs are missing.")

    source, archive, directory = source_details()
    check(archive.exists(), "recorded raw GTFS archive exists", failures)
    check(directory.exists(), "recorded extracted GTFS directory exists", failures)
    if failures:
        raise SystemExit(f"Validation stopped: {len(failures)} source inputs are missing.")
    check(sha256_file(archive) == source["sha256"], "raw archive SHA-256 matches source metadata", failures)
    check(archive.stat().st_size == source["file_size_bytes"], "raw archive size matches source metadata", failures)
    check(extraction_matches_archive(archive, directory), "extracted files are byte-identical to the raw ZIP", failures)

    routes = pd.read_csv(directory / "routes.txt", dtype=str, keep_default_na=False)
    trips = pd.read_csv(directory / "trips.txt", dtype=str, keep_default_na=False)
    shapes = pd.read_csv(directory / "shapes.txt", dtype=str, keep_default_na=False)
    source_stops = pd.read_csv(directory / "stops.txt", dtype=str, keep_default_na=False)
    exclusions = json.loads(
        (METADATA_DIR / "route_geometry_exclusions.json").read_text(encoding="utf-8")
    )
    excluded_shape_ids = {str(item["shape_id"]) for item in exclusions["shapes"]}
    patterns = gpd.read_file(outputs["route patterns"])
    simplified = gpd.read_file(outputs["simplified routes"])
    stops = gpd.read_file(outputs["stops"])
    summary = pd.read_csv(outputs["route summary"], dtype={"route_id": str})
    check(True, "all GeoJSON files can be opened", failures)
    check(
        inspection_reports_match(archive, source, directory, routes, trips, shapes, source_stops),
        "checksum, GTFS inventory, and inspection summary match the raw archive",
        failures,
    )

    check(not routes["route_id"].duplicated().any(), "source route IDs are unique", failures)
    check(not trips["trip_id"].duplicated().any(), "source trip IDs are unique", failures)
    check(not source_stops["stop_id"].duplicated().any(), "source stop IDs are unique", failures)
    check(set(trips["route_id"]) <= set(routes["route_id"]), "source trip route IDs exist in routes.txt", failures)
    check(
        set(trips["shape_id"].loc[trips["shape_id"].str.strip().ne("")]) <= set(shapes["shape_id"]),
        "source trip shape IDs exist in shapes.txt",
        failures,
    )
    check(
        stop_time_references_are_valid(directory, trips, source_stops),
        "source stop-time trip and stop IDs are valid",
        failures,
    )
    check(
        exclusions_are_justified(shapes, trips, exclusions),
        "route-geometry exclusions are complete and justified by source data",
        failures,
    )

    check(set(patterns.geom_type) == {"LineString"}, "route patterns use LineString geometry", failures)
    check(set(simplified.geom_type) == {"LineString"}, "simplified routes use LineString geometry", failures)
    check(set(stops.geom_type) == {"Point"}, "stops use Point geometry", failures)
    check(is_epsg_4326(patterns), "route patterns CRS is EPSG:4326", failures)
    check(is_epsg_4326(simplified), "simplified routes CRS is EPSG:4326", failures)
    check(is_epsg_4326(stops), "stops CRS is EPSG:4326", failures)

    for label, layer in [
        ("route patterns", patterns),
        ("simplified routes", simplified),
        ("stops", stops),
    ]:
        check(not layer.empty, f"{label} contains features", failures)
        check(not layer.geometry.is_empty.any(), f"{label} has no empty geometry", failures)
        check(not layer.geometry.isna().any(), f"{label} has no missing geometry", failures)
        check(coordinates_are_global(layer), f"{label} coordinates are globally valid", failures)

    check(
        patterns.geometry.map(lambda geometry: len(geometry.coords) >= 2).all(),
        "every route pattern has at least two points",
        failures,
    )
    min_x, min_y, max_x, max_y = stops.total_bounds
    west, south, east, north = METRO_BOUNDS
    check(
        west <= min_x <= max_x <= east and south <= min_y <= max_y <= north,
        "all stop coordinates fall within the plausible Metro Vancouver region",
        failures,
    )

    source_route_ids = set(routes["route_id"])
    source_shape_ids = set(shapes["shape_id"])
    check(set(patterns["route_id"].astype(str)) <= source_route_ids, "pattern route IDs exist in routes.txt", failures)
    check(set(simplified["route_id"].astype(str)) <= source_route_ids, "simplified route IDs exist in routes.txt", failures)
    check(set(patterns["shape_id"].astype(str)) <= source_shape_ids, "pattern shape IDs exist in shapes.txt", failures)
    check(set(simplified["shape_id"].astype(str)) <= source_shape_ids, "simplified shape IDs exist in shapes.txt", failures)

    expected_counts = expected_pattern_counts(trips, excluded_shape_ids)
    check(pattern_counts_match(patterns, expected_counts), "pattern identifiers and trip counts match trips.txt", failures)
    check(pattern_geometries_match(patterns, shapes), "pattern coordinates and point counts match ordered shapes.txt points", failures)
    check(route_attributes_match(patterns, routes), "pattern route attributes match routes.txt", failures)
    check(representatives_match(simplified, patterns), "simplified patterns follow the documented ranking rule", failures)
    check(route_attributes_match(simplified, routes), "simplified route attributes match routes.txt", failures)
    check(stops_match_source(stops, source_stops), "stop features, attributes, and coordinates match stops.txt", failures)

    pattern_duplicates = add_direction_key(patterns).duplicated(["route_id", "shape_id", "_direction"])
    simplified_duplicates = add_direction_key(simplified).duplicated(["route_id", "_direction"])
    check(not pattern_duplicates.any(), "detailed route-pattern identifiers are unique", failures)
    check(not simplified_duplicates.any(), "simplified route/direction identifiers are unique", failures)
    check(not stops["stop_id"].astype(str).duplicated().any(), "processed stop IDs are unique", failures)
    check(not summary["route_id"].astype(str).duplicated().any(), "route summary IDs are unique", failures)
    check(set(summary["route_id"].astype(str)) == source_route_ids, "route summary covers exactly the source routes", failures)
    expected_summary = expected_route_summary(directory, routes, trips, simplified)
    check(route_summary_matches(summary, expected_summary), "all route-summary fields and EPSG:26910 lengths are reproducible", failures)

    print(
        f"\nValidation result: {len(failures)} failure(s), "
        f"{len(failures) == 0 and 'all checks passed' or 'review required'}."
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Validation failed to run: {exc}") from exc
