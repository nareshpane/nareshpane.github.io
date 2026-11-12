"""Create route-level GTFS statistics and projected representative lengths."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_DIR / "data" / "metadata"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CHUNK_SIZE = 250_000


def extracted_snapshot() -> Path:
    """Return the extracted directory identified by source metadata."""
    source_file = METADATA_DIR / "source.json"
    if not source_file.exists():
        raise FileNotFoundError("Run scripts/01_download_gtfs.py first.")
    source = json.loads(source_file.read_text(encoding="utf-8"))
    directory = PROJECT_DIR / "data" / "extracted" / str(source["download_date"])
    if not directory.exists():
        raise FileNotFoundError(f"Extracted GTFS directory is missing: {directory}")
    return directory


def read_table(directory: Path, filename: str) -> pd.DataFrame:
    """Read a required GTFS table as strings."""
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Required GTFS table is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def stops_by_route(
    directory: Path, trips: pd.DataFrame, source_stop_ids: set[str]
) -> dict[str, int]:
    """Count distinct served stop IDs by mapping stop-time trips to routes."""
    stop_times_path = directory / "stop_times.txt"
    if not stop_times_path.exists():
        raise FileNotFoundError(f"Required GTFS table is missing: {stop_times_path}")
    trip_routes = trips.set_index("trip_id")["route_id"].to_dict()
    route_stops: defaultdict[str, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(
        stop_times_path,
        dtype=str,
        keep_default_na=False,
        usecols=["trip_id", "stop_id"],
        chunksize=CHUNK_SIZE,
    ):
        chunk["route_id"] = chunk["trip_id"].map(trip_routes)
        if chunk["route_id"].isna().any():
            raise ValueError("stop_times.txt refers to trip_id values absent from trips.txt.")
        unknown_stops = set(chunk["stop_id"].loc[chunk["stop_id"].ne("")]).difference(
            source_stop_ids
        )
        if unknown_stops:
            raise ValueError(
                "stop_times.txt refers to stop_id values absent from stops.txt: "
                + ", ".join(sorted(unknown_stops)[:10])
            )
        for route_id, group in chunk.groupby("route_id"):
            route_stops[str(route_id)].update(group["stop_id"].loc[group["stop_id"].ne("")])
    return {route_id: len(stop_ids) for route_id, stop_ids in route_stops.items()}


def representative_statistics() -> pd.DataFrame:
    """Summarize selected shape IDs and mean directional length in EPSG:26910."""
    path = PROCESSED_DIR / "transit_routes_simplified.geojson"
    if not path.exists():
        raise FileNotFoundError("Run scripts/03_build_routes.py before route summary.")
    selected = gpd.read_file(path)
    if selected.crs is None:
        raise ValueError("Simplified route layer has no CRS.")
    selected = selected.to_crs("EPSG:26910")
    selected["_length_km"] = selected.geometry.length / 1000
    records = []
    for route_id, group in selected.groupby("route_id", sort=False):
        shape_ids = sorted(group["shape_id"].astype(str).unique())
        records.append(
            {
                "route_id": str(route_id),
                "representative_shape_ids": "|".join(shape_ids),
                "representative_length_km": round(float(group["_length_km"].mean()), 3),
            }
        )
    return pd.DataFrame(records)


def build_summary(directory: Path) -> pd.DataFrame:
    """Join route, trip, stop, service, and representative geometry statistics."""
    routes = read_table(directory, "routes.txt")
    trips = read_table(directory, "trips.txt")
    stops = read_table(directory, "stops.txt")
    required_routes = {"route_id", "route_type"}
    required_trips = {"route_id", "trip_id", "shape_id", "service_id"}
    if missing := required_routes.difference(routes.columns):
        raise ValueError(f"routes.txt lacks fields: {', '.join(sorted(missing))}")
    if missing := required_trips.difference(trips.columns):
        raise ValueError(f"trips.txt lacks fields: {', '.join(sorted(missing))}")
    if "stop_id" not in stops.columns:
        raise ValueError("stops.txt lacks field: stop_id")
    if routes["route_id"].duplicated().any():
        raise ValueError("routes.txt contains duplicate route_id values.")
    if trips["trip_id"].duplicated().any():
        raise ValueError("trips.txt contains duplicate trip_id values.")
    if stops["stop_id"].duplicated().any():
        raise ValueError("stops.txt contains duplicate stop_id values.")
    unknown_routes = set(trips["route_id"]).difference(routes["route_id"])
    if unknown_routes:
        raise ValueError(
            "trips.txt refers to route_id values absent from routes.txt: "
            + ", ".join(sorted(unknown_routes)[:10])
        )
    if "direction_id" not in trips.columns:
        trips["direction_id"] = ""

    for field in ["route_short_name", "route_long_name", "route_color"]:
        if field not in routes.columns:
            routes[field] = ""
    base = routes[
        ["route_id", "route_short_name", "route_long_name", "route_type", "route_color"]
    ].copy()
    grouped = trips.groupby("route_id")
    statistics = grouped.agg(
        trip_count=("trip_id", "size"),
        unique_shape_count=("shape_id", lambda values: values.loc[values.str.strip().ne("")].nunique()),
        direction_count=("direction_id", lambda values: values.loc[values.str.strip().ne("")].nunique()),
        service_id_count=("service_id", lambda values: values.loc[values.str.strip().ne("")].nunique()),
    ).reset_index()
    stop_counts = stops_by_route(directory, trips, set(stops["stop_id"]))
    statistics["unique_stop_count"] = statistics["route_id"].map(stop_counts).fillna(0).astype(int)

    summary = base.merge(statistics, on="route_id", how="left", validate="one_to_one")
    count_fields = [
        "trip_count",
        "unique_shape_count",
        "direction_count",
        "unique_stop_count",
        "service_id_count",
    ]
    summary[count_fields] = summary[count_fields].fillna(0).astype(int)
    summary = summary.merge(
        representative_statistics(), on="route_id", how="left", validate="one_to_one"
    )
    summary["representative_shape_ids"] = summary["representative_shape_ids"].fillna("")
    fields = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_color",
        "trip_count",
        "unique_shape_count",
        "direction_count",
        "unique_stop_count",
        "service_id_count",
        "representative_shape_ids",
        "representative_length_km",
    ]
    return summary[fields]


def main() -> None:
    """Write one descriptive row for every route in routes.txt."""
    summary = build_summary(extracted_snapshot())
    output = PROCESSED_DIR / "route_summary.csv"
    summary.to_csv(output, index=False)
    print(f"Route summary: {len(summary):,} records -> {output}")
    print("Representative lengths use mean selected-direction length in EPSG:26910.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Route summary construction failed: {exc}") from exc
