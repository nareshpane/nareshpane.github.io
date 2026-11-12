"""Convert ordered GTFS shape points into detailed and representative routes."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
METADATA_DIR = PROJECT_DIR / "data" / "metadata"


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


def read_required_table(directory: Path, filename: str) -> pd.DataFrame:
    """Read a required GTFS table as strings with clear failure messages."""
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Required GTFS table is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def build_shape_geometries(shapes: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Validate point sequences and construct every possible LineString."""
    required = {"shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"}
    missing = required.difference(shapes.columns)
    if missing:
        raise ValueError(f"shapes.txt lacks fields: {', '.join(sorted(missing))}")

    work = shapes[list(required)].copy()
    for field in ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]:
        work[field] = pd.to_numeric(work[field], errors="coerce")
    if work[["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]].isna().any().any():
        raise ValueError("shapes.txt contains non-numeric or missing coordinates/sequences.")
    if work["shape_id"].str.strip().eq("").any():
        raise ValueError("shapes.txt contains a blank shape_id.")

    records: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for shape_id, points in work.groupby("shape_id", sort=False):
        points = points.sort_values("shape_pt_sequence", kind="stable")
        if len(points) < 2:
            exclusions.append(
                {
                    "shape_id": shape_id,
                    "source_point_count": len(points),
                    "reason": "Fewer than two source points; a LineString cannot be constructed.",
                }
            )
            continue
        if points["shape_pt_sequence"].duplicated().any():
            raise ValueError(f"Shape {shape_id!r} has duplicate shape_pt_sequence values.")
        coordinates = list(zip(points["shape_pt_lon"], points["shape_pt_lat"]))
        geometry = LineString(coordinates)
        if geometry.is_empty or geometry.length == 0:
            exclusions.append(
                {
                    "shape_id": shape_id,
                    "source_point_count": len(points),
                    "reason": "The ordered coordinates do not form a nonzero LineString.",
                }
            )
            continue
        records.append(
            {
                "shape_id": shape_id,
                "shape_point_count": len(points),
                "geometry": geometry,
            }
        )
    return pd.DataFrame(records), exclusions


def route_metadata(routes: pd.DataFrame) -> pd.DataFrame:
    """Select the route attributes used in both outputs, filling optional fields."""
    required = {"route_id", "route_type"}
    missing = required.difference(routes.columns)
    if missing:
        raise ValueError(f"routes.txt lacks fields: {', '.join(sorted(missing))}")
    fields = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_color",
        "route_text_color",
    ]
    output = routes.copy()
    for field in fields:
        if field not in output.columns:
            output[field] = ""
    return output[fields]


def build_patterns(
    routes: pd.DataFrame, trips: pd.DataFrame, shape_geometries: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Reduce trips to unique route/shape/direction combinations with counts."""
    required = {"route_id", "trip_id", "shape_id"}
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError(f"trips.txt lacks fields: {', '.join(sorted(missing))}")
    work = trips.copy()
    if "direction_id" not in work.columns:
        work["direction_id"] = ""
    work = work.loc[work["shape_id"].str.strip().ne("")].copy()
    work = work.loc[work["shape_id"].isin(shape_geometries["shape_id"])].copy()
    if work.empty:
        raise ValueError("trips.txt contains no trips with shape_id values.")

    combinations = (
        work.groupby(["route_id", "shape_id", "direction_id"], dropna=False)
        .size()
        .rename("trip_count")
        .reset_index()
    )
    combinations = combinations.merge(
        route_metadata(routes), on="route_id", how="left", validate="many_to_one"
    )
    if combinations["route_type"].isna().any():
        unknown = combinations.loc[combinations["route_type"].isna(), "route_id"].unique()
        raise ValueError(f"Trips refer to unknown route_id values: {', '.join(unknown[:10])}")
    combinations = combinations.merge(
        shape_geometries, on="shape_id", how="left", validate="many_to_one"
    )
    if combinations["geometry"].isna().any():
        unknown = combinations.loc[combinations["geometry"].isna(), "shape_id"].unique()
        raise ValueError(f"Trips refer to unknown shape_id values: {', '.join(unknown[:10])}")

    fields = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_color",
        "route_text_color",
        "shape_id",
        "direction_id",
        "trip_count",
        "shape_point_count",
        "geometry",
    ]
    return gpd.GeoDataFrame(combinations[fields], geometry="geometry", crs="EPSG:4326")


def select_representative_patterns(patterns: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Choose one well-supported shape for every route and direction."""
    ranked = patterns.copy()
    ranked["_direction_key"] = ranked["direction_id"].fillna("")
    ranked = ranked.sort_values(
        ["route_id", "_direction_key", "trip_count", "shape_point_count", "shape_id"],
        ascending=[True, True, False, False, True],
        kind="stable",
    )
    selected = ranked.drop_duplicates(["route_id", "_direction_key"], keep="first").copy()
    selected = selected.drop(columns=["_direction_key", "shape_point_count"])
    fields = [
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "route_color",
        "route_text_color",
        "direction_id",
        "shape_id",
        "trip_count",
        "geometry",
    ]
    return gpd.GeoDataFrame(selected[fields], geometry="geometry", crs="EPSG:4326")


def main() -> None:
    """Build full route patterns and a simplified representative layer."""
    directory = extracted_snapshot()
    routes = read_required_table(directory, "routes.txt")
    trips = read_required_table(directory, "trips.txt")
    shapes = read_required_table(directory, "shapes.txt")
    source_shape_ids = set(shapes["shape_id"])
    referenced_shape_ids = set(trips["shape_id"].loc[trips["shape_id"].str.strip().ne("")])
    unknown_shape_ids = sorted(referenced_shape_ids.difference(source_shape_ids))
    if unknown_shape_ids:
        raise ValueError(
            "trips.txt refers to shape_id values absent from shapes.txt: "
            + ", ".join(unknown_shape_ids[:10])
        )
    shape_geometries, exclusions = build_shape_geometries(shapes)
    patterns = build_patterns(routes, trips, shape_geometries)
    simplified = select_representative_patterns(patterns)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    patterns_path = PROCESSED_DIR / "transit_route_patterns.geojson"
    simplified_path = PROCESSED_DIR / "transit_routes_simplified.geojson"
    patterns.to_file(patterns_path, driver="GeoJSON", index=False)
    simplified.to_file(simplified_path, driver="GeoJSON", index=False)
    for exclusion in exclusions:
        matching_trips = trips.loc[trips["shape_id"].eq(exclusion["shape_id"])]
        exclusion["trip_count"] = len(matching_trips)
        exclusion["route_ids"] = sorted(matching_trips["route_id"].unique().tolist())
    exclusion_report = {
        "excluded_shape_count": len(exclusions),
        "trips_without_shape_id": int(trips["shape_id"].str.strip().eq("").sum()),
        "reason": "GTFS shapes that cannot form valid LineStrings are not emitted as route features.",
        "shapes": exclusions,
    }
    (METADATA_DIR / "route_geometry_exclusions.json").write_text(
        json.dumps(exclusion_report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Detailed route patterns: {len(patterns):,} features -> {patterns_path}")
    print(f"Representative routes: {len(simplified):,} features -> {simplified_path}")
    print(f"Unconstructable source shapes excluded: {len(exclusions):,}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
        raise SystemExit(f"Route construction failed: {exc}") from exc
