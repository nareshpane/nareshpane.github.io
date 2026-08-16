# Mapping Metro Vancouver Transit Routes with QGIS

This reproducible learning project demonstrates how Metro Vancouver public-transit routes can be reconstructed from official TransLink GTFS Static data, processed with Python, and prepared for later analysis, styling, and web mapping in QGIS.

## Completed project scope

The reproducible data phase covers:

- downloading and preserving a dated GTFS snapshot;
- inspecting the GTFS tables and recording metadata;
- reconstructing detailed and representative route geometries;
- creating a GIS-ready stop layer;
- producing route-level summary data; and
- validating the processed outputs against the source feed.

The cartographic phase adds a portable QGIS project, rule-based route styling, filtered stop-location layers, progressive labels, a normalized qgis2web/Leaflet export, and a curated screenshot record. The draft long-form tutorial is `../metro-vancouver-transit-qgis.html`; it is not listed in `research.html` until publication is approved.

## Directory structure

```text
GIS Learning/
├── data/
│   ├── raw/          # Immutable downloaded GTFS ZIP snapshots
│   ├── extracted/    # Unmodified ZIP members, grouped by snapshot date
│   ├── processed/    # GIS-ready GeoJSON and route summary CSV
│   └── metadata/     # Provenance, checksum, inventory, and feed statistics
├── docs/             # GTFS, QGIS, reproducibility, and field documentation
├── figures/          # Curated QGIS workflow screenshots and manifest
├── qgis/styles/      # QGIS project and reusable route style
├── scripts/          # Ordered Python preprocessing scripts
└── web/qgis2web/     # Stable Leaflet export with index.html
```

The immutable archive in `data/raw/` is the evidence for the exact source used. Extracted and processed data are deliberately kept separate from it.

## Reproduction

From the repository root, use Python 3.10 or newer on Linux:

```bash
cd "research/GIS Learning"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt

python scripts/01_download_gtfs.py
python scripts/02_inspect_gtfs.py
python scripts/03_build_routes.py
python scripts/04_build_stops.py
python scripts/05_build_route_summary.py
python scripts/06_validate_outputs.py
```

The downloader never overwrites the retained archive. When `source.json` and its recorded ZIP already exist, it verifies and reuses that immutable snapshot, recreating the ignored dated extraction when needed. Then run scripts 02 through 06. See [docs/reproducibility.md](docs/reproducibility.md) for the provenance model.

## Outputs

- `data/processed/transit_route_patterns.geojson`: every meaningful unique route/shape/direction pattern, with the number of trips using it.
- `data/processed/transit_routes_simplified.geojson`: one representative shape per route/direction group, including a blank direction group if a future feed omits `direction_id`, selected reproducibly for a cleaner overview.
- `data/processed/transit_stops.geojson`: all source stop, platform, and station records as points without flattening GTFS location types.
- `data/processed/route_summary.csv`: route metadata, trip/shape/direction/stop/service counts, representative shape IDs, and representative length.
- `data/metadata/source.json`: source URL, timestamp, filename, size, checksum, format, and coverage.
- `data/metadata/gtfs_summary.json`: machine-readable feed statistics and field checks.
- `data/metadata/gtfs_files.csv`: row and column inventory for every GTFS text table.
- `data/metadata/route_geometry_exclusions.json`: source shapes that could not form valid lines, with reasons and trip usage.
- `qgis/metro-vancouver-transit.qgz`: portable QGIS project with relative layer paths.
- `qgis/styles/transit_routes_detailed.qml`: reusable rule-based renderer for the representative route layer.
- `web/qgis2web/index.html`: responsive standalone Leaflet map with TransLink and OpenStreetMap attribution.
- `figures/qgis_screenshots/screenshot_manifest.csv`: visual workflow inventory and accessible image descriptions.

## Coordinate reference systems

GTFS longitude and latitude are WGS 84 coordinates. All exported GeoJSON geometries remain in **EPSG:4326 / WGS 84**, which is suitable for exchange and web maps.

Longitude/latitude degrees are not treated as planar metres. `05_build_route_summary.py` temporarily reprojects selected route geometries to **EPSG:26910 / NAD83 UTM zone 10N**, measures each selected directional shape in metres, converts it to kilometres, and reports the mean selected-direction length for each route. The exported web geometries are not changed by this measurement step.

The 2026-08-15 source contains one HandyDART shape with only one point. Because a valid `LineString` requires at least two points, that shape and its one associated trip cannot become route geometry. The route remains in `route_summary.csv`, and the exclusion is recorded rather than fabricating a line.

## Attribution

The source feed is third-party TransLink data and is not public domain. Read [ATTRIBUTION.md](ATTRIBUTION.md) and [DATA_LICENSE.md](DATA_LICENSE.md) before redistributing or presenting the data.
