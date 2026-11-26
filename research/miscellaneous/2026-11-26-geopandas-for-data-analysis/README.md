# GeoPandas in Alberta

The companion HTML page is one directory above this project. The local-only build was developed against Python 3.13.1 on Windows. The pinned versions are in requirements.txt and the measured runtime environment is in metadata/processing_manifest.json.

## Run from this directory

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/run_all.py
python -m unittest discover -s tests -v
python tests/serve_preview.py 8765
~~~

The preview prints its research page URL. It binds to loopback and serves the project HTML and its public assets. With the preview running, execute python tests/browser_check.py in a second terminal to repeat browser validation. The test uses installed Chrome on Windows or Playwright Chromium when available. Run scripts/run_all.py with --offline to reproduce from frozen local raw caches. The online build reuses those caches. On a new download, the ten rolling wildfire layers may contain different calendar years; the acquisition manifest and output verify the records loaded. The 2026 build used 2016–2025 perimeters. To reconstruct this vintage after the service rolls, consult DATA_ACQUISITION.md and the existing manifest.

## Analyses

| Directory | Data and outputs |
|---|---|
| data/shared | 2021 DA geometry, census counts, official DA locators, population centres and Alberta GeoPackage |
| data/01-healthcare-accessibility | Facility coordinates, nearest distances, catchments, candidate positions and scenario inputs |
| data/02-wildfire-exposure | Annual footprint unions, proximity, ignition events and hexagonal summaries |
| data/03-flood-infrastructure | Central Calgary study features, scenario polygons, intersection pieces and tabular exposure |
| data/04-socioeconomic-geography | Joined census data, fitted spatial statistics and census indicator dictionary |
| maps | WGS 84 display layers, interactive model arrays and local Moran data |
| figures | Matplotlib maps generated from the actual analytical layers |
| results | Computed summaries, timing and the GeoDataFrame listing used on the page |
| metadata | Sources, checksums, build environment and invariant validation |

See each application’s README for units, record grain and extraction limits. Files under data/*/raw are cached locally and ignored by Git. The sources provide download links for the original releases. The map uses vendored Leaflet under BSD-2-Clause. Cartography and derived datasets cite their public data providers on the page and in metadata/data_sources.json.


