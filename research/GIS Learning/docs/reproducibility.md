# Reproducibility and Provenance

## Snapshot model

`01_download_gtfs.py` downloads the official URL into a temporary file and streams the received bytes unchanged into a dated ZIP filename. It computes SHA-256 over those exact bytes, records the byte count and local timestamp, then moves the completed file into `data/raw/`. It does not normalize, edit, or recompress the archive.

The script refuses to overwrite an existing dated ZIP or a populated dated extraction directory. This protects the source evidence from accidental replacement. The ZIP members are copied separately to `data/extracted/<download-date>/`; all generated datasets go to `data/processed/`.

`data/metadata/source.json` identifies the source, snapshot date, local download timestamp, filename, size, format, coverage, and SHA-256. `sha256.txt` provides the checksum in a conventional checksum-file format.

## Pipeline order

Run the numbered scripts in order. Each later script reads the snapshot date from `source.json`, so processing does not depend on the current working directory or on manually typing a date. All project paths are derived with `pathlib.Path` from each script's location, including when the directory name contains a space.

The inspection script reads the original ZIP for its inventory and measurements. Geometry scripts read its separately extracted members. Validation compares processed identifiers with the extracted source tables and verifies geometry, CRS, coordinates, uniqueness, and coverage.

## Reproducing a retained snapshot

If the repository already contains the raw ZIP and matching metadata, run the downloader step normally. It verifies and reuses the retained snapshot, and recreates the ignored dated extraction if it is absent. It never overwrites the archive. You can also verify the archive directly from this directory:

```bash
sha256sum -c data/metadata/sha256.txt --ignore-missing
```

Then run scripts 02 through 06. To acquire a future snapshot intentionally, archive the current active metadata and outputs first; `source.json` describes one active processing snapshot at a time.

## Deterministic processing decisions

- Detailed routes are unique `route_id`/`shape_id`/`direction_id` combinations with trip counts.
- Representative routes are ranked independently within each route and direction by descending trip count, then descending source shape-point count, then ascending `shape_id` as a deterministic final tie-break.
- Route length is the mean length of the selected directional representatives, measured after reprojection to EPSG:26910 and rounded to three decimal places.
- No geometry simplification is applied in Phase 1.

The retained 2026-08-15 feed has one exceptional HandyDART shape (`shape_id` `4484`) containing a single point and used by one trip. It cannot mathematically produce a `LineString`, so the route builder excludes it from the two geometry layers and records the decision in `data/metadata/route_geometry_exclusions.json`. The HandyDART route is still retained in the route summary. This is a source-data limitation, not a silent cleanup.

Package versions are constrained to compatible major-version ranges in `requirements.txt`. Exact environment locking can be added later if a concrete version-specific issue is discovered; the project avoids freezing unrelated transitive packages.
