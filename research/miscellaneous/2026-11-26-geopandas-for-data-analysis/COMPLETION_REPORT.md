# GeoPandas in Alberta

Completion report · November 25, 2026

## Deliverables

- The research HTML is one directory above this project. Its asset directory contains authored styles and JavaScript, vendored Leaflet, four data dictionaries, Python stages and a pipeline driver, validation tests, Matplotlib figures, downloadable GIS and tabular results, and source metadata.
- The opening atlas layers actual Alberta geography, census areas, facilities, wildfire perimeters, flood designations and roads. The study includes interactive predicate, spatial-index, flood-probability and graph demonstrations. The four applications include threshold controls, scenario views, a wildfire chronology and a linked Moran scatterplot.
- The download panels link processed files, each analysis script, the tested environment and the provenance manifests. Full raw inputs live in the ignored acquisition cache. Detailed yearly wildfire footprints are split into single-year GeoParquet files. The overview is generalized for download and display.

## Datasets and study coverage

| Input | Period and study coverage | Source |
|---|---|---|
| Alberta census boundaries, population centres, Census Profile, census counts and representative points | 2021 Alberta geography and Census, with 2020 household income | [Alberta boundaries](https://open.canada.ca/data/en/dataset/1e35d18f-cfa1-4bfe-b3ef-6ac8e18960e1), [2021 Census Profile](https://www150.statcan.gc.ca/n1/en/catalogue/98-401-X2021006), [Geographic Attribute File](https://www150.statcan.gc.ca/n1/en/catalogue/92-151-X2021001) |
| Open Database of Healthcare Facilities | September 2026 version 2.0 and official 2020 version 1.1 hospital records in Alberta. Facility source observations have varied dates. | [ODHF v2.0](https://www150.statcan.gc.ca/n1/pub/13-26-0001/132600012020001-eng.htm), [official v1.1 map service](https://maps-cartes.services.geo.ca/server2_serveur2/rest/services/StatCan/OpenDatabaseHealthFacilities/MapServer) |
| Historical wildfire records and perimeters | Provincial incident records 2006–2025 and the single verified rolling perimeter series 2016–2025 | [Alberta wildfire data](https://www.alberta.ca/wildfire-maps-and-data) |
| FHIMP, 2021 Road Network File and Calgary roof outlines | 20, 50, 100, 200, 500-year published open-water scenarios intersect the selected central Calgary rectangle. Road geometry is from 2021 and roof observations have varying or unspecified vintages. Provincial designated hazards provide geographic context. | [FHIMP](https://www.alberta.ca/flood-hazard-identification-program), [2021 Road Network File](https://geo.statcan.gc.ca/geo_wa/rest/services/2021/Road_Network_File/MapServer), [Calgary roof outlines](https://data.calgary.ca/Base-Maps/Buildings/uc4c-6kbd) |

Automated acquisition succeeded for the inputs used in all four applications. The Statistics Canada bulk ZIP endpoints for the Profile and Geographic Attribute File returned a security verification response. Public redistributions carried the same census data through Data for Canada / Source Cooperative and MountainMath. The linked acquisition guide gives the optional original ZIP routes, exact cache filenames and replay command. [Acquisition guide](DATA_ACQUISITION.md) · [Source catalogue and licences](metadata/data_sources.json) · [URLs, SHA-256 hashes and service queries](metadata/acquisition_manifest.json).

## Executed analyses and findings

- **Shared geography.** The 6,203 Alberta DA geometries include 21 with unavailable counts. The joined population reconciles to 4,262,635. All 6,203 DA locators come from the Geographic Attribute File. 10 DA geometries required repair. [Shared summary](results/shared.json).
- **Healthcare accessibility.** The 2026 hospital class yields 21 listed locations, a 34.2 km population-weighted mean and 722,398 represented residents beyond 50 km. The 2020 hospital class yields 127 locations and a 7.7 km mean. Across 120 population-centre candidates, the best single addition for mean distance changes from Fort McMurray in v2.0 to Airdrie in v1.1. The v2.0 maximum-coverage objective at 50 km selects a candidate that places 3,618,733 residents within the threshold. Both scenarios retain existing listed sites. Calculations include indexed nearest joins, weighted quantiles, coverage, clipped Voronoi cells, distance buffers and exhaustive candidate search. [Healthcare summary](results/healthcare.json).
- **Wildfire exposure.** The incident archive yielded 27,828 unique year and fire-number keys. The largest unioned footprint year was 2023 with 34,281 km². The unique footprint across the verified decade covered 64,409 km². The sum of annual mapped areas was 65,149 km². The perimeter calculation excluded 13,479 records coded as prescribed burns. The 25 and 50 km grids estimate cell fractions and recurrence. The application also computes population locator proximity and bandwidth-sensitive ignition intensity. [Wildfire summary](results/wildfire.json).
- **Flood infrastructure.** The 1-in-100 open-water footprint intersects 39.50 of 696.12 clipped road km (5.7%). It intersects 2,653 mapped roof outlines and 471,692 m² of their clipped area. Exact line–polygon and polygon–polygon overlays use a tiled footprint to limit candidate work. Original fragments are dissolved per asset before length and area are measured. The map shows alternative screening thresholds and hypothetical affected fragments. [Flood summary](results/flood.json).
- **Socioeconomic geography.** Queen-contiguity Moran’s I for unemployment was 0.235 (global permutation p = 0.002). 831 areas met the raw local significance cutoff and 0 met the Benjamini–Hochberg cutoff. Median household income had Moran’s I of 0.622 and 494 adjusted-significant local areas. The project compares queen, rook and eight-nearest-neighbour weights with 499 permutations per test and provides a complete-case weighted composite. [Statistical summary](results/socioeconomic.json).

## Performance and verification

- The build used Python 3.13.1 on Windows with GeoPandas 1.2.0, Shapely 2.2.0, pyproj 3.8.0, pyogrio 0.13.0, SciPy 1.18.1, libpysal 4.15.1, Matplotlib 3.11.2, PyArrow 22.0.0 and NetworkX 3.7. The hardware reported an 11th Gen Intel Core i5-1135G7. [Tested requirements](requirements.txt) · [Processing manifest](metadata/processing_manifest.json).
- The benchmark compared 17,850 potential naïve pairs with 1,552 index envelope candidates. Median query time was 14.47 ms for vectorized pairwise tests and 6.22 ms for indexed exact search, excluding index construction. The benchmark also records build time for the tree and timings for reading identical data in GeoJSON, GeoPackage and GeoParquet. [Benchmark](results/benchmark.json).
- The data pipeline and eight known-solution tests executed. Validation reconciled the population, checked fractions and wildfire unions, verified reproducible statistics and inspected 65 vector map assets for geometry validity, emptiness and CRS. [Data validation](metadata/validation_report.json).
- Playwright used Chromium 158.0.8127.37 at 1,440, 1,920, 1,024, 768, 390 px. Checks covered map layers and zoom, timelines and sliders, linked statistics, code expansion and copying, resources, downloads, mobile navigation, console errors and horizontal page overflow. Screenshots are in tests/screenshots. [Browser validation](metadata/browser_validation.json).

## Analytical limits and implementation decisions

- Healthcare results are straight-line measures of the selected inventories and 2021 DA representation. Source inventory coverage, cross-border facilities, site capacity and road access matter for a planning decision.
- The fire history measures mapped wildfires and recorded incidents, with differing observation coverage. Older perimeters are available in a separate provincial archive. The empirical series uses one comparable rolling service and its verified years. Mapped unions and tabulated incident-size estimates measure different source coverage.
- Calgary was selected because the official inundation layers, federal roads and municipal roof outlines overlap there. Scenario coverage varies by source study. The road data were used for spatial exposure. The province-wide context retains 31,179 of 42,792 service records as polygons after display generalization and repair. The graph demonstration uses illustrative routing because grade separation and turn connectivity were not validated for the empirical road arcs.
- Socioeconomic indicators have distinct universes and suppression patterns. Local estimates and adjusted significance vary by weights and test resolution. The composite is an educational weighted score.
- The initial build identified the GPT‑6 family. The presentation revision displays the four project metadata values supplied in the refinement brief.

The project was saved locally. Git operations were limited to inspection.

