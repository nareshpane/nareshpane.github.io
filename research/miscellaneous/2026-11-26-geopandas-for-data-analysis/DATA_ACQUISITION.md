# Data acquisition

Run python scripts/00_download_data.py from this project directory. To retry individual sources, pass their names as arguments. The manifest records original URLs, retrieval timestamps, observed byte sizes, SHA-256 hashes, layer metadata and query parameters. Cached data is validated before reuse. ArcGIS query results are stored by layer, with object ID checks across pages. Raw caches are excluded from version control.

## Successfully used distributions

| Source | Automated route and local name |
|---|---|
| Alberta Census Boundaries – Current | Alberta’s original shapefile ZIP as data/shared/raw/Alberta_Census_Boundaries_SHP.zip |
| Statistics Canada census counts | Table 98-10-0015-01 original CSV archive as data/shared/raw/98100015-eng.zip |
| Geographic Attribute File | MountainMath public Parquet representation of the 2021 Statistics Canada Geographic Attribute File as data/shared/raw/geosuite_da_2021.parquet |
| Statistics Canada population centres | 2021 cartographic boundary service, PC layer, Alberta and near-border query as data/shared/raw/population_centres.geojson |
| Statistics Canada healthcare | ODHF v2 original zipped GeoPackage as data/01-healthcare-accessibility/raw/odhf_v2_gpkg.zip |
| Statistics Canada healthcare sensitivity | Official 2020 hospital points under data/01-healthcare-accessibility/raw/hospital_2020.geojson |
| Historical wildfire record | Alberta’s 2006–2025 incident CSV and all ten rolling Alberta FeatureServer perimeter layers under data/02-wildfire-exposure/raw |
| Alberta flood mapping | Open Water scenario service layers intersecting the central Calgary query extent, and a province-wide generalized map of designated flood hazard polygons under data/03-flood-infrastructure/raw |
| Road network | Statistics Canada 2021 Road Network File, all five cartographic ranks in the Calgary envelope under data/03-flood-infrastructure/raw |
| Roof outlines | City of Calgary Digital Aerial Survey polygons intersecting the project envelope as data/03-flood-infrastructure/raw/buildings.geojson |
| Census Profile | Statistics Canada DA profile redistributed as wide Parquet by Data for Canada on Source Cooperative as data/04-socioeconomic-geography/raw/census_profile.parquet |

## Optional manual routes

The analyses use automatically acquired data. The Statistics Canada bulk Census Profile and Geographic Attribute File ZIP endpoints returned HTTP 403 with a security-verification page during programmatic acquisition. These routes let you use the original ZIPs if the mirrors become unavailable.

| Dataset | Official download and selection | Local placement |
|---|---|---|
| 2021 Census Profile, 98-401-X2021006 | [Census Profile downloads](https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger.cfm?Lang=E), Comprehensive download files, Canada/provinces/CDs/CSDs/DAs — Prairies only, CSV. Direct distribution [GetFile.cfm](https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger/comp/GetFile.cfm?FILETYPE=CSV&GEONO=006_Prairies&Lang=E). | Put the ZIP in data/04-socioeconomic-geography/raw/98-401-X2021006_eng_CSV_Prairies.zip. The script accepts an English Census Profile archive containing English_CSV_data members. |
| 2021 Geographic Attribute File, 92-151-X2021001 | [Catalog](https://www150.statcan.gc.ca/n1/en/catalogue/92-151-X2021001), [official 2021 ZIP](https://www12.statcan.gc.ca/census-recensement/2021/geo/aip-pia/attribute-attribs/files-fichiers/2021_92-151_X.zip). | Put 2021_92-151_X.zip in data/shared/raw. |

After adding either ZIP, run python scripts/run_all.py --offline. The build uses the ZIP in preference to the corresponding mirror. Full original province-wide flood data are also available through Alberta’s [published geodatabase ZIP](https://extranet.gov.ab.ca/srd/geodiscover/srd_pub/inlandWaters/ABFloodMapping.zip); the page uses the documented REST layers instead. Alberta’s [historical wildfire archive](https://www.alberta.ca/wildfire-maps-and-data) is a separate geospatial release with older years. The empirical chronology is based on comparable layers from one verified rolling service. Keep the frozen raw service files and acquisition manifest for exact data-vintage replay.



