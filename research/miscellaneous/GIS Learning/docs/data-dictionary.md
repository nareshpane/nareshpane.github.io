# Processed Data Dictionary

All GeoJSON layers use **EPSG:4326 / WGS 84**. A blank or null optional value means the source feed did not supply a value; the scripts do not invent replacements.

## `transit_route_patterns.geojson`

One feature represents a unique route/shape/direction combination used by at least one trip and having enough valid source points to form a line. Any unconstructable source shape is reported in `data/metadata/route_geometry_exclusions.json` rather than represented with invented geometry.

| Field | Origin | Meaning |
| --- | --- | --- |
| `route_id` | Source: `routes.txt` / `trips.txt` | GTFS identifier for the route. |
| `route_short_name` | Source: `routes.txt` | Public short route name or number, when supplied. |
| `route_long_name` | Source: `routes.txt` | Public long route name, when supplied. |
| `route_type` | Source: `routes.txt` | GTFS mode code for the route. |
| `route_color` | Source: `routes.txt` | Six-character route colour without `#`, when supplied. |
| `route_text_color` | Source: `routes.txt` | Six-character contrasting label colour without `#`, when supplied. |
| `shape_id` | Source: `trips.txt` / `shapes.txt` | Identifier of the ordered GTFS shape used to build the line. |
| `direction_id` | Source: `trips.txt` | GTFS binary direction indicator when supplied; it distinguishes directions but is not inherently inbound/outbound. |
| `trip_count` | Derived | Number of source trip rows with this route, shape, and direction combination. |
| `shape_point_count` | Derived | Number of source `shapes.txt` points used to construct the line. |
| `geometry` | Derived from `shapes.txt` | `LineString` connecting longitude/latitude points in numeric `shape_pt_sequence` order. |

## `transit_routes_simplified.geojson`

One feature is selected for each route and direction represented in the detailed layer. The script ranks candidate shapes by highest `trip_count`, then greatest `shape_point_count`, then lexicographically smallest `shape_id`. The latter two criteria provide transparent deterministic handling of trip-count ties. Opposite directions remain separate.

| Field | Origin | Meaning |
| --- | --- | --- |
| `route_id` | Source | GTFS route identifier. |
| `route_short_name` | Source | Public short route name or number. |
| `route_long_name` | Source | Public long route name. |
| `route_type` | Source | GTFS mode code. |
| `route_color` | Source | Optional route display colour. |
| `route_text_color` | Source | Optional route label colour. |
| `direction_id` | Source | Direction represented by the selected shape. |
| `shape_id` | Source, selected by script | Shape chosen by the documented ranking rule. |
| `trip_count` | Derived | Number of trips supporting the selected route/shape/direction pattern. |
| `geometry` | Derived | Selected detailed `LineString`; no Phase 1 lossy geometry simplification is applied. |

## `transit_stops.geojson`

Every source `stops.txt` row is retained as a point. `location_type` and `parent_station` preserve the distinction between stations and their child stops or platforms.

| Field | Origin | Meaning |
| --- | --- | --- |
| `stop_id` | Source: `stops.txt` | Unique GTFS stop/location identifier. |
| `stop_code` | Source | Public-facing stop code, when supplied. |
| `stop_name` | Source | Stop, platform, or station name. |
| `stop_desc` | Source | Optional stop description. |
| `stop_lat` | Source, parsed as number | WGS 84 latitude used for geometry. |
| `stop_lon` | Source, parsed as number | WGS 84 longitude used for geometry. |
| `zone_id` | Source | Fare zone identifier, when supplied. |
| `location_type` | Source | GTFS location category; blank normally means a stop/platform (`0`). |
| `parent_station` | Source | Parent station identifier for a child location, when supplied. |
| `wheelchair_boarding` | Source | GTFS accessibility code, when supplied. |
| `geometry` | Derived | WGS 84 `Point` made from `stop_lon` and `stop_lat`. |

## `route_summary.csv`

One row is retained for every row in `routes.txt`, including a route with no current trips if such a source row exists.

| Field | Origin | Meaning |
| --- | --- | --- |
| `route_id` | Source: `routes.txt` | GTFS route identifier. |
| `route_short_name` | Source | Public short route name or number. |
| `route_long_name` | Source | Public long route name. |
| `route_type` | Source | GTFS mode code. |
| `route_color` | Source | Optional route display colour. |
| `trip_count` | Derived from `trips.txt` | Number of scheduled trip records assigned to the route. This is a feed-record count, not a date-expanded service frequency. |
| `unique_shape_count` | Derived from `trips.txt` | Number of distinct nonblank shape IDs used by the route. |
| `direction_count` | Derived from `trips.txt` | Number of distinct populated direction IDs. |
| `unique_stop_count` | Derived from `trips.txt` + `stop_times.txt` | Number of distinct stop IDs visited by any trip on the route. |
| `service_id_count` | Derived from `trips.txt` | Number of distinct populated service IDs used by the route. |
| `representative_shape_ids` | Derived from simplified GeoJSON | Pipe-separated selected shape IDs, usually one per populated direction. |
| `representative_length_km` | Derived spatial measurement | Arithmetic mean of the selected directional shape lengths. Lines are temporarily projected to NAD83 / UTM zone 10N (EPSG:26910), measured in metres, converted to kilometres, and rounded to three decimals. |
