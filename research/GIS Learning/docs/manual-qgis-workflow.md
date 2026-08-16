# Initial Manual QGIS Workflow

This is a version-independent outline for reconstructing the raw GTFS feed manually in QGIS without the Python-generated GeoJSON. The completed project used QGIS 3.44.13; tool locations can vary between QGIS versions, so the workflow emphasizes verified operations rather than fragile menu paths.

1. Extract the dated GTFS archive without editing the original ZIP.
2. Load `shapes.txt` as a delimited-text point layer.
3. Set the X field to `shape_pt_lon`.
4. Set the Y field to `shape_pt_lat`.
5. Assign **EPSG:4326 / WGS 84** as the coordinate reference system.
6. Use `shape_id` as the grouping field.
7. Use numeric `shape_pt_sequence` as the ordering field.
8. Run **Points to Path** or the equivalent processing tool to connect each ordered group into its own line.
9. Join the resulting shape lines to `trips.txt` by `shape_id`. Aggregate duplicate trip records first if the goal is one row per route/shape/direction pattern rather than one duplicate geometry per trip.
10. Join route attributes from `routes.txt` by `route_id`.
11. Load `stops.txt` as another delimited-text point layer using `stop_lon`, `stop_lat`, and EPSG:4326. Retain `location_type` and `parent_station` so stations and child stops remain distinguishable.
12. Export the resulting route and stop layers as GeoJSON in EPSG:4326.

Before relying on the result, verify that point sequences were interpreted numerically, each `shape_id` remained separate, route joins did not multiply features unexpectedly, and opposite directions were not concatenated. These checks are easier to automate with the project scripts, which is why the completed QGIS project uses the validated Python outputs as its source layers.
