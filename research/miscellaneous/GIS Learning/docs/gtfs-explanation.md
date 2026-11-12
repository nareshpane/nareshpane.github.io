# How the GTFS Tables Fit Together

General Transit Feed Specification (GTFS) Static data is a collection of related text tables. Each file is CSV-formatted even though its filename ends in `.txt`. The tables describe different parts of a transit system and are connected by identifier fields.

## From routes to geometry

```text
routes.txt
    │ route_id
    ▼
trips.txt
    │ shape_id
    ▼
shapes.txt
    ▼
route geometry
```

`routes.txt` describes the public-facing route concept. It commonly contains a short route number or name, a longer name, a GTFS route type, and optional display colours. A row in `routes.txt` does not itself contain a line geometry.

`trips.txt` describes scheduled journeys. Every trip refers to a `route_id` and `service_id`; it can also refer to a `shape_id` and identify a `direction_id`. This table is the bridge between the public-facing route and the path travelled by a particular service pattern.

`shapes.txt` stores geometry as ordered points. Rows with the same `shape_id` belong to one path. `shape_pt_sequence` gives their order, while `shape_pt_lon` and `shape_pt_lat` supply WGS 84 coordinates. Sorting the points by sequence and connecting them creates a GIS `LineString`.

GTFS uses ordered points instead of prebuilt GIS geometries because GTFS is a tabular interchange format that can be consumed without specialized spatial file support. The point representation also gives schedule and mapping tools an explicit travel order.

## Why one route can have many shapes

A route is a service identity, not necessarily one unchanging path. Its trips may run in opposite directions, begin or end early, use branches or alternate terminals, or follow service-specific variations. Each variation can have its own `shape_id`. It would therefore be incorrect to choose an arbitrary shape and call it the complete route, or to concatenate unrelated shapes into one malformed line.

The detailed output in this project preserves unique route/shape/direction combinations and counts the trips supporting each combination. The separate simplified output chooses one representative shape within each route and direction for an overview map.

## From trips to stops

```text
trips.txt
     │ trip_id
     ▼
stop_times.txt
     │ stop_id
     ▼
stops.txt
```

`stop_times.txt` is the ordered schedule for each trip. A row links a `trip_id` to a `stop_id`, records the stop sequence, and normally includes arrival and departure times. Joining it to `trips.txt` tells us which routes serve each stop.

`stops.txt` contains stop or station names and coordinates. GTFS can distinguish stops/platforms, stations, entrances/exits, generic nodes, and boarding areas through `location_type`; `parent_station` can connect a child location to a larger station. The processed stop layer preserves these distinctions rather than forcing every row into the same conceptual category.

## Service calendars

`calendar.txt` defines regular weekly service patterns and their date ranges. `calendar_dates.txt` records date-specific additions and removals, such as holiday exceptions. Both use `service_id`, which connects to `trips.txt`.

Some feeds describe regular service in `calendar.txt`, exceptions in `calendar_dates.txt`, or a combination of both. Scripts should discover optional tables rather than assume both are always present. A fully date-specific analysis must apply the regular calendar and exceptions together; Phase 1 reports service-ID counts but does not expand service into daily timetables.
