# Flood infrastructure

Central Calgary rectangle in WGS 84 order west, south, east, north = -114.135, 51.020, -114.013, 51.077. Roads are 2021 Statistics Canada Road Network File arcs, clipped to the rectangle with identical normalized linework removed. Buildings are City of Calgary Digital Aerial Survey roof-outline polygons. Roof outlines describe observable surfaces. The source does not establish a uniform construction or survey reference date for each footprint.

Each scenario_summary.csv row describes an available open-water return period. road_km and road_pct refer to exposed fragments divided by the total clipped network length. roofs counts clipped roof polygons with a positive-area intersection; roof_m2 sums intersected square metres. threshold_summary.csv records the results as each asset’s minimum exposed fraction changes. The area and road files use the equal-area and local metre-based projections, respectively. All measurements use the detailed input geometry. Provincial hazard designations are separately acquired at display resolution and do not define the boundary of assessed territory.

