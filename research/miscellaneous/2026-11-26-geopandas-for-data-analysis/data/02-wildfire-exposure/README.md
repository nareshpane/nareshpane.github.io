# Wildfire exposure

annual.csv uses the incident year, unique non-missing YEAR/FIRE_NUMBER record count and square kilometres of unioned wildfire footprint for acquired perimeter years. The ingredient reports distinguish incident records from geometrically mapped footprints. Wildfire codes B and B_* are eligible for the perimeter area calculations. Prescribed burns PB and PB_* are excluded from those measurements. Web assets show the 2016–2025 rolling service vintage.

footprint_YYYY.parquet files contain detailed yearly unioned geometry. annual_footprints.parquet is the 125 m simplified overview and stores the detailed measured area as a separate field. Both use a NAD83 Albers equal-area CRS. grid_25.parquet and grid_50.parquet contain clipped hexagons, cell land-and-water polygon area km2, yYYYY fractions, unique period fraction and frequency_0, frequency_1 and frequency_5 calculated across the observed years. Fragments under one square kilometre are excluded. population_proximity.parquet stores kilometres from 2021 populated DA locators to the period’s footprint. ignitions.parquet contains in-boundary usable ignition points. Smoothed intensity uses a 10 km grid and Gaussian bandwidths of 20, 50 or 100 km. It describes the input catalogue’s events per square kilometre over 2006–2025.


