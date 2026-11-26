# Shared geography

2021 Census DAs are matched to Table 98-10-0015-01 by uid, the trailing eight digits of the Alberta DA DGUID. population counts persons under the 2021 census reference, land_km2 is published land area and density is persons per square kilometre. Some DA geometry rows have absent count records. The joined count sum is checked against Alberta’s published total. Demand geometries use official population-weighted DA representative points from the Geographic Attribute File. Candidate centres use the 2021 cartographic population-centre boundaries and interior points.

The Alberta GeoPackage contains the union of the current Alberta DA boundary file in EPSG:3400. Input geometry repairs and unmatched DAs are recorded in the shared result and validation report. The province geometry includes water insofar as it appears in the source boundary. Published Census land area is distinct from the analysis polygon area.

