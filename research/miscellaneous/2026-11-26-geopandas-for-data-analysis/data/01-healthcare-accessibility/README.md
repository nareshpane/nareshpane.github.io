# Healthcare accessibility

Each row of distances.csv describes one 2021 DA with available population. uid is the census DA identifier; division is its four-digit census division prefix. population is the published count, age65 is a rounded Census Profile count where available and the *_km fields are minimum projected distances to one facility class. The geometry is the official DA locator.

facilities.parquet retains included Statistics Canada provider-listed sites, class and geometry construction. A source facility polygon is represented by an interior locator. Same-class co-located points are deduplicated. candidates.parquet contains 2021 census population-centre interior points. voronoi_* contains clipped planar catchments. Candidate files provide precomputed projected distances in kilometres, in the same row order as the model. The optimization enumerates a single additional location while keeping the baseline listed locations open.

Only classified Alberta facilities enter the relevant class model. The ODHF is a compiled listing and coverage can differ by vintage, data provider and class. Weighted results describe that inventory and 2021 census population assumptions.

