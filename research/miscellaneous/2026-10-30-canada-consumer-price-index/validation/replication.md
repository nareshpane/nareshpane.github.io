# September 2026 CPI validation

Scanned 1,154,918 raw observations; 242 independent release-table comparisons.

{
  "Matched": 143,
  "Within rounding tolerance": 99,
  "Calculated; no release comparator": 44,
  "Approximate": 1,
  "Not independently reproducible": 1
}

## Approximate aggregation

{
  "method": "Eight rounded major-component link-month weights, normalized; April 2026 headline anchor. Within-component basket changes are not undone.",
  "link_month": "2026-04",
  "weights_2025": [
    0.16928307169283072,
    0.282971702829717,
    0.127987201279872,
    0.044995500449955005,
    0.1853814618538146,
    0.05399460053994601,
    0.09809019098090191,
    0.0372962703729627
  ],
  "link_index": 168.0,
  "factor": 1.011484858040922,
  "estimated_index": 169.92945615087487,
  "official_index": 170.0,
  "index_difference": -0.07054384912513001,
  "estimated_yoy": 3.0610282066308336,
  "official_yoy": 3.1,
  "naive_weighted_levels": 173.87726227377263
}

## Limitations

- Level B validates public series extraction, not retail-price construction.
- Core figures match release Table 4 and Bank of Canada HTML; construction is not independently reconstructed.
- Seasonal data are retrieved public results; X-12-ARIMA estimation is not rerun.
- Public index rounding introduces extra ratio uncertainty; intervals shown separately.