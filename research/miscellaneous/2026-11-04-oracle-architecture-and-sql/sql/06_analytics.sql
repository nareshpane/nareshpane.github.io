-- All 16 lines survive this analytic query; aggregate GROUP BY product_key
-- would produce two product rows. Window ordering does not sort the result.
SELECT dispense_line_key, product_key, total_cost,
       SUM(total_cost) OVER (PARTITION BY product_key) AS product_known_cost,
       AVG(total_cost) OVER (PARTITION BY product_key) AS product_mean_known_cost,
       COUNT(*) OVER (PARTITION BY product_key) AS product_line_count,
       ROW_NUMBER() OVER (PARTITION BY product_key ORDER BY total_cost DESC NULLS LAST,
                         dispense_line_key) AS deterministic_row_number,
       RANK() OVER (PARTITION BY product_key ORDER BY total_cost DESC NULLS LAST) AS cost_rank,
       DENSE_RANK() OVER (PARTITION BY product_key ORDER BY total_cost DESC NULLS LAST) AS dense_cost_rank,
       PERCENT_RANK() OVER (PARTITION BY product_key ORDER BY total_cost NULLS LAST) AS percent_rank,
       CUME_DIST() OVER (PARTITION BY product_key ORDER BY total_cost NULLS LAST) AS cumulative_distribution,
       NTILE(4) OVER (PARTITION BY product_key ORDER BY total_cost NULLS LAST, dispense_line_key) AS quartile,
       RATIO_TO_REPORT(total_cost) OVER (PARTITION BY product_key) AS known_cost_share
FROM cb_dtl_ft_pharmacy_dispense
ORDER BY product_key, total_cost DESC NULLS LAST, dispense_line_key;
-- RANK gives peers equal rank and leaves gaps; DENSE_RANK does not leave gaps.
-- NULL is explicitly positioned, but remains a ranked row. RATIO_TO_REPORT
-- returns NULL for a NULL expression. Its denominator ignores unknown costs.

SELECT client_key, dispense_line_key, dispensed_at, total_cost,
       LAG(total_cost) OVER (PARTITION BY client_key ORDER BY dispensed_at, dispense_line_key) AS previous_cost,
       LEAD(total_cost) OVER (PARTITION BY client_key ORDER BY dispensed_at, dispense_line_key) AS next_cost,
       FIRST_VALUE(total_cost) OVER (PARTITION BY client_key
         ORDER BY dispensed_at, dispense_line_key
         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_cost,
       LAST_VALUE(total_cost) OVER (PARTITION BY client_key
         ORDER BY dispensed_at, dispense_line_key
         ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_cost,
       SUM(total_cost) OVER (PARTITION BY client_key
         ORDER BY dispensed_at, dispense_line_key
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_known_cost,
       AVG(total_cost) OVER (PARTITION BY client_key
         ORDER BY dispensed_at, dispense_line_key
         ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS trailing_three_line_mean,
       COUNT(*) OVER (PARTITION BY client_key ORDER BY dispensed_at
         RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND CURRENT ROW) AS trailing_30_day_lines
FROM cb_dtl_ft_pharmacy_dispense
ORDER BY client_key, dispensed_at, dispense_line_key;
-- ROWS counts physical rows and requires deterministic order for stable results.
-- RANGE selects a value/time interval and includes all peers at its boundary.
-- The temporal RANGE above has exactly one sort key. LAST_VALUE's explicit
-- full-partition frame avoids the default ending at the current ordering value.

-- A tie demonstration using identical source rows: ROWS advances one line at
-- a time, whereas RANGE includes every peer with the same cost ordering value.
SELECT dispense_line_key, total_cost,
       SUM(total_cost) OVER (ORDER BY total_cost, dispense_line_key
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS physical_running_sum,
       SUM(total_cost) OVER (ORDER BY total_cost
         RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS peer_running_sum
FROM cb_dtl_ft_pharmacy_dispense WHERE total_cost IS NOT NULL
ORDER BY total_cost, dispense_line_key;

-- Analytic results are filtered by an outer query in 19c (there is no QUALIFY).
SELECT * FROM (
  SELECT f.*, ROW_NUMBER() OVER (PARTITION BY client_key
         ORDER BY dispensed_at DESC, dispense_line_key DESC) AS rn
  FROM cb_dtl_ft_pharmacy_dispense f
) WHERE rn = 1 ORDER BY client_key;
