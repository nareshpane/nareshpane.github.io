-- Oracle 19c aggregate lab: input 16 dispensing lines, including one NULL cost.
-- COUNT(*) = 16, COUNT(total_cost) = 15, SUM(total_cost) = 1610,
-- AVG(total_cost) = 107.333333..., distinct clients = 4, quantity total = 370.
SELECT COUNT(*) AS lines, COUNT(total_cost) AS recorded_costs,
       COUNT(DISTINCT client_key) AS distinct_clients,
       SUM(quantity) AS quantity, SUM(total_cost) AS known_cost,
       AVG(total_cost) AS mean_known_cost, MEDIAN(total_cost) AS median_known_cost,
       MIN(total_cost) AS minimum_known_cost, MAX(total_cost) AS maximum_known_cost,
       STDDEV(total_cost) AS oracle_stddev, STDDEV_SAMP(total_cost) AS sample_sd,
       VAR_POP(total_cost) AS population_variance,
       VAR_SAMP(total_cost) AS sample_variance
FROM cb_dtl_ft_pharmacy_dispense;
-- NULL is unknown, not a free dispensing event. Replacing it with zero changes
-- the estimand: SUM(NVL(total_cost,0))/COUNT(*) = 100.625 rather than 107.3333.

-- WHERE filters rows before grouping; HAVING filters the resulting groups.
SELECT g.district, COUNT(*) AS dispensing_lines,
       COUNT(DISTINCT f.client_key) AS patients, SUM(f.total_cost) AS known_cost
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
WHERE f.dispense_date_key >= 20260101 AND f.dispense_date_key < 20260301
GROUP BY g.district
HAVING COUNT(*) >= 3
ORDER BY g.district;

SELECT product_key,
       LISTAGG(client_id, ', ') WITHIN GROUP (ORDER BY client_id) AS clients
FROM (SELECT DISTINCT f.product_key, c.client_id
      FROM cb_dtl_ft_pharmacy_dispense f
      JOIN cb_dtl_dm_client c ON c.client_key = f.client_key)
GROUP BY product_key ORDER BY product_key;
-- Pre-distinct the patient list; LISTAGG has a bounded VARCHAR2 result and may
-- need ON OVERFLOW TRUNCATE in a larger reporting example.

SELECT CORR(total_cost, quantity) AS pair_correlation,
       REGR_SLOPE(total_cost, quantity) AS slope_cost_on_quantity,
       REGR_INTERCEPT(total_cost, quantity) AS intercept_cost_on_quantity,
       REGR_COUNT(total_cost, quantity) AS complete_pairs
FROM cb_dtl_ft_pharmacy_dispense;
-- Regression discards pairs with either value NULL. These synthetic prices
-- and quantities do not identify causal effects or comparable clinical doses.

SELECT g.province, d.month_label, COUNT(*) AS lines,
       GROUPING(g.province) AS province_is_total,
       GROUPING(d.month_label) AS month_is_total
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
JOIN cb_dtl_dm_date d ON d.date_key = f.dispense_date_key
GROUP BY ROLLUP(g.province, d.month_label)
ORDER BY g.province NULLS LAST, d.month_label NULLS LAST;

SELECT g.province, d.month_label, COUNT(*) AS lines,
       GROUPING_ID(g.province, d.month_label) AS grouping_level
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
JOIN cb_dtl_dm_date d ON d.date_key = f.dispense_date_key
GROUP BY CUBE(g.province, d.month_label)
ORDER BY grouping_level, g.province, d.month_label;

SELECT geography_key, product_key, COUNT(*) AS lines
FROM cb_dtl_ft_pharmacy_dispense
GROUP BY GROUPING SETS ((geography_key), (product_key), ())
ORDER BY geography_key NULLS LAST, product_key NULLS LAST;
-- The grouping functions distinguish a generated subtotal NULL from a data NULL.

-- Fan-out warning: six joined diagnosis rows do not represent six admissions.
SELECT COUNT(*) AS diagnosis_rows, COUNT(DISTINCT h.hospital_stay_key) AS stays,
       SUM(h.length_of_stay_hours) AS inflated_los
FROM cb_dtl_ft_hospital_stay h
JOIN cb_dtl_br_stay_diagnosis b ON b.hospital_stay_key = h.hospital_stay_key;
-- For diagnosis-qualified stays, EXISTS preserves the admission grain.
SELECT COUNT(*) AS stays_with_code
FROM cb_dtl_ft_hospital_stay h
WHERE EXISTS (SELECT 1 FROM cb_dtl_br_stay_diagnosis b
              WHERE b.hospital_stay_key = h.hospital_stay_key AND b.diagnosis_key = 1);
