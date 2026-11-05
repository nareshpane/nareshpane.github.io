-- Oracle 19c analyst exercises. Outputs and assumptions: ../data/exercises.json

-- 1. Count dispensing lines by province and month
-- Grain: One province-month
-- Filter: January through March 2026; dispensing date in Nepal local calendar
SELECT g.province, d.month_label, COUNT(*) AS dispensing_lines
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_date d ON d.date_key = f.dispense_date_key
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
WHERE d.calendar_date >= DATE '2026-01-01' AND d.calendar_date < DATE '2026-04-01'
GROUP BY g.province, d.month_label ORDER BY g.province, d.month_label;

-- 2. Count distinct recipients of product A
-- Grain: One selected product
-- Filter: Product code SYN-A; all fixture dates
SELECT p.product_code, COUNT(*) AS dispensing_lines,
       COUNT(DISTINCT f.client_key) AS distinct_clients
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_health_product p ON p.product_key = f.product_key
WHERE p.product_code = 'SYN-A'
GROUP BY p.product_code;

-- 3. Average exact hospitalization duration by facility
-- Grain: One facility
-- Filter: Completed stays; all fixture dates
SELECT f.facility_name, COUNT(*) AS stays,
       ROUND(AVG(h.computed_los_hours), 6) AS mean_los_hours
FROM hdn_v_hospital_duration h
JOIN cb_dtl_dm_facility f ON f.facility_key = h.facility_key
GROUP BY f.facility_name ORDER BY f.facility_name;

-- 4. Find the next admission within 30 days of prior discharge
-- Grain: One qualifying subsequent admission
-- Filter: Nonnegative gap no greater than 30 elapsed days; sequential prior admission
WITH sequenced AS (
 SELECT h.*, LAG(discharged_at) OVER
  (PARTITION BY client_key ORDER BY admitted_at, hospital_stay_key) AS prior_discharge
 FROM cb_dtl_ft_hospital_stay h
)
SELECT client_key, hospital_stay_key,
 ROUND((CAST(SYS_EXTRACT_UTC(admitted_at) AS DATE)
      - CAST(SYS_EXTRACT_UTC(prior_discharge) AS DATE)), 6) AS gap_days
FROM sequenced
WHERE admitted_at >= prior_discharge
  AND admitted_at <= prior_discharge + NUMTODSINTERVAL(30, 'DAY')
ORDER BY client_key, admitted_at, hospital_stay_key;

-- 5. Find clients with repeated emergency attendances
-- Grain: One client
-- Filter: At least two attendances during January–March
SELECT client_key, COUNT(*) AS emergency_attendances
FROM cb_dtl_ft_emergency_visit
WHERE arrival_date_key >= 20260101 AND arrival_date_key < 20260401
GROUP BY client_key HAVING COUNT(*) >= 2 ORDER BY client_key;

-- 6. Compare recorded pharmacy utilization by geography
-- Grain: One dispensing province
-- Filter: All fixture dates; retain unknown cost
SELECT g.province, COUNT(*) AS dispensing_lines, SUM(f.quantity) AS units,
 COUNT(f.total_cost) AS recorded_cost_lines, SUM(f.total_cost) AS known_cost_npr
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
GROUP BY g.province ORDER BY g.province;

-- 7. Construct age bands at a fixed reporting date
-- Grain: One age band
-- Filter: Age at 2026-03-31; completed years using Oracle MONTHS_BETWEEN convention
WITH ages AS (
 SELECT client_key, FLOOR(MONTHS_BETWEEN(DATE '2026-03-31', birth_date)/12) AS age_years
 FROM cb_dtl_dm_client
), banded AS (
 SELECT client_key, CASE WHEN age_years < 18 THEN '00-17'
 WHEN age_years < 45 THEN '18-44' WHEN age_years < 65 THEN '45-64'
 ELSE '65+' END AS age_band FROM ages
)
SELECT b.age_band, COUNT(*) AS dispensing_lines,
 COUNT(DISTINCT f.client_key) AS distinct_clients
FROM cb_dtl_ft_pharmacy_dispense f JOIN banded b ON b.client_key = f.client_key
GROUP BY b.age_band ORDER BY b.age_band;

-- 8. Rank facilities by admission count
-- Grain: One facility
-- Filter: All four admissions; count admissions, not discharge or diagnosis rows
WITH utilization AS (
 SELECT facility_key, COUNT(*) AS admissions
 FROM cb_dtl_ft_hospital_stay GROUP BY facility_key
)
SELECT f.facility_name, u.admissions,
 RANK() OVER (ORDER BY u.admissions DESC) AS utilization_rank
FROM utilization u JOIN cb_dtl_dm_facility f ON f.facility_key = u.facility_key
ORDER BY utilization_rank, f.facility_name;

-- 9. Create a client-level longitudinal utilization dataset
-- Grain: One client
-- Filter: All fixture dates; include clients with zero encounters in a setting
WITH p AS (SELECT client_key, COUNT(*) AS pharmacy_lines
 FROM cb_dtl_ft_pharmacy_dispense GROUP BY client_key),
h AS (SELECT client_key, COUNT(*) AS admissions
 FROM cb_dtl_ft_hospital_stay GROUP BY client_key),
e AS (SELECT client_key, COUNT(*) AS ed_visits
 FROM cb_dtl_ft_emergency_visit GROUP BY client_key),
v AS (SELECT client_key, COUNT(*) AS physician_visits
 FROM cb_dtl_ft_physician_visit GROUP BY client_key)
SELECT c.client_id, NVL(p.pharmacy_lines,0) AS pharmacy_lines,
 NVL(h.admissions,0) AS admissions, NVL(e.ed_visits,0) AS ed_visits,
 NVL(v.physician_visits,0) AS physician_visits
FROM cb_dtl_dm_client c LEFT JOIN p ON p.client_key=c.client_key
LEFT JOIN h ON h.client_key=c.client_key LEFT JOIN e ON e.client_key=c.client_key
LEFT JOIN v ON v.client_key=c.client_key ORDER BY c.client_id;

-- 10. Build a balanced province-month panel
-- Grain: One province-month including zero-use cells
-- Filter: January–March 2026; panel geography dimension contains one row per illustrated province
WITH months AS (SELECT DISTINCT month_label FROM cb_dtl_dm_date
 WHERE calendar_date >= DATE '2026-01-01' AND calendar_date < DATE '2026-04-01'),
counts AS (SELECT f.geography_key, d.month_label, COUNT(*) AS dispensing_lines
 FROM cb_dtl_ft_pharmacy_dispense f JOIN cb_dtl_dm_date d ON d.date_key=f.dispense_date_key
 GROUP BY f.geography_key, d.month_label)
SELECT g.province, m.month_label, NVL(c.dispensing_lines,0) AS dispensing_lines
FROM cb_dtl_dm_geography g CROSS JOIN months m
LEFT JOIN counts c ON c.geography_key=g.geography_key AND c.month_label=m.month_label
ORDER BY g.province, m.month_label;

-- 11. Calculate a rolling three-calendar-month utilization count
-- Grain: One calendar month for selected client 1
-- Filter: Client 1; January–March 2026, no earlier history
WITH months AS (SELECT DISTINCT month_label FROM cb_dtl_dm_date),
counts AS (SELECT d.month_label, COUNT(*) AS lines
 FROM cb_dtl_ft_pharmacy_dispense f JOIN cb_dtl_dm_date d ON d.date_key=f.dispense_date_key
 WHERE f.client_key=1 GROUP BY d.month_label),
complete_months AS (SELECT m.month_label, NVL(c.lines,0) AS lines
 FROM months m LEFT JOIN counts c ON c.month_label=m.month_label)
SELECT month_label, lines,
 SUM(lines) OVER (ORDER BY month_label ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
 AS trailing_three_month_lines
FROM complete_months ORDER BY month_label;

-- 12. Form an illustrative chronic-condition cohort candidate
-- Grain: One candidate client
-- Filter: At least two distinct local dates with illustrative ICD-10 E11.9 plus SYN-A exposure during fixture dates
WITH diagnosis_dates AS (
 SELECT v.client_key, v.visit_date_key AS diagnosis_date_key
 FROM cb_dtl_ft_physician_visit v JOIN cb_dtl_dm_diagnosis d ON d.diagnosis_key=v.diagnosis_key
 WHERE d.code_system='ICD-10' AND d.diagnosis_code='E11.9'
 UNION
 SELECT h.client_key, h.admit_date_key
 FROM cb_dtl_ft_hospital_stay h
 JOIN cb_dtl_br_stay_diagnosis b ON b.hospital_stay_key=h.hospital_stay_key
 JOIN cb_dtl_dm_diagnosis d ON d.diagnosis_key=b.diagnosis_key
 WHERE d.code_system='ICD-10' AND d.diagnosis_code='E11.9'
), evidence AS (SELECT client_key, COUNT(*) AS diagnosis_dates
 FROM diagnosis_dates GROUP BY client_key HAVING COUNT(*)>=2),
exposure AS (SELECT f.client_key, COUNT(*) AS exposure_lines
 FROM cb_dtl_ft_pharmacy_dispense f JOIN cb_dtl_dm_health_product p ON p.product_key=f.product_key
 WHERE p.product_code='SYN-A' GROUP BY f.client_key)
SELECT e.client_key, e.diagnosis_dates, x.exposure_lines
FROM evidence e JOIN exposure x ON x.client_key=e.client_key ORDER BY e.client_key;
