-- Views preserve the declared fact grain. Costs remain NULL when unrecorded.
CREATE OR REPLACE VIEW hdn_v_pharmacy AS
SELECT f.*, g.province, g.district, p.product_name, p.classification,
       c.client_id, ph.pharmacy_name
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
JOIN cb_dtl_dm_health_product p ON p.product_key = f.product_key
JOIN cb_dtl_dm_client c ON c.client_key = f.client_key
JOIN cb_dtl_dm_pharmacy ph ON ph.pharmacy_key = f.pharmacy_key;

-- SYS_EXTRACT_UTC normalizes each instant; DATE casts preserve seconds here
-- because the educational timestamp columns deliberately have precision zero.
-- Keep an INTERVAL-based calculation instead when fractional seconds matter.
CREATE OR REPLACE VIEW hdn_v_hospital_duration AS
SELECT h.*,
       (CAST(SYS_EXTRACT_UTC(discharged_at) AS DATE)
        - CAST(SYS_EXTRACT_UTC(admitted_at) AS DATE)) * 24 AS computed_los_hours
FROM cb_dtl_ft_hospital_stay h;

CREATE OR REPLACE VIEW hdn_v_monthly_pharmacy AS
SELECT d.month_label, g.province, COUNT(*) AS dispensing_lines,
       COUNT(DISTINCT f.client_key) AS distinct_clients,
       COUNT(f.total_cost) AS recorded_cost_lines, SUM(f.total_cost) AS known_cost
FROM cb_dtl_ft_pharmacy_dispense f
JOIN cb_dtl_dm_date d ON d.date_key = f.dispense_date_key
JOIN cb_dtl_dm_geography g ON g.geography_key = f.geography_key
GROUP BY d.month_label, g.province;
