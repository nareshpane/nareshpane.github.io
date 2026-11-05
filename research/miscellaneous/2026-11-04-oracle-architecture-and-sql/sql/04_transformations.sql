-- Minimal, runnable ETL demonstration. Existing warehouse fixtures stay unchanged.
-- Source event uniqueness is (source_system, source_record_id); a line key alone
-- does not establish that two submissions are different clinical events.
CREATE TABLE hdn_stg_pharmacy AS
SELECT CAST(NULL AS NUMBER) AS staging_row, p.*
FROM cb_dtl_ft_pharmacy_dispense p WHERE 1 = 0;
INSERT INTO hdn_stg_pharmacy
SELECT 1, p.* FROM cb_dtl_ft_pharmacy_dispense p WHERE dispense_line_key = 1;
INSERT INTO hdn_stg_pharmacy
SELECT 2, p.* FROM cb_dtl_ft_pharmacy_dispense p WHERE dispense_line_key = 1;
INSERT INTO hdn_stg_pharmacy
SELECT 3, p.* FROM cb_dtl_ft_pharmacy_dispense p WHERE dispense_line_key = 2;
UPDATE hdn_stg_pharmacy SET product_key = 999 WHERE staging_row = 3;

CREATE OR REPLACE VIEW hdn_v_stage_validation AS
SELECT s.*,
       ROW_NUMBER() OVER (
         PARTITION BY source_system, source_record_id
         ORDER BY staging_row) AS submission_number,
       CASE WHEN EXISTS (SELECT 1 FROM cb_dtl_dm_health_product p
                         WHERE p.product_key = s.product_key)
            THEN 'VALID' ELSE 'REJECT_PRODUCT' END AS validation_status
FROM hdn_stg_pharmacy s;

-- Expected: VALID first submission=1; duplicate submission=1; rejected product=1.
SELECT staging_row, source_record_id, submission_number, validation_status
FROM hdn_v_stage_validation ORDER BY staging_row;

-- Idempotent insert-only load. The accepted event already exists, so zero new
-- rows are inserted. No existing clinical facts are overwritten by a resend.
-- New accepted lines require warehouse key allocation in a production design.
MERGE INTO cb_dtl_ft_pharmacy_dispense t
USING (SELECT * FROM hdn_v_stage_validation
       WHERE submission_number = 1 AND validation_status = 'VALID') s
ON (t.source_system = s.source_system AND t.source_record_id = s.source_record_id)
WHEN NOT MATCHED THEN INSERT (
  dispense_line_key, source_system, source_record_id, prescription_id, dispensing_id,
  client_key, prescriber_key, product_key, pharmacy_key, geography_key,
  dispense_date_key, dispensed_at, quantity, days_supply,
  ingredient_cost, dispensing_fee, total_cost, payment_category)
VALUES (
  s.dispense_line_key, s.source_system, s.source_record_id, s.prescription_id, s.dispensing_id,
  s.client_key, s.prescriber_key, s.product_key, s.pharmacy_key, s.geography_key,
  s.dispense_date_key, s.dispensed_at, s.quantity, s.days_supply,
  s.ingredient_cost, s.dispensing_fee, s.total_cost, s.payment_category);
COMMIT;

-- Parse a raw source timestamp with an explicit format and retain its offset.
SELECT TO_TIMESTAMP_TZ('2026-01-10T10:15:00+05:45',
                      'YYYY-MM-DD"T"HH24:MI:SSTZH:TZM') AS canonical_instant
FROM dual;

-- Type 2 teaching example in an isolated table: half-open effective intervals.
-- Both rows refer to one business identity; only the second version is current.
CREATE TABLE hdn_scd_client_demo (
  client_key NUMBER PRIMARY KEY,
  client_id VARCHAR2(30) NOT NULL,
  geography_key NUMBER NOT NULL REFERENCES cb_dtl_dm_geography(geography_key),
  effective_from DATE NOT NULL,
  effective_to DATE,
  current_flag CHAR(1) NOT NULL CHECK (current_flag IN ('Y','N')),
  UNIQUE (client_id, effective_from),
  CHECK (effective_to IS NULL OR effective_to > effective_from)
);
INSERT INTO hdn_scd_client_demo VALUES (101, 'CLIENT_SYN_005', 1, DATE '2026-01-01', DATE '2026-02-01', 'N');
INSERT INTO hdn_scd_client_demo VALUES (102, 'CLIENT_SYN_005', 2, DATE '2026-02-01', NULL, 'Y');
CREATE UNIQUE INDEX hdn_one_current_client
ON hdn_scd_client_demo(CASE WHEN current_flag = 'Y' THEN client_id END);
-- NULL keys are omitted by this single-column B-tree expression, so only
-- current rows are unique. Non-overlapping history also needs load validation.
SELECT client_key FROM hdn_scd_client_demo
WHERE client_id = 'CLIENT_SYN_005' AND DATE '2026-02-10' >= effective_from
  AND (effective_to IS NULL OR DATE '2026-02-10' < effective_to);
COMMIT;
