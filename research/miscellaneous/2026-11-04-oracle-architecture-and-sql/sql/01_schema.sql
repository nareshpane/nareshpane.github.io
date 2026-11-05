-- Oracle 19c. Run once in an EMPTY dedicated educational schema.
-- Synthetic teaching design; no health-data security or production provisioning is implied.
CREATE TABLE cb_dtl_dm_date (
  date_key NUMBER(8) NOT NULL,
  calendar_date DATE NOT NULL,
  calendar_year NUMBER(4) NOT NULL,
  month_number NUMBER(2) NOT NULL,
  month_label VARCHAR2(7) NOT NULL,
  quarter_number NUMBER(1) NOT NULL,
  day_of_week NUMBER(1) NOT NULL,
  fiscal_period VARCHAR2(40) NOT NULL,
  PRIMARY KEY (date_key),
  UNIQUE (calendar_date)
);

CREATE TABLE cb_dtl_dm_geography (
  geography_key NUMBER NOT NULL,
  province VARCHAR2(40) NOT NULL,
  district VARCHAR2(40) NOT NULL,
  municipality VARCHAR2(80) NOT NULL,
  ward NUMBER(2) NOT NULL,
  PRIMARY KEY (geography_key)
);

CREATE TABLE cb_dtl_dm_client (
  client_key NUMBER NOT NULL,
  client_id VARCHAR2(30) NOT NULL,
  birth_date DATE NOT NULL,
  residence_geography_key NUMBER NOT NULL,
  effective_from DATE NOT NULL,
  effective_to DATE,
  current_flag CHAR(1) NOT NULL,
  PRIMARY KEY (client_key),
  UNIQUE (client_id, effective_from),
  FOREIGN KEY (residence_geography_key) REFERENCES cb_dtl_dm_geography (geography_key),
  CHECK (current_flag IN ('Y','N')),
  CHECK (effective_to IS NULL OR effective_to > effective_from)
);

CREATE TABLE cb_dtl_dm_prescriber (
  prescriber_key NUMBER NOT NULL,
  prescriber_id VARCHAR2(30) NOT NULL,
  prescriber_name VARCHAR2(80) NOT NULL,
  specialty VARCHAR2(60) NOT NULL,
  status VARCHAR2(12) NOT NULL,
  geography_key NUMBER NOT NULL,
  PRIMARY KEY (prescriber_key),
  UNIQUE (prescriber_id),
  FOREIGN KEY (geography_key) REFERENCES cb_dtl_dm_geography (geography_key)
);

CREATE TABLE cb_dtl_dm_health_product (
  product_key NUMBER NOT NULL,
  product_code VARCHAR2(30) NOT NULL,
  product_name VARCHAR2(80) NOT NULL,
  ingredient VARCHAR2(80) NOT NULL,
  strength VARCHAR2(40) NOT NULL,
  dosage_form VARCHAR2(30) NOT NULL,
  classification VARCHAR2(80) NOT NULL,
  brand VARCHAR2(50) NOT NULL,
  effective_from DATE NOT NULL,
  effective_to DATE,
  PRIMARY KEY (product_key),
  UNIQUE (product_code)
);

CREATE TABLE cb_dtl_dm_pharmacy (
  pharmacy_key NUMBER NOT NULL,
  pharmacy_id VARCHAR2(30) NOT NULL,
  pharmacy_name VARCHAR2(80) NOT NULL,
  geography_key NUMBER NOT NULL,
  PRIMARY KEY (pharmacy_key),
  UNIQUE (pharmacy_id),
  FOREIGN KEY (geography_key) REFERENCES cb_dtl_dm_geography (geography_key)
);

CREATE TABLE cb_dtl_dm_facility (
  facility_key NUMBER NOT NULL,
  facility_id VARCHAR2(30) NOT NULL,
  facility_name VARCHAR2(80) NOT NULL,
  facility_type VARCHAR2(40) NOT NULL,
  ownership_category VARCHAR2(40) NOT NULL,
  service_capability VARCHAR2(100) NOT NULL,
  geography_key NUMBER NOT NULL,
  PRIMARY KEY (facility_key),
  UNIQUE (facility_id),
  FOREIGN KEY (geography_key) REFERENCES cb_dtl_dm_geography (geography_key)
);

CREATE TABLE cb_dtl_dm_diagnosis (
  diagnosis_key NUMBER NOT NULL,
  diagnosis_code VARCHAR2(16) NOT NULL,
  code_system VARCHAR2(20) NOT NULL,
  code_version VARCHAR2(30) NOT NULL,
  description VARCHAR2(120) NOT NULL,
  PRIMARY KEY (diagnosis_key),
  UNIQUE (diagnosis_code, code_system, code_version)
);

CREATE TABLE cb_dtl_dm_service (
  service_key NUMBER NOT NULL,
  service_code VARCHAR2(30) NOT NULL,
  service_description VARCHAR2(100) NOT NULL,
  PRIMARY KEY (service_key),
  UNIQUE (service_code)
);

CREATE TABLE cb_dtl_ft_pharmacy_dispense (
  dispense_line_key NUMBER NOT NULL,
  source_system VARCHAR2(40) NOT NULL,
  source_record_id VARCHAR2(40) NOT NULL,
  prescription_id VARCHAR2(40) NOT NULL,
  dispensing_id VARCHAR2(40) NOT NULL,
  client_key NUMBER NOT NULL,
  prescriber_key NUMBER NOT NULL,
  product_key NUMBER NOT NULL,
  pharmacy_key NUMBER NOT NULL,
  geography_key NUMBER NOT NULL,
  dispense_date_key NUMBER NOT NULL,
  dispensed_at TIMESTAMP(0) WITH TIME ZONE NOT NULL,
  quantity NUMBER(12,2) NOT NULL,
  days_supply NUMBER NOT NULL,
  ingredient_cost NUMBER(12,2),
  dispensing_fee NUMBER(12,2),
  total_cost NUMBER(12,2),
  payment_category VARCHAR2(40) NOT NULL,
  PRIMARY KEY (dispense_line_key),
  UNIQUE (source_system, source_record_id),
  FOREIGN KEY (client_key) REFERENCES cb_dtl_dm_client (client_key),
  FOREIGN KEY (prescriber_key) REFERENCES cb_dtl_dm_prescriber (prescriber_key),
  FOREIGN KEY (product_key) REFERENCES cb_dtl_dm_health_product (product_key),
  FOREIGN KEY (pharmacy_key) REFERENCES cb_dtl_dm_pharmacy (pharmacy_key),
  FOREIGN KEY (geography_key) REFERENCES cb_dtl_dm_geography (geography_key),
  FOREIGN KEY (dispense_date_key) REFERENCES cb_dtl_dm_date (date_key),
  CHECK (quantity > 0),
  CHECK (days_supply > 0),
  CHECK (total_cost IS NULL OR total_cost >= 0)
);

CREATE TABLE cb_dtl_ft_hospital_stay (
  hospital_stay_key NUMBER NOT NULL,
  encounter_id VARCHAR2(40) NOT NULL,
  client_key NUMBER NOT NULL,
  facility_key NUMBER NOT NULL,
  admit_date_key NUMBER NOT NULL,
  discharge_date_key NUMBER NOT NULL,
  admitted_at TIMESTAMP(0) WITH TIME ZONE NOT NULL,
  discharged_at TIMESTAMP(0) WITH TIME ZONE NOT NULL,
  length_of_stay_hours NUMBER(16,8) NOT NULL,
  admission_type VARCHAR2(40) NOT NULL,
  referral_source VARCHAR2(40) NOT NULL,
  discharge_disposition_code VARCHAR2(40) NOT NULL,
  PRIMARY KEY (hospital_stay_key),
  UNIQUE (encounter_id),
  FOREIGN KEY (client_key) REFERENCES cb_dtl_dm_client (client_key),
  FOREIGN KEY (facility_key) REFERENCES cb_dtl_dm_facility (facility_key),
  FOREIGN KEY (admit_date_key) REFERENCES cb_dtl_dm_date (date_key),
  FOREIGN KEY (discharge_date_key) REFERENCES cb_dtl_dm_date (date_key),
  CHECK (discharged_at >= admitted_at),
  CHECK (length_of_stay_hours >= 0)
);

CREATE TABLE cb_dtl_br_stay_diagnosis (
  hospital_stay_key NUMBER NOT NULL,
  diagnosis_key NUMBER NOT NULL,
  diagnosis_position NUMBER(3) NOT NULL,
  PRIMARY KEY (hospital_stay_key, diagnosis_position),
  UNIQUE (hospital_stay_key, diagnosis_key),
  FOREIGN KEY (hospital_stay_key) REFERENCES cb_dtl_ft_hospital_stay (hospital_stay_key),
  FOREIGN KEY (diagnosis_key) REFERENCES cb_dtl_dm_diagnosis (diagnosis_key),
  CHECK (diagnosis_position > 0)
);

CREATE TABLE cb_dtl_ft_physician_visit (
  physician_visit_key NUMBER NOT NULL,
  encounter_id VARCHAR2(40) NOT NULL,
  client_key NUMBER NOT NULL,
  prescriber_key NUMBER NOT NULL,
  facility_key NUMBER NOT NULL,
  geography_key NUMBER NOT NULL,
  service_key NUMBER NOT NULL,
  diagnosis_key NUMBER NOT NULL,
  visit_date_key NUMBER NOT NULL,
  scheduled_at TIMESTAMP(0) WITH TIME ZONE NOT NULL,
  encountered_at TIMESTAMP(0) WITH TIME ZONE NOT NULL,
  encounter_status VARCHAR2(40) NOT NULL,
  consultation_mode VARCHAR2(40) NOT NULL,
  service_amount NUMBER(12,2) NOT NULL,
  claim_amount NUMBER(12,2),
  PRIMARY KEY (physician_visit_key),
  UNIQUE (encounter_id),
  FOREIGN KEY (client_key) REFERENCES cb_dtl_dm_client (client_key),
  FOREIGN KEY (prescriber_key) REFERENCES cb_dtl_dm_prescriber (prescriber_key),
  FOREIGN KEY (facility_key) REFERENCES cb_dtl_dm_facility (facility_key),
  FOREIGN KEY (geography_key) REFERENCES cb_dtl_dm_geography (geography_key),
  FOREIGN KEY (service_key) REFERENCES cb_dtl_dm_service (service_key),
  FOREIGN KEY (diagnosis_key) REFERENCES cb_dtl_dm_diagnosis (diagnosis_key),
  FOREIGN KEY (visit_date_key) REFERENCES cb_dtl_dm_date (date_key),
  CHECK (encounter_status = 'COMPLETED')
);

CREATE TABLE cb_dtl_ft_emergency_visit (
  emergency_visit_key NUMBER NOT NULL,
  encounter_id VARCHAR2(40) NOT NULL,
  client_key NUMBER NOT NULL,
  facility_key NUMBER NOT NULL,
  diagnosis_key NUMBER NOT NULL,
  arrival_date_key NUMBER NOT NULL,
  arrived_at TIMESTAMP(0) WITH TIME ZONE NOT NULL,
  triaged_at TIMESTAMP(0) WITH TIME ZONE NOT NULL,
  departed_at TIMESTAMP(0) WITH TIME ZONE NOT NULL,
  triage_category NUMBER NOT NULL,
  waiting_minutes NUMBER NOT NULL,
  ed_length_of_stay_minutes NUMBER NOT NULL,
  disposition VARCHAR2(40) NOT NULL,
  hospital_stay_key NUMBER,
  PRIMARY KEY (emergency_visit_key),
  UNIQUE (encounter_id),
  FOREIGN KEY (client_key) REFERENCES cb_dtl_dm_client (client_key),
  FOREIGN KEY (facility_key) REFERENCES cb_dtl_dm_facility (facility_key),
  FOREIGN KEY (diagnosis_key) REFERENCES cb_dtl_dm_diagnosis (diagnosis_key),
  FOREIGN KEY (arrival_date_key) REFERENCES cb_dtl_dm_date (date_key),
  FOREIGN KEY (hospital_stay_key) REFERENCES cb_dtl_ft_hospital_stay (hospital_stay_key),
  CHECK (departed_at >= arrived_at),
  CHECK (triaged_at >= arrived_at),
  CHECK (triaged_at <= departed_at),
  CHECK (triage_category BETWEEN 1 AND 5)
);

