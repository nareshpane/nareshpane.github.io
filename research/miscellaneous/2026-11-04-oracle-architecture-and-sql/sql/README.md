# HealthDataNepal: Oracle 19c teaching scripts

All records, provider identities, product identities, and facilities are synthetic. Real Nepal province/district labels provide geographic context. The proposed HealthDataNepal and Pharma-Nepal systems are fictional. No real health records are included.

## Run in dependency order

Use an **empty, dedicated educational Oracle 19c schema**, with CREATE SESSION, CREATE TABLE, CREATE VIEW and an appropriate tablespace quota already arranged by its administrator. Owners can create indexes on their own tables without a separate CREATE INDEX system privilege. Open `00_run_all.sql` in Oracle SQL Developer and choose **Run Script (F5)**, or invoke it from SQL*Plus/SQLcl. Relative `@@` references resolve to this script's directory. The scripts create objects and commit; they are not rerunnable migrations and do not remove existing objects. No user creation, external service, production grants, partitioning option, or special health dataset is required.

1. `01_schema.sql` creates dimensions before their dependent facts and the diagnosis bridge.
2. `02_seed.sql` inserts deterministic dimensions and facts.
3. `03_views.sql` exposes enriched dispensing rows, UTC-based exact stay durations, and monthly summaries.
4. `04_transformations.sql` creates an isolated staging example, validates a missing product, deduplicates a resend, and demonstrates an insert-only `MERGE`. Its accepted event already exists; the original fixtures remain unchanged. A separate small SCD2 table illustrates effective versions and current-row uniqueness.
5. `05_aggregates.sql` covers counts, moments, LISTAGG, correlation/regression, grouping sets, ROLLUP/CUBE, and diagnosis join fan-out.
6. `06_analytics.sql` covers ranking/ties, distributions, offsets, first/last values, running and moving windows, temporal RANGE, and filtering analytic results in an outer query.
7. `07_analyst_exercises.sql` contains the 12 annotated analytical queries. Inputs, joins, filters, output grain, results, pitfalls, and interpretation are in `../data/exercises.json`.

`WHENEVER SQLERROR EXIT` is intended for the fresh-schema batch. If adapting statements interactively, review Oracle DDL's implicit commits. Do not run this teaching setup against a populated operational or production schema.

## Schema and browser data contract

The browser loads `../data/sample.js` as `window.HDN_DATA`; the same object is preserved as `sample.json` for other tools. `exercises.js` and `exercises.json` similarly share all 12 exercise definitions. No Oracle connection is made by the page. The date dimension has all 90 Gregorian dates in January–March 2026. Fiscal labels are expressly illustrative, not an official Nepal fiscal-calendar mapping.

| Object key | Oracle table | Grain / primary key |
|---|---|---|
| `date` | `cb_dtl_dm_date` | One Gregorian date; `date_key` |
| `geography` | `cb_dtl_dm_geography` | One illustrated locality; `geography_key` |
| `clients` | `cb_dtl_dm_client` | One effective client version; `client_key` |
| `prescribers` | `cb_dtl_dm_prescriber` | One provider; `prescriber_key` |
| `products` | `cb_dtl_dm_health_product` | One product; `product_key` |
| `pharmacies` | `cb_dtl_dm_pharmacy` | One pharmacy; `pharmacy_key` |
| `facilities` | `cb_dtl_dm_facility` | One hospital facility; `facility_key` |
| `diagnoses` | `cb_dtl_dm_diagnosis` | One versioned classification code; `diagnosis_key` |
| `services` | `cb_dtl_dm_service` | One service definition; `service_key` |
| `pharmacy` | `cb_dtl_ft_pharmacy_dispense` | One product dispensing line; `dispense_line_key` |
| `hospital` | `cb_dtl_ft_hospital_stay` | One completed admission episode; `hospital_stay_key` |
| `hospital_diagnosis` | `cb_dtl_br_stay_diagnosis` | One positioned diagnosis per stay; `(hospital_stay_key, diagnosis_position)` |
| `physician` | `cb_dtl_ft_physician_visit` | One completed encounter; `physician_visit_key` |
| `emergency` | `cb_dtl_ft_emergency_visit` | One ED attendance; `emergency_visit_key` |

All rows use snake_case column names matching the SQL definitions. Pharmacy `geography_key` is the dispensing location; client `residence_geography_key` is a distinct role. Dates use `YYYYMMDD` surrogate values, without replacing full event timestamps. All fixture event strings carry explicit `+05:45`; SQL uses `TIMESTAMP(0) WITH TIME ZONE` and explicit `TO_TIMESTAMP_TZ` format masks. UTC normalization is used before duration arithmetic. Gregorian display dates and Nepal-local date keys are aligned for this fixture.

Hospital timestamps are `admitted_at` and `discharged_at`; pharmacy time is `dispensed_at`; physician times are `scheduled_at` and `encountered_at`; emergency times are `arrived_at`, `triaged_at`, and `departed_at`. Scheduling does not imply attendance. The physician fact contains completed encounters only; `claim_amount` remains NULL because no reimbursement claim has been established. ED `hospital_stay_key` is an optional link to the resulting episode, preserving both encounter grains.

Dimension business identifiers are unique in the simplified provider/product/facility dimensions. Client versions instead use `(client_id, effective_from)` and a surrogate key. The isolated SCD2 demonstration provides two versions for a separate business identity. In a production Type 2 model, every versioned provider/product/facility would also need versioned business-key uniqueness, nonoverlap validation, event-time lookup, and late-arrival/correction handling. Do not add current-only dimension joins to historical facts blindly.

The bridge's position 1 is the principal diagnosis, while subsequent positions are secondary. It makes no allocation of length of stay or cost across diagnoses. Code descriptions and the use of an ICD-10 label are illustrative; the fixture does not prescribe Nepal's classification release or substitute ICD-10-CA for ICD-10. Product labels are fictional and quantities across products are not standardized doses.

## Rebuild and validate

From the asset directory, using Python 3 with no external packages:

```powershell
python data/build_samples.py
python data/build_exercises.py
python data/verify_samples.py
```

The generators replace the checked-in fixture, seed DDL/inserts, and exercise files deterministically. The validator checks relational references, unique identifiers, timestamp durations, ED-to-admission linkage, JSON/JS equivalence, seed row counts, dimension-before-fact creation order, and independently recomputes the output of all 12 exercises. Its numerical report is `../data/validation.json`.

Expected baseline: **16 dispensing lines; 4 clients; 15 recorded costs; NPR 1,610 known cost; quantity 370; 4 stays; 6 diagnosis associations; 4 physician encounters; 4 ED attendances**. The unknown cost is preserved as NULL. Mean known cost is 1,610/15 = 107.333333; treating the unknown cost as zero would instead yield 100.625 per dispensing line. The first admission lasts 54.508333 hours, including 30 seconds. Diagnosis fan-out produces six joined rows from four stays, inflating summed stay duration from 174.508333 to 277.016667 hours.

**Validation limitation:** No Oracle server was available and these scripts have not been executed on Oracle. Static review and independent Python calculations are complete; actual Oracle parsing, permissions, NLS environment, and SQL Developer execution remain unverified. The design is a minimal teaching schema, not a deployed national warehouse, privacy control implementation, or clinical cohort definition.

Technical references reviewed: [Oracle 19c analytic functions](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/Analytic-Functions.html), [aggregate functions](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/Aggregate-Functions.html), and [MERGE](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/MERGE.html). All explanations and data are original.
