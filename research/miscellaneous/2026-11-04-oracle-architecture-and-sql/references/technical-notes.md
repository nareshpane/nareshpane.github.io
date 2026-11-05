# HealthDataNepal: source and technical review notes

References checked 2026-11-05 UTC, beginning 00:22:25 UTC (November 4 in America/Los_Angeles). These are original summaries of public documentation, with no confidential database access. Oracle **19c** is the compatibility target. The [requested 18c aggregate page](https://docs.oracle.com/en/database/oracle/oracle-database/18/sqlrf/Aggregate-Functions.html) and [requested 21c analytic page](https://docs.oracle.com/en/database/oracle/oracle-database/21/sqlrf/Analytic-Functions.html) were accessible and are retained as supplemental references; 21c is later than 19c. The structured bibliography is `sources.json`.

## What is documented, and what is proposed

| Documented Canadian practice | Architectural lesson (interpretation) | Possible HealthDataNepal adaptation (proposal) |
|---|---|---|
| B.C. grants Healthideas warehouse access under specified circumstances; its academic research disclosure route uses Population Data B.C. [B.C. Ministry of Health](https://www2.gov.bc.ca/gov/content/health/conducting-health-research-evaluation/data-access-health-data-central/health-sector-partner-access-to-data/third-party-access) | Warehouse access needs institutional processes and explicit purposes. | A government-governed warehouse with approved research workspaces and documented access applications. |
| ICES describes record-level, coded, linkable datasets, including longitudinal publicly funded Ontario health-service records and integration with registries and surveys. [ICES](https://www.ices.on.ca/about-ices-data/) | Linkage and research integration need managed identifiers and metadata. | Separate identity-linkage service and pseudonymous analytical client identifiers. |
| Alberta lists community pharmacy PIN dispenses, ambulatory care including ED, inpatient/DAD, population registry and practitioner claims; PIN dispensing data exclude financial information. [Alberta Health data access](https://www.alberta.ca/health-research) | Similar subject areas can have different coverage, fields and administrative meanings. | Separate encounter and dispensing facts; record source provenance and availability of cost measures. |
| DAD captures administrative, clinical and demographic information about hospital discharges, including deaths, sign-outs and transfers; jurisdictional coverage and some day-surgery participation vary. [CIHI DAD](https://www.cihi.ca/en/discharge-abstract-database-dad-metadata) | Specify event grain and coverage before interpreting totals. | A hospitalization fact with admission/discharge times, disposition and a bridge for multiple diagnoses; publish coverage metadata. |
| NACRS covers ED, day surgeries and outpatient/community clinics, with jurisdiction-dependent participation and reporting levels. [CIHI NACRS](https://www.cihi.ca/en/national-ambulatory-care-reporting-system-nacrs-metadata) | Ambulatory care is broader than ED and does not mean ambulance transport. | Separate ED, outpatient and ambulance datasets when their events differ. |

These sources do **not** establish that the jurisdictions share Oracle software or identical internal schemas. They do not document HealthDataNepal or Pharma-Nepal, which are fictional. No confidential Canadian schema has been inferred. No claim is made that this educational design complies with Nepalese law. A real system needs an established legal basis, accountable custodians, applicable approvals and locally determined disclosure rules.

## Oracle keys and NULL: points that affect demonstrations

- A primary key is unique and non-null. A foreign key references a primary or unique key; a nullable foreign key does not require a parent for a null value. A composite foreign key with any null component can satisfy the referential constraint.
- A single-column `UNIQUE` permits multiple nulls. A composite unique key permits all-null rows; repeated non-null components with nulls in the other positions can still violate uniqueness. Example: `(source_system, source_id)` repeated as `('PHARMA', NULL)` conflicts, while `(NULL, NULL)` can repeat. Use `NOT NULL` on all event-identity components.
- A `CHECK` accepts TRUE or UNKNOWN, so `CHECK (quantity > 0)` alone allows null. Pair it with `NOT NULL` if quantity is compulsory.
- Oracle 19c does not allow `TIMESTAMP WITH TIME ZONE` in a primary or unique key. Use surrogate keys and, where necessary, UTC `TIMESTAMP` version-boundary columns for constrained SCD keys.

Source: [Oracle 19c constraint](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/constraint.html).

## Aggregates and analytic windows

Grouped aggregates reduce each group to one row. `HAVING` filters groups. `SUM` and `AVG` ignore null arguments; null is not an observed zero. An all-null or empty input produces null for these aggregates. [Oracle 19c Aggregate Functions](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/Aggregate-Functions.html)

`COUNT(*)` counts rows; `COUNT(expression)` counts non-null expressions; `COUNT(DISTINCT client_key)` counts distinct non-null clients. Analytic `COUNT(DISTINCT ...)` allows partitioning but not analytic ordering or a window frame. [Oracle 19c COUNT](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/COUNT.html)

Analytic expressions act after joins, `WHERE`, grouping and `HAVING`. Filter an analytic result through an outer query/CTE. For functions supporting a frame, an analytic ordering without an explicit frame defaults to `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which includes peers. A rowwise running total should explicitly use `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` and a unique tie-breaker such as `(dispensed_at, dispense_line_key)`. Use a final query `ORDER BY` to guarantee display order. `ROW_NUMBER` can otherwise assign ties nondeterministically; ranking functions intentionally share ranks across ties. A three-row window is not a three-month window unless rows are first aggregated and missing months filled. [Oracle 19c Analytic Functions](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/Analytic-Functions.html)

`VARIANCE` returns zero for one non-null value and sample variance for multiple values; `STDDEV` similarly returns zero for a singleton, whereas `STDDEV_SAMP` returns null. Teach population and sample denominators separately. [VARIANCE](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/VARIANCE.html), [STDDEV](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/STDDEV.html)

## Architecture accuracy

The instance manages persistent database files through memory and processes. One database can have multiple instances under RAC; one instance manages one database at a time. [Oracle Database Instance](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/oracle-database-instance.html)

The SGA is shared by server/background processes; the PGA is private to an Oracle process. UGA contains session state and its placement depends on connection architecture. Avoid drawing a PGA as a shared SGA pool. [Memory Architecture](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/memory-architecture.html)

DBW writes dirty data buffers; LGWR writes redo, including commit records. A normal synchronous commit waits for required redo persistence, not all changed data blocks. CKPT updates checkpoint information and signals DBW; CKPT does not write data blocks. SMON performs system cleanup/recovery duties; the PMON group handles process monitoring/cleanup. LREG registers services with the listener. The listener is distinct from SQL execution and is not the buffer cache. On Windows, Oracle background processes are implemented as threads within a process. [Process Architecture](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/process-architecture.html)

Oracle 19c supports CDB/PDB and traditional non-CDB configurations. A PDB is a portable collection of schemas and other objects, not a separately allocated SGA/instance. Schemas are namespaces/collections of database objects, not tablespaces or PDBs. [Introduction to Oracle Database](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/introduction-to-oracle-database.html)

Persistent structures include data files, control files and online redo logs. Logical tables and indexes are independent of physical file naming. [Physical Storage Structures](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/physical-storage-structures.html)

SQL is declarative. Parsing includes syntax/semantic work and cursor handling; optimization chooses an execution plan when needed. A shared compatible cursor can avoid a new hard parse. Execution and fetch are distinct; logical SQL clause order is not a promise about physical operator order. [SQL processing](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/sql.html)

## Time and version history

Oracle `DATE` includes time to whole seconds, with no zone or fractional seconds. `TIMESTAMP` adds fractional precision but no zone. `TIMESTAMP WITH TIME ZONE` carries zone information. `TIMESTAMP WITH LOCAL TIME ZONE` normalizes storage to the database zone and renders in the session zone; it does not preserve the original source zone. `DATE - DATE` returns days, possibly fractional; subtracting appropriate timestamps returns a day-to-second interval. Avoid premature `TRUNC`, implicit text conversions and mixed-type arithmetic that discards precision. [Oracle 19c Data Types](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/Data-Types.html)

Nepal local time for the modern teaching events is UTC+05:45. The IANA `Asia/Kathmandu` record shows this offset since 1986; historical Nepal offsets differ. [IANA Asia time-zone data](https://data.iana.org/time-zones/tzdb/asia)

Design proposal: retain the source timestamp/zone and receipt time; derive a canonical UTC instant; derive a Nepal-local calendar date for the date-dimension key. Use explicit half-open query intervals `[start, next_start)` instead of adding an artificial end-of-day time. A date key does not replace the event timestamp. Unknown source zone is a quality defect, not permission to invent one.

Oracle documents Type 1 overwrite, Type 2 new row/new surrogate key and Type 3 prior-value columns. A snowflake normalizes dimensions, adding joins. [Data Warehousing Logical Design](https://docs.oracle.com/en/database/oracle/oracle-database/19/dwhsg/data-warehouse-logical-design.html)

Design proposal: Type 2 versions use half-open validity `[valid_from, valid_to)`, with facts resolved using event time. Existing facts keep the historical surrogate key. A simple `UNIQUE(business_key, valid_from)` cannot prevent interval overlaps; use serialized ETL/transaction controls and an explicit overlap audit. A conditional unique index on a `CASE WHEN is_current = 'Y' THEN business_key END` expression can enforce at most one current row, but cannot establish contiguous history or prevent every overlap. Reconcile late-arriving events and retroactive corrections separately.

`MERGE` combines conditional insert/update, but it cannot update the same target row repeatedly in one statement. Canonicalize/deduplicate source rows first. An ordinary update-style MERGE does not automatically implement Type 2 close-and-insert history. [Oracle 19c MERGE](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/MERGE.html)

## Performance, access and analytical interpretation

Partition pruning uses relevant predicates to exclude partitions; inspect `PSTART`/`PSTOP` in a real plan. A conceptual partition animation does not prove pruning or a timing gain. Use predicates matching the partition-key data type and test the actual plan. [Oracle 19c Partition Pruning](https://docs.oracle.com/en/database/oracle/oracle-database/19/vldbg/partition-pruning.html)

Grant object privileges to narrowly scoped roles. `GRANT SELECT ON owner.view_name TO analyst_role` needs an existing role and a grantor with appropriate authority. Creating roles, users and cross-schema objects is a separate administrative step. [Oracle 19c GRANT](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/GRANT.html)

Analytical design proposals: pre-aggregate each fact to the intended comparison grain before combining facts; use `EXISTS` for cohort membership when row multiplication is unwanted. A diagnosis bridge produces one row per encounter/code, so joining it can multiply admissions and length-of-stay sums. `COUNT(DISTINCT encounter_key)` can fix a count but does not fix duplicated additive measures. Appointment, completed encounter and paid claim are different events. A dispensing line is not a prescription; supplied quantity and days are not adherence. Facility utilization rank is not a quality rank. Administrative-code cohorts require validated algorithms and coverage assessment. A balanced monthly panel must create zero months deliberately and separate zero observed use from missing source coverage.

## Validation boundary

Browser demonstrations are deterministic educational calculations over synthetic records. They do not connect to Oracle or evaluate arbitrary SQL. Documentation review and static syntax checks do not substitute for running scripts on the intended Oracle 19c release and privilege configuration. Production choices such as partitioning, encryption, audit retention, recovery objectives and optional features require environment-specific testing and entitlement review.
