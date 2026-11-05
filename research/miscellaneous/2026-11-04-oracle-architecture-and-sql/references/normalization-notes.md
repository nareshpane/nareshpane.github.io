# Normalization refinement: source and implementation notes

Added the normalization section directly to the monograph after dimensional modeling and before Oracle internals. The original introductory 1NF–BCNF material is intentionally retained by the page owner. Section ID is `normalization-design`; the interactive mount is `normalization-lab`. The four progression cards use `normalization-stages`.

## Scope and limitations

- `PrescriptionHeader`, `PrescriptionLine`, `Product`, and `Client` are proposed illustrative operational relations, not supplied DDL. The operational SQL is clearly labeled non-installed illustration. It represents prescription lines, not completed dispensing events.
- Actual warehouse SQL identifiers were checked against `sql/01_schema.sql`: `cb_dtl_ft_pharmacy_dispense`, `cb_dtl_dm_client`, `cb_dtl_dm_health_product`; joins use `client_key` and `product_key`, descriptions use `product_name`, client filter uses `client_id = 'CLIENT_SYN_001'`.
- Candidate-key definitions are dependency rules, not claims that a tiny dataset proves a normal form. The BCNF explanation requires examining all determinants; a surrogate key alone does not establish BCNF.
- Performance comparisons are conceptual. No measured Oracle timing, I/O, optimizer plan, storage size, or benchmark is reported. The separate browser lab simulates organization/consistency only. It projects the same 16 dispensing lines into operational-style tables without adding a prescribing dataset or inferring prescription times.
- Optional `cb_an_pharmacy_product_mv` is additional DDL displayed in the section, not added to default install scripts. It has not been executed on Oracle. It groups all periods by client version and product key, uses complete on-demand refresh, and deliberately avoids fast-refresh and query-rewrite promises. The final sample query selects client surrogate `1`, explicitly distinguished from all business-key versions.
- `SUM(total_cost)` omits null values and is described as recorded rather than complete expenditure. Quantities are grouped by product, not added across incompatible products. No clinical conclusions are implied.

## Oracle 19c primary references checked November 4, 2026

1. [Data Integrity](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/data-integrity.html) — primary, unique, and foreign-key constraints support identity and parent references; business rules extend beyond reference existence.
2. [Data Warehousing Logical Design](https://docs.oracle.com/en/database/oracle/oracle-database/19/dwhsg/data-warehouse-logical-design.html) — 3NF reduces redundant data manipulation and anomaly risks; analytical queries can need many joins; star schemas accept descriptive redundancy. The section does not convert those general observations into unconditional performance guarantees.
3. [Dimensions](https://docs.oracle.com/en/database/oracle/oracle-database/19/dwhsg/dimensions.html) — normalized and denormalized dimension hierarchies are both possible. Star dimensions can collect descriptive attributes while preserving fact-to-dimension relationships.
4. [Joins](https://docs.oracle.com/en/database/oracle/oracle-database/19/tgsql/joins.html) — access paths, join order, and join methods are optimizer decisions; estimated I/O and CPU costs depend on the row sources.
5. [Optimizer Access Paths](https://docs.oracle.com/en/database/oracle/oracle-database/19/tgsql/optimizer-access-paths.html) — index lookup is not always preferable; selectivity and the fraction of blocks required affect scan choice.
6. [Optimizer Statistics Concepts](https://docs.oracle.com/en/database/oracle/oracle-database/19/tgsql/optimizer-statistics-concepts.html) — object, column, index, and system statistics support cardinality and cost estimates.
7. [Partition Pruning](https://docs.oracle.com/en/database/oracle/oracle-database/19/vldbg/partition-pruning.html) — predicates may eliminate irrelevant partitions; functions and conversions can affect pruning. No partitioning speedup is quantified.
8. [Partitions, Views, and Other Schema Objects](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/partitions-views-and-other-schema-objects.html) — ordinary views versus stored materialized results; materialized views occupy storage and can store aggregates.
9. [CREATE MATERIALIZED VIEW](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/CREATE-MATERIALIZED-VIEW.html) — directly granted creation and source-access prerequisites, quota, immediate build, complete refresh, on-demand semantics. The section advises owning the source in a dedicated schema and explains qualified cross-schema access.
10. [Basic Materialized Views](https://docs.oracle.com/en/database/oracle/oracle-database/19/dwhsg/basic-materialized-views.html) — refresh methods/modes, aggregate materialized views, and the separate eligibility requirements for fast refresh.
11. [DBMS_MVIEW](https://docs.oracle.com/en/database/oracle/oracle-database/19/arpls/DBMS_MVIEW.html) — `REFRESH`, method `C`, `atomic_refresh`, invoker privileges, and capability inspection.

All source links are attached near the corresponding teaching material. The static section is original prose and example design, with no lengthy quoted source text.
