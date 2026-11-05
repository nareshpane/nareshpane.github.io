# Oracle Database Architecture and SQL: Designing HealthDataNepal

An original, local-first research monograph using a fictional national health warehouse. All identities, organizations and transactions are synthetic. Real Nepal province/district labels are contextual; municipality and ward examples are illustrative.

Open `../2026-11-04-oracle-architecture-and-sql.html` directly, or serve the repository root and visit `/research/miscellaneous/2026-11-04-oracle-architecture-and-sql.html`. No build step, CDN, database connection or external runtime service is required. JSON has a matching checked-in JavaScript wrapper so file-based browsing does not depend on fetch permissions.

## Contents

Sixteen chapters cover relational foundations, normalization, keys, national integration architecture, dimensional grain/history, Oracle internals, engineering, beginning SQL/SQL Developer, aggregates, analytics, twelve practical exercises, exact time, Canadian comparisons, governance and reproducibility. References conclude the text. Sixteen required visual concepts are covered by twenty laboratory mounts, including additional transaction, service-window and data-quality demonstrations, plus original static diagrams.

## Asset organization

- `css/monograph.css` and `css/refinements.css`: preserved base presentation plus bounded editorial refinements, responsive layouts and reduced-motion rules.
- `js/monograph.js`: local simulations, keyboard-operable controls, compact navigation and system reduced-motion support.
- `svg/`: redesigned self-contained architecture/workload overviews; `svg/illustrations/` contains 50 original coordinated vector drawings. The detailed interactive star is generated as inline SVG.
- `data/sample.json` and `sample.js`: matching deterministic records for 14 related tables.
- `data/exercises.json` and `exercises.js`: matching twelve query exercises, documented result grains and expected outputs.
- `data/build_samples.py`, `build_exercises.py`, `verify_samples.py`: deterministic generation and independent numeric/integrity checks.
- `sql/01_schema.sql` through `07_analyst_exercises.sql`: dependency-ordered Oracle 19c educational examples; read `sql/README.md` before execution.
- `references/`: authoritative source register and technical scope notes. No third-party documentation is republished.
- `validation/`: static validation tools, original SVG generator and a report of checks actually performed.

## Reproduce numerical checks

From the monograph asset directory:

```powershell
python data/verify_samples.py
node --check js/monograph.js
python validation/check_page.py
```

The dataset includes 16 pharmacy lines, four inpatient stays, six diagnosis associations, four physician visits and four ED attendances. Fifteen known pharmacy costs total NPR 1,610. The exact first stay lasts 54.508333 hours. A diagnosis join expands four stay rows to six, illustrating why event counts and sums require cardinality-aware queries.

The appointment timestamp is not the completed encounter timestamp. `emergency.waiting_minutes` is arrival-to-triage in this fixture, rather than a validated arrival-to-first-clinician wait. Pharmacy geography describes dispensing location; client residence is a separate role. Fiscal labels are an explicit teaching convention, not a Nepali fiscal-calendar implementation.

## SQL scope and limitations

No Oracle instance was available. The queries are statically reviewed against Oracle 19c references; actual parsing, privileges, optimizer plans, licensing/feature availability and runtime execution remain unverified. Use an empty, disposable authorized educational schema. The core scripts do not provision users or security. Five production schemas, SCD2 architecture and monthly partitioning are separately labelled design illustrations. The completed-stay table requires non-NULL discharge values; an open-stay production design needs a different constraint and duration policy.

`01` defines dimensions before facts and the diagnosis bridge. `02` inserts synthetic data. `03` creates views. `04` demonstrates staging, validation, deduplication, an insert-only MERGE and isolated Type 2 history. `05` and `06` contain aggregate/analytic queries; `07` supplies the exercises. Replaying CREATE statements in the same schema will fail; the *load logic's* idempotence does not mean all DDL scripts are rerunnable.

The browser's constrained selections generate query text, but never execute it. It does not accept arbitrary SQL or connect to real records. Canonical UTC and Nepal-local display are distinguished explicitly.

## Sources and editorial metadata

Primary technical sources are Oracle Database 19c SQL Language Reference, Database Concepts and Data Warehousing Guide. The requested 18c aggregate and 21c analytic references are supplemental. Institutional comparisons use public BC Ministry, ICES, Alberta and CIHI pages; Nepal's offset is verified through IANA. See `references/sources.json` for retrieval and claim scope. Normalization references are detailed in `references/normalization-notes.md`; the later Oracle/ETL review is recorded in `references/oracle-refinement-notes.md`.

Editorial creation date: November 4, 2026 (America/Los_Angeles). User-specified display metadata is Codex on Windows, GPT-6.1 Sol, High; no runtime model-detection feature is implemented.

No changes were staged, committed, pushed or deployed.

## Visual and pedagogical refinement

The later normalization/denormalization section retains the earlier normal-form introduction. Its isolated browser lab compares normalized tables, a flat redundant extract, the existing star and a key-based product summary. Buttons demonstrate a one-copy update anomaly, a governed catalog correction and explicit extract rebuild, then reset. The full output and matching Oracle query can be inspected in each view. Structural counts are illustrative; no Oracle timing or storage benchmark is claimed.

Client IDs follow `CLIENT_SYN_001` through `CLIENT_SYN_004`; dispensing source IDs follow `DISPENSE_SYN_001` through `DISPENSE_SYN_016`. Optional encounter labels begin with `VISIT_SYN_001`, `EMERGENCY_SYN_001` and `HOSPITAL_SYN_001`. Numeric surrogate and foreign keys, product codes, transaction/prescription IDs, timestamps, quantities and costs remain unchanged. The isolated Type 2 teaching identity is `CLIENT_SYN_005`.

Run `python validation/build_illustrations.py` then `python validation/build_diagrams.py` to regenerate the original vectors. Clinical scenes and technical symbols were drawn from first principles using the supplied screenshots as conceptual references; no screenshot pixels were copied into page assets.

## Final teaching and layout pass

The first title line retains its size; the italic line is 6pt smaller. Chapter heading sizes use one shared 5pt reduction token applied to their existing responsive bases. The subtitle spans the full header row and fits on one line at 1440px. Metadata is 2pt larger in a compact four-row green box. SQL blocks have 26px desktop insets (10px more per side), retaining 4px mobile insets.

The opening chapter now has its own original database/SQL ecosystem illustration. Six dimension cards, five query stages, five transaction stages and six ETL stages use new explanatory mini-scenes. The Oracle component explorer separates shared SGA, private process PGA, background work and persistent files, with a synchronized accessible selector and diagram buttons. Conditional hard parsing, independent DBWn timing and ordinary synchronous redo durability are explicit.

The six patient cards retain independent blue, peach, rose, lavender, ivory and gray identities when active. Their spring-green explanation panel remains distinct. The national diagram measures the source and destination cards and redraws four converging SVG branches plus one final arrow on resize; the two-column mobile source layout routes upper sources through the central gutter.

Reading-progress and the visible Reduce Motion control were removed. Sticky navigation, anchors, active highlighting and expand/collapse remain. Operating-system reduced-motion preferences still control CSS animations and stop an active player when the preference changes.

The ETL teaching ledger reuses one dispensing line without modifying the extract. An unchanged accepted replay adds a duplicate receipt and zero facts; unknown-code receipts remain rejected. Choosing a stage inspects the same run without resubmitting it. Production correction/versioning is explained, not implemented by this tiny fixed-payload ledger. Source update and ingestion timestamps are not present in the fixture and are explicitly labeled as not supplied rather than invented.

Run `python validation/build_teaching_illustrations.py` to regenerate the 26 additional illustrations (25 compact scenes plus the larger opening ecosystem). The prior 24-illustration generator remains separate. Final QA evidence is recorded in `validation/final-refinements.md` and its screenshots. The research index consolidates the existing Oracle entry to first position, CPI second, preserving the remaining entries' relative order and content. Nothing was staged, committed, pushed or deployed.
