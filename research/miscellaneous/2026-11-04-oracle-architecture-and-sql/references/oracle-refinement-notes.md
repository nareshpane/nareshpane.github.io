# Oracle 19c refinement checks

Checked November 4, 2026 against official Oracle 19c documentation. This is a read-only technical review of the existing Oracle and pipeline teaching material; it does not establish that any SQL was executed on Oracle.

## Findings and wording recommendations

### Database and instance

Retain the existing distinction. Suggested intuitive introduction: **The database is the persistent collection of files; the instance is the running machinery that manages and serves them.** An instance can be stopped without deleting the files. Shared SGA memory and background processes belong in the instance boundary, with server processes and their private memory shown as associated running components. The single-instance diagram should not imply that RAC or a PDB creates a separate independent copy of the same architecture for every schema.

Evidence: [Database Concepts — Oracle Database Instance](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/oracle-database-instance.html), Introduction, Database Instance Structure, and Database Instance Configurations.

### Shared and private memory

Expand first-use acronyms: **System Global Area (SGA)** and **Program Global Area (PGA)**. The SGA is shared; each server or background process has private PGA memory. Work areas support operations such as sorting and hashing. UGA session state is in the PGA for dedicated servers and in shared memory for shared servers. The current paragraph already captures this distinction accurately; avoid simplifying PGA into a universal per-session box.

Evidence: [Database Concepts — Memory Architecture](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/memory-architecture.html), Basic Memory Structures and Overview of the User Global Area.

### Process responsibilities

Use these concise descriptions:

- **Server process:** handles a connected client's requests.
- **Database Writer (DBWn):** writes dirty buffer-cache blocks to data files.
- **Log Writer (LGWR):** persists redo in online redo logs.
- **Checkpoint (CKPT):** records checkpoint information and signals DBWn; it does not write data blocks or redo blocks.
- **System Monitor (SMON):** performs instance recovery when required at startup and other system cleanup.
- **Process Monitor (PMON):** detects abnormal termination and delegates cleanup to CLMN; CLMN can use CLnn helpers.

The existing combined PMON/cleanup row is broadly accurate but benefits from the explicit delegation. Do not describe PMON alone as doing all cleanup. On Windows, these named Oracle processes can be implemented as threads. Required redo must reach storage before DBWn writes the corresponding dirty buffers.

Evidence: [Database Concepts — Process Architecture](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/process-architecture.html), PMON Group, SMON, DBW, LGWR, CKPT, and Introduction to Processes. [Database Reference — Background Processes](https://docs.oracle.com/en/database/oracle/oracle-database/19/refrn/background-processes.html) independently specifies CLMN/CLnn. The verified 19c behavior is sufficient; do not assert a particular 12c introduction release without an additional version-specific source. Listener registration belongs to LREG in 19c.

### Commit and independent data-file writes

Suggested label: **Redo durability and independent data-file writes**. The existing title's “later” can imply DBWn only works after commit. Oracle explicitly permits uncommitted changes in data files while some committed changes exist only in redo. Therefore describe DBWn timing as **before or after commit**, independently of acknowledging that transaction. Instance recovery reapplies redo and reverses uncommitted work.

Evidence: [Database Concepts — Oracle Database Instance](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/oracle-database-instance.html), Purpose of Instance Recovery.

For the illustrated ordinary synchronous case, explicitly state **COMMIT WRITE WAIT IMMEDIATE**: success returns after required redo is durable in online redo logs. It does not require all changed data blocks to be written. Session/system commit settings can change omitted-clause behavior; NOWAIT has different durability acknowledgment semantics. A lost success message can leave the client uncertain even though the transaction committed.

Evidence: [SQL Language Reference — COMMIT](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/COMMIT.html), WRITE, WAIT/NOWAIT, and IMMEDIATE/BATCH clauses. Retain the page's configured-storage/recovery qualification.

### Parsing and SELECT processing

The existing prose is accurate, but the unconditional Parse → Optimize animation is misleading. Label optimization **Hard parse only**, or add a path that bypasses it for reusable code. A hard parse builds a new executable version; a soft parse reuses suitable existing code and skips optimization and row-source generation. Identical text alone does not ensure reuse: object meaning and optimizer environment also matter. Reuse of a plan is not reuse of previously returned result rows. Execution and fetching still perform work, and access may use cached blocks or physical reads.

Evidence: [SQL Tuning Guide — SQL Processing](https://docs.oracle.com/en/database/oracle/oracle-database/19/tgsql/sql-processing.html), Shared Pool Check and SQL Execution; [Database Concepts — SQL](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/sql.html), Stages of SQL Processing.

### ETL, ELT, and dependable replay

Expand **Extract, Transform, Load (ETL)** and **Extract, Load, Transform (ELT)**. Introduce them as where/when transformation takes place; validation, transport, orchestration, lineage, and reconciliation remain necessary in either approach. ELT is not permission to publish unchecked raw submissions. The page's current mixed staging/warehouse description is appropriate.

Evidence: [Data Warehousing Guide — ETL Overview](https://docs.oracle.com/en/database/oracle/oracle-database/19/dwhsg/etl-overview.html), which describes ETL as a broader process than three rigid stages; [VLDB Guide — Parallel Execution Tips](https://docs.oracle.com/en/database/oracle/oracle-database/19/vldbg/parallel-exec-tips.html), explicit ETL/ELT terminology.

The following are **HealthDataNepal design recommendations**, rather than automatic Oracle guarantees: identify an event by source system plus stable event/line identity; identify versions separately; reject ambiguous conflicting versions; apply deterministic precedence; preserve provenance; make fact publication and load-control advancement recoverable together. Replay of an identical accepted version adds zero facts. A correction changes the intended state under explicit rules; it is not a new clinical event. Reconcile mutually exclusive received outcomes and measures, not just totals of rows. Existing text already supports these principles.

`MERGE` can implement part of an upsert but is not sufficient for these guarantees: one target row cannot be updated multiple times in the same statement, so resolve duplicate source matches first. Version precedence, deletion rules, concurrency, and history remain explicit design responsibilities.

Evidence: [SQL Language Reference — MERGE](https://docs.oracle.com/en/database/oracle/oracle-database/19/sqlrf/MERGE.html). See also [Database Concepts — Transactions](https://docs.oracle.com/en/database/oracle/oracle-database/19/cncpt/transactions.html), Transaction Guard and Lost Commit Messages, for the separate problem of an uncertain client commit outcome.
