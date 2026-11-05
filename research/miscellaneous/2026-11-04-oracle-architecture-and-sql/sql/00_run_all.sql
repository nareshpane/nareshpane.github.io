-- SQL Developer: open this file and Run Script (F5), using an EMPTY dedicated
-- educational schema with CREATE TABLE, CREATE VIEW and tablespace quota.
-- A table owner can index their own table; CREATE INDEX is not a system grant.
-- SQL*Plus / SQLcl: @00_run_all.sql from this directory.
-- Oracle DDL commits implicitly. These scripts are one-time setup, not a reset.
-- They contain no DROP statements and never provision users or grant privileges.
WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK
SET DEFINE OFF
@@01_schema.sql
@@02_seed.sql
@@03_views.sql
@@04_transformations.sql
@@05_aggregates.sql
@@06_analytics.sql
@@07_analyst_exercises.sql
