-- Migration 3: index on first_seen_at.
--
-- jobs_today filters on first_seen_at and orders by it, but had no index on that
-- column. SQLite therefore scanned and sorted every matching row before applying
-- LIMIT: 590 ms to return 300 rows, rising to 1.8 s with an offset, on a
-- 14k-row table. With this index the same reads are 5-22 ms.
--
-- Plain ascending, not DESC: SQLite walks an ascending index backwards at the
-- same cost, and the simpler form is one less thing to explain.

CREATE INDEX IF NOT EXISTS idx_jobs_first_seen_at ON jobs (first_seen_at);
