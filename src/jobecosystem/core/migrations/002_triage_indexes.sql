-- Migration 2: composite index for the triage reads.
--
-- The triage views filter on status and order by last_seen_at. SQLite would
-- satisfy that with idx_jobs_status, then sort every matching row in a temp
-- B-tree before applying LIMIT -- 2.2 seconds at ~14k rows to return 300.
-- An index on (status, last_seen_at DESC, id DESC) lets it walk in order and
-- stop early, which measures at 3.4 ms.
--
-- IF NOT EXISTS so this is safe on a database created from the current
-- schema.sql, where the index already exists.

CREATE INDEX IF NOT EXISTS idx_jobs_status_seen
    ON jobs (status, last_seen_at DESC, id DESC);
