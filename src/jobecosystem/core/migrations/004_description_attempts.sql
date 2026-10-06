-- Migration 4: per-row description fetch attempt counter.
--
-- The description queue retried failed fetches forever, so a permanently dead
-- posting (a 403 or 404) was re-requested on every run. This column lets the
-- queue cap retries. It is incremented before each request, so an interrupted
-- or crashed run still consumes the attempt.
--
-- Rows start at 0, which is also what makes ADD COLUMN legal on a NOT NULL
-- column. The definition matches schema.sql, so a fresh database and a migrated
-- one end up with the same shape.

ALTER TABLE jobs ADD COLUMN description_attempts INTEGER NOT NULL DEFAULT 0
    CHECK (description_attempts >= 0);
