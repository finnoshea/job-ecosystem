-- Views over the jobs tables. Applied by core/db.py on every connect, so every
-- statement must be safe to re-run.
--
-- Views carry no state of their own, which means they are NOT part of the
-- versioned migration ledger: a changed view definition is picked up by the
-- next process to connect. That only works if re-applying the file always
-- produces the current text, so each view is dropped and recreated rather than
-- created with IF NOT EXISTS.
--
-- Windows are computed against the most recent scrape rather than "now", so a
-- stale database does not make every job look ancient. `scrape_anchor` below
-- is that reference point; it is the newest last_seen_at across the table.

DROP VIEW IF EXISTS scrape_anchor;
DROP VIEW IF EXISTS jobs_today;
DROP VIEW IF EXISTS jobs_unseen;
DROP VIEW IF EXISTS jobs_reposted;
DROP VIEW IF EXISTS jobs_stale;
DROP VIEW IF EXISTS jobs_with_embeddings;
DROP VIEW IF EXISTS jobs_needing_descriptions;
DROP VIEW IF EXISTS run_stats;
DROP VIEW IF EXISTS run_errors_by_source;
DROP VIEW IF EXISTS description_progress;

-- ---------------------------------------------------------------------------
-- scrape_anchor: single row holding the freshest last_seen_at. A view with no
-- rows (empty database) yields NULL arithmetic, which the consumers treat as
-- "no data" rather than an error.
-- ---------------------------------------------------------------------------
CREATE VIEW scrape_anchor AS
SELECT MAX(last_seen_at) AS anchor
FROM jobs;

-- ---------------------------------------------------------------------------
-- jobs_today: the daily review queue -- jobs first seen in the 24h window
-- before the newest scrape.
-- ---------------------------------------------------------------------------
CREATE VIEW jobs_today AS
SELECT j.*
FROM jobs j, scrape_anchor a
WHERE a.anchor IS NOT NULL
  AND j.first_seen_at > datetime(a.anchor, '-1 day')
ORDER BY j.posted_at DESC NULLS LAST, j.first_seen_at DESC;

-- ---------------------------------------------------------------------------
-- jobs_unseen: everything not yet triaged. This is the TUI's default filter.
-- ---------------------------------------------------------------------------
CREATE VIEW jobs_unseen AS
SELECT j.*
FROM jobs j
WHERE j.status = 'new'
ORDER BY j.last_seen_at DESC, j.id DESC;

-- ---------------------------------------------------------------------------
-- jobs_reposted: repeat-posting signal -- rows seen more than once, or whose
-- content appears under more than one row. Rows with a NULL content_hash are
-- excluded: the hash cannot be computed until the description is fetched, and
-- comparing titles alone would report every same-titled opening as a repost.
-- ---------------------------------------------------------------------------
CREATE VIEW jobs_reposted AS
SELECT j.*,
       (SELECT COUNT(*) - 1
          FROM jobs k
         WHERE k.content_hash = j.content_hash) AS duplicate_content_count
FROM jobs j
WHERE j.content_hash IS NOT NULL
  AND (
        j.repost_count > 0
     OR EXISTS (
          SELECT 1
            FROM jobs k
           WHERE k.content_hash = j.content_hash
             AND k.id <> j.id
        )
      )
ORDER BY j.repost_count DESC, j.last_seen_at DESC;

-- ---------------------------------------------------------------------------
-- jobs_stale: ghost-job candidates -- not seen in the last 14 days of scrape
-- activity, or never re-seen at all long after being posted. Kept separate
-- from jobs_reposted because "old" and "reposted" are different signals.
--
-- Written as a UNION of two single-index branches rather than one OR. SQLite
-- 3.45.1 stopped applying the "OR optimization" to the OR form and fell back to
-- a full index scan evaluating the predicate per row -- 20 seconds on a 14k-row
-- table, against 29 ms for the same database on 3.40.1, which still decomposed
-- the OR into two indexed searches. The UNION joins them explicitly, so the plan
-- no longer depends on the planner's mood. Both branches use an existing index:
-- idx_jobs_last_seen_at and idx_jobs_posted_at.
-- ---------------------------------------------------------------------------
CREATE VIEW jobs_stale AS
SELECT * FROM (
  SELECT j.*,
         CAST(
           julianday(a.anchor) - julianday(j.last_seen_at)
           AS INTEGER
         ) AS days_since_seen
  FROM jobs j, scrape_anchor a
  WHERE j.status <> 'applied'
    AND a.anchor IS NOT NULL
    AND j.last_seen_at < datetime(a.anchor, '-14 days')

  UNION ALL

  SELECT j.*,
         CAST(
           julianday(a.anchor) - julianday(j.last_seen_at)
           AS INTEGER
         ) AS days_since_seen
  FROM jobs j, scrape_anchor a
  WHERE j.status <> 'applied'
    AND a.anchor IS NOT NULL
    AND j.posted_at IS NOT NULL
    AND j.first_seen_at = j.last_seen_at          -- never re-seen since first sighting
    AND j.posted_at < datetime(a.anchor, '-60 days')
    -- Excluded so a row matching both branches is not returned twice.
    AND NOT (j.last_seen_at < datetime(a.anchor, '-14 days'))
)
-- Applied to the union, not to each branch: a LIMIT without an ORDER BY would
-- return arbitrary rows, and callers do pass a limit.
ORDER BY last_seen_at ASC;

-- ---------------------------------------------------------------------------
-- jobs_with_embeddings: the candidate set for similarity. Ranking itself
-- happens in Python (core/similarity.py) because SQLite cannot compute cosine
-- over BLOBs; this view just pairs vectors with their jobs.
-- ---------------------------------------------------------------------------
CREATE VIEW jobs_with_embeddings AS
SELECT j.id           AS job_id,
       j.source,
       j.company,
       j.title,
       j.location,
       j.status,
       j.last_seen_at,
       e.vector,
       e.dim,
       e.model,
       e.updated_at   AS embedded_at
FROM jobs j
JOIN job_embeddings e ON e.job_id = j.id;

-- ---------------------------------------------------------------------------
-- jobs_needing_descriptions: work queue for the paced description fetch.
-- A NULL content_hash means the row was stored from a listing only, so its
-- description has not been fetched yet. Newest first, and applied/seen rows are
-- deprioritized last so triage decisions accelerate the useful work.
-- ---------------------------------------------------------------------------
CREATE VIEW jobs_needing_descriptions AS
SELECT j.*
FROM jobs j
WHERE j.content_hash IS NULL
  AND j.status <> 'hidden'
ORDER BY j.status = 'new' DESC, j.first_seen_at DESC;

-- ---------------------------------------------------------------------------
-- run_stats: one row per scrape run, with the things you would otherwise
-- compute by hand. Intended to be queried from outside this codebase -- the
-- database is the durable record of what happened, and this view is its
-- reporting surface.
--
--   status: 'running' while finished_at is NULL. A crashed run also looks like
--           this, so treat a "running" row with a large minutes_since_start as
--           a crash rather than progress.
-- ---------------------------------------------------------------------------
CREATE VIEW run_stats AS
SELECT r.id,
       r.source,
       r.started_at,
       r.finished_at,
       CASE
         WHEN r.finished_at IS NULL THEN 'running'
         ELSE 'finished'
       END AS status,
       CASE
         WHEN r.finished_at IS NULL THEN NULL
         ELSE ROUND(
           (julianday(r.finished_at) - julianday(r.started_at)) * 86400.0,
           1
         )
       END AS duration_seconds,
       -- Always populated, including for an in-flight run: that is the row where
       -- "how long has this been going" is the whole question.
       ROUND(
         (julianday('now') - julianday(r.started_at)) * 1440.0,
         1
       ) AS minutes_since_start,
       r.fetched,
       r.inserted,
       r.updated,
       r.errors,
       CASE
         WHEN r.fetched > 0 THEN ROUND(100.0 * r.inserted / r.fetched, 1)
         ELSE NULL
       END AS inserted_percent
FROM scrape_runs r;

-- ---------------------------------------------------------------------------
-- run_errors_by_source: error totals rolled up per source, for spotting a
-- tenant that is consistently broken rather than transiently unlucky.
-- ---------------------------------------------------------------------------
CREATE VIEW run_errors_by_source AS
SELECT source,
       COUNT(*)                                   AS runs,
       SUM(errors)                                AS error_count,
       SUM(CASE WHEN errors > 0 THEN 1 ELSE 0 END) AS runs_with_errors,
       MAX(started_at)                             AS last_run_at
FROM scrape_runs
GROUP BY source;

-- ---------------------------------------------------------------------------
-- description_progress: how much of the description backlog is done.
--
-- Descriptions are fetched one paced request per job, so a full pass takes a
-- while; this is the view to poll while it runs. "pending" counts only rows
-- eligible for the queue (SQL view jobs_needing_descriptions), so it is the
-- number that will actually be worked on.
-- ---------------------------------------------------------------------------
CREATE VIEW description_progress AS
SELECT COUNT(*)                                            AS total_jobs,
       SUM(CASE WHEN content_hash IS NOT NULL THEN 1 ELSE 0 END)
                                                           AS described,
       SUM(CASE WHEN content_hash IS NULL THEN 1 ELSE 0 END)
                                                           AS pending,
       ROUND(
         100.0 * SUM(CASE WHEN content_hash IS NOT NULL THEN 1 ELSE 0 END)
         / MAX(COUNT(*), 1),
         1
       )                                                   AS described_percent,
       MIN(description_fetched_at)                         AS first_fetched_at,
       MAX(description_fetched_at)                         AS last_fetched_at
FROM jobs;
