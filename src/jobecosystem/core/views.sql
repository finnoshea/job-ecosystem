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
-- content appears under more than one row.
-- ---------------------------------------------------------------------------
CREATE VIEW jobs_reposted AS
SELECT j.*,
       (SELECT COUNT(*) - 1
          FROM jobs k
         WHERE k.content_hash = j.content_hash) AS duplicate_content_count
FROM jobs j
WHERE j.repost_count > 0
   OR EXISTS (
        SELECT 1
          FROM jobs k
         WHERE k.content_hash = j.content_hash
           AND k.id <> j.id
      )
ORDER BY j.repost_count DESC, j.last_seen_at DESC;

-- ---------------------------------------------------------------------------
-- jobs_stale: ghost-job candidates -- not seen in the last 14 days of scrape
-- activity, or never re-seen at all long after being posted. Kept separate
-- from jobs_reposted because "old" and "reposted" are different signals.
-- ---------------------------------------------------------------------------
CREATE VIEW jobs_stale AS
SELECT j.*,
       CASE
         WHEN a.anchor IS NULL THEN NULL
         ELSE CAST(
           julianday(a.anchor) - julianday(j.last_seen_at)
           AS INTEGER
         )
       END AS days_since_seen
FROM jobs j, scrape_anchor a
WHERE j.status <> 'applied'
  AND a.anchor IS NOT NULL
  AND (
        j.last_seen_at < datetime(a.anchor, '-14 days')
     OR (
          j.posted_at IS NOT NULL
          AND j.first_seen_at = j.last_seen_at          -- never re-seen since first sighting
          AND j.posted_at < datetime(a.anchor, '-60 days')
        )
      )
ORDER BY j.last_seen_at ASC;

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
