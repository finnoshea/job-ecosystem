-- Schema for the job-hunting ecosystem.
--
-- Applied by jobecosystem.core.db.migrate(). Every statement is idempotent
-- (IF NOT EXISTS), so re-running is safe. Table order matters only for the
-- foreign keys; jobs must exist before job_embeddings and scrape_errors.
--
-- Conventions:
--   * all timestamps are ISO-8601 UTC TEXT: YYYY-MM-DDTHH:MM:SSZ
--   * all primary keys are INTEGER (SQLite rowid aliases)
--   * vectors are stored as little-endian float32 BLOBs, decoded in Python

-- ---------------------------------------------------------------------------
-- jobs: the canonical, deduplicated job rows. One row per (source, external_id).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY,
    source        TEXT    NOT NULL,                  -- 'ashby' | 'workday' | ...
    external_id   TEXT    NOT NULL,                  -- the vendor's own job id
    company       TEXT    NOT NULL,
    title         TEXT    NOT NULL,
    location      TEXT,
    description   TEXT,                              -- plain text: embedding + keyword input
    url           TEXT,
    salary_min    INTEGER,                           -- annual USD
    salary_max    INTEGER,                           -- annual USD; absent when only a floor is given
    posted_at     TEXT,                              -- from the board; often absent or wrong
    first_seen_at TEXT    NOT NULL,                  -- when *we* first scraped it
    last_seen_at  TEXT    NOT NULL,                  -- bumped every scrape -> ghost detection
    description_fetched_at TEXT,                     -- when the description was fetched; NULL = only the listing is stored
    description_url TEXT,                             -- where to fetch the full text, for sources that need a second request
    -- description_attempts (retry counter) is added by migration 004, not
    -- listed here: a fresh database runs every migration after this one, and
    -- ALTER TABLE ADD COLUMN has no IF NOT EXISTS to make that idempotent.
    repost_count  INTEGER NOT NULL DEFAULT 0
                  CHECK (repost_count >= 0),         -- bumped on reappearance / reappearance after a gap
    status        TEXT    NOT NULL DEFAULT 'new'     -- triage state, never set by scrapers
                  CHECK (status IN ('new', 'seen', 'applied', 'hidden')),
    rating        INTEGER                            -- triage judgment, 0-5; NULL = unrated
                  CHECK (rating IS NULL OR (rating >= 0 AND rating <= 5)),
    content_hash  TEXT,                              -- hash(title+company+description); NULL until the description arrives
    raw_json      TEXT,                              -- original vendor payload, for debugging parsers
    UNIQUE (source, external_id)                     -- upsert target: dedup key
);

CREATE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs (content_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_company      ON jobs (company);
CREATE INDEX IF NOT EXISTS idx_jobs_status       ON jobs (status);
-- Composite, matching how the triage views are actually read: filter on status,
-- order by last_seen_at. Without it SQLite uses idx_jobs_status, collects every
-- matching row into a temp B-tree to sort, and only then applies the LIMIT --
-- seconds of work at ~15k rows to return 300.
CREATE INDEX IF NOT EXISTS idx_jobs_status_seen
    ON jobs (status, last_seen_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_rating       ON jobs (rating);
CREATE INDEX IF NOT EXISTS idx_jobs_last_seen_at ON jobs (last_seen_at);
-- Journals the "today" window filters and sorts on; see migration 3.
CREATE INDEX IF NOT EXISTS idx_jobs_first_seen_at ON jobs (first_seen_at);
CREATE INDEX IF NOT EXISTS idx_jobs_posted_at    ON jobs (posted_at);
CREATE INDEX IF NOT EXISTS idx_jobs_salary_min   ON jobs (salary_min);
-- Partial: rows without a description are excluded, so the "needs fetching"
-- sweep stays cheap as the table grows.
CREATE INDEX IF NOT EXISTS idx_jobs_needs_description
    ON jobs (id) WHERE content_hash IS NULL;
-- ---------------------------------------------------------------------------
-- job_embeddings: one row per embedded job. Vectors live here rather than on
-- jobs so a job row stays cheap to read and can exist before it is embedded.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_embeddings (
    job_id     INTEGER PRIMARY KEY
               REFERENCES jobs (id) ON DELETE CASCADE,
    vector     BLOB    NOT NULL,                     -- array('f').tobytes(); cosine computed in Python
    dim        INTEGER NOT NULL,                     -- 768 for nomic-embed-text-v1.5
    model      TEXT    NOT NULL,                     -- enables re-embed-everything on model change
    updated_at TEXT    NOT NULL
);

-- ---------------------------------------------------------------------------
-- scrape_runs: one row per source per execution, written even on failure.
-- Summary tallies only; per-tenant detail belongs in scrape_errors.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scrape_runs (
    id          INTEGER PRIMARY KEY,
    source      TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    finished_at TEXT,                                -- NULL while a run is in flight
    fetched     INTEGER NOT NULL DEFAULT 0,
    inserted    INTEGER NOT NULL DEFAULT 0,
    updated     INTEGER NOT NULL DEFAULT 0,
    errors      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_scrape_runs_source_started
    ON scrape_runs (source, started_at);

-- ---------------------------------------------------------------------------
-- scrape_errors: failure detail, so one dead Workday tenant cannot hide inside
-- a count. tenant is nullable for sources that have no tenant concept.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scrape_errors (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL
                REFERENCES scrape_runs (id) ON DELETE CASCADE,
    tenant      TEXT,
    error       TEXT    NOT NULL,
    occurred_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scrape_errors_run ON scrape_errors (run_id);
