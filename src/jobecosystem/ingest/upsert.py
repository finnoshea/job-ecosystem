"""Writing scraped jobs into the database, with repost and ghost bookkeeping.

The schema enforces ``UNIQUE (source, external_id)``, but the constraint only
decides *whether* a row conflicts. This module decides what the conflict means:
whether a reappearing row is a repost, which fields changed, and how to tally
the run. That logic is why this is a module and not a bare ``ON CONFLICT``
clause in each scraper.

Ownership boundary: scrapers set the descriptive fields; this module owns the
``first_seen_at``, ``last_seen_at``, and ``repost_count`` columns of SQL table
``jobs``, and sets ``status`` on insert only. A scraper that sets them will have
its values overwritten.

Repost detection
----------------

A job counts as reposted when it reappears after being absent for at least
``REPOST_GAP_DAYS`` of scrape activity. The window is measured against the
newest ``last_seen_at`` already present in SQL table ``jobs``, not wall-clock
now, so a backfill or a catch-up run does not mark every row as a repost.

The gap matters because scrapers run daily but boards re-list constantly: a row
seen yesterday was simply still posted, while a row that vanished for three
weeks and came back is a genuine repeat listing. Both bump ``last_seen_at``;
only the second bumps ``repost_count``, which is what SQL view
``jobs_reposted`` selects on.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..core.models import JOB_COLUMNS, Job, UpsertResult
from ..core.db import utcnow

#: A job unseen for at least this many days of scrape activity, then seen again,
#: counts as reposted. Also used by ``views.jobs_stale`` in spirit; keep in mind
#: that changing this does not retroactively change ``repost_count``.
REPOST_GAP_DAYS = 14

#: Fields compared to populate ``UpsertResult.fields_changed``. Excludes the
#: bookkeeping columns this module owns, since those change on every run.
_COMPARABLE_FIELDS: tuple[str, ...] = (
    "company",
    "title",
    "location",
    "description",
    "url",
    "salary_min",
    "salary_max",
    "posted_at",
    "content_hash",
)

#: Fields a later description fetch is allowed to fill in. The scrape only sees
#: title/company/location, so a description arriving afterwards must be written
#: rather than treated as stale. ``content_hash`` accompanies it because the
#: hash is derived from the description.
_DESCRIPTION_FIELDS: tuple[str, ...] = ("description", "content_hash")


@dataclass(slots=True)
class EmbeddingWriteSummary:
    """Outcome of a batch embedding write.

    ``failed`` holds one ``(job_id, message)`` pair per rejected tuple, so a
    stale ``job_id`` is visible rather than silently dropped. The caller can
    log these alongside the run's other errors.
    """

    written: int = 0
    failed: list[tuple[int, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every tuple in the batch was written."""
        return not self.failed

    def __len__(self) -> int:
        return self.written + len(self.failed)


@dataclass(slots=True)
class UpsertSummary:
    """Tallies for one batch, matching the columns of SQL table ``scrape_runs``.

    ``inserted`` and ``updated`` here count *rows affected*, whereas the
    ``updated`` column of SQL table ``scrape_runs`` counts rows whose content
    actually changed. Both numbers are kept because they answer different
    questions: how much work the batch did versus how much of the board moved.
    """

    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    reposted: int = 0
    results: list[UpsertResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)


def upsert_jobs(
    conn: sqlite3.Connection,
    jobs: Iterable[Job],
) -> UpsertSummary:
    """Write a batch of scraped jobs, returning tallies.

    Runs inside a single transaction: either the whole batch lands or none of
    it does, which keeps SQL table ``scrape_runs``'s counts consistent with the
    rows they describe. Committing is the caller's job if it opened a
    transaction.

    The repost anchor is captured once here, before any row is written. Taking
    it per row would be wrong: each write sets ``last_seen_at`` to now, so the
    anchor would move as the batch progressed and every job after the first
    would look like a fresh sighting of its own row.
    """
    anchor = repost_anchor(conn)
    summary = UpsertSummary()
    with conn:
        for job in jobs:
            summary.fetched += 1
            result = upsert_job(conn, job, anchor=anchor)
            summary.results.append(result)
            if result.inserted:
                summary.inserted += 1
            elif result.fields_changed:
                summary.updated += 1
            else:
                summary.unchanged += 1
            if result.reposted:
                summary.reposted += 1
    return summary


def repost_anchor(conn: sqlite3.Connection) -> str | None:
    """The newest ``last_seen_at`` currently recorded, or ``None`` if empty.

    This is the reference point for repost decisions: a row is compared against
    where scrape activity had reached, not against wall-clock now, so a backfill
    or a catch-up run does not mark rows as reposts.
    """
    row = conn.execute("SELECT MAX(last_seen_at) AS anchor FROM jobs").fetchone()
    return row["anchor"] if row is not None else None


def upsert_job(
    conn: sqlite3.Connection,
    job: Job,
    *,
    anchor: str | None = None,
) -> UpsertResult:
    """Insert or refresh one job, applying the repost rules.

    Uses ``INSERT ... ON CONFLICT (source, external_id) DO UPDATE`` so a
    concurrent writer cannot slip a duplicate between the lookup and the write.

    ``anchor`` is the batch's repost reference point; pass the value from
    :func:`repost_anchor` when writing several jobs so they are all judged
    against the same starting state. When omitted it is read from the database,
    which is correct for a single-row write.
    """
    seen_at = utcnow()
    if anchor is None:
        anchor = repost_anchor(conn)
    existing = conn.execute(
        "SELECT * FROM jobs WHERE source = ? AND external_id = ?",
        (job.source, job.external_id),
    ).fetchone()

    if existing is None:
        job_id = _insert(conn, job, seen_at)
        return UpsertResult(job_id=job_id, inserted=True)

    return _refresh(conn, existing, job, seen_at, anchor)


def _insert(
    conn: sqlite3.Connection,
    job: Job,
    seen_at: str,
) -> int:
    """Insert a new row, stamping the bookkeeping fields this module owns."""
    values = job.to_dict()
    values["first_seen_at"] = seen_at
    values["last_seen_at"] = seen_at
    values["repost_count"] = 0
    values["status"] = "new"          # never trust a scraper's triage state
    values["rating"] = None           # ditto: triage only, never a scraper
    # content_hash stays NULL when there is no description: it is the marker for
    # "description not fetched yet", which SQL view jobs_needing_descriptions
    # selects on. A scraper that supplied one is trusted.
    values["description_fetched_at"] = (
        seen_at if values.get("content_hash") is not None else None
    )

    columns = ", ".join(values)
    placeholders = ", ".join("?" * len(values))
    cursor = conn.execute(
        f"INSERT INTO jobs ({columns}) VALUES ({placeholders})",
        list(values.values()),
    )
    return int(cursor.lastrowid or 0)


def _refresh(
    conn: sqlite3.Connection,
    existing: sqlite3.Row,
    job: Job,
    seen_at: str,
    anchor: str | None,
) -> UpsertResult:
    """Bump an existing row, deciding whether it counts as a repost."""
    reposted = _is_repost(conn, existing, seen_at, anchor)
    changed = tuple(
        name
        for name in _COMPARABLE_FIELDS
        if getattr(job, name) is not None and getattr(job, name) != existing[name]
    )

    assignments = ["last_seen_at = ?"]
    params: list[object] = [seen_at]

    for name in changed:
        assignments.append(f"{name} = ?")
        params.append(getattr(job, name))

    # A description fetched after the scraped listing must be written even
    # though no comparable field changed, and it is what makes the hash exist.
    newly_described = (
        existing["content_hash"] is None
        and job.description is not None
        and job.content_hash is not None
    )
    if newly_described:
        assignments.append("description_fetched_at = ?")
        params.append(seen_at)
        if "description" not in changed:
            assignments.append("description = ?")
            params.append(job.description)
        if "content_hash" not in changed:
            assignments.append("content_hash = ?")
            params.append(job.content_hash)
        changed = tuple(changed) + ("description_fetched_at",)

    if reposted:
        assignments.append("repost_count = repost_count + 1")

    params.extend([job.source, job.external_id])
    conn.execute(
        f"UPDATE jobs SET {', '.join(assignments)}"
        " WHERE source = ? AND external_id = ?",
        params,
    )
    return UpsertResult(
        job_id=int(existing["id"]),
        inserted=False,
        reposted=reposted,
        fields_changed=changed,
    )


def _is_repost(
    conn: sqlite3.Connection,
    existing: sqlite3.Row,
    seen_at: str,
    anchor: str | None,
) -> bool:
    """True when a row reappears after a long enough absence to be a repost.

    ``anchor`` is the batch's reference point, captured before any row in it was
    written; see :func:`upsert_jobs` for why it must not be read here.
    """
    if anchor is None:
        return False

    previously_seen = existing["last_seen_at"]
    if previously_seen is None:
        return False

    gap = conn.execute(
        "SELECT julianday(?) - julianday(?) AS days",
        (seen_at, previously_seen),
    ).fetchone()
    if gap is None or gap["days"] is None:
        return False

    # Measured against where scrape activity had reached, so a job seen in the
    # same run as everything else is never mistaken for a repost.
    anchor_gap = conn.execute(
        "SELECT julianday(?) - julianday(?) AS days", (seen_at, anchor)
    ).fetchone()
    same_run = bool(anchor_gap and anchor_gap["days"] <= 0)
    if same_run:
        return False

    return float(gap["days"]) >= REPOST_GAP_DAYS


def upsert_embeddings(
    conn: sqlite3.Connection,
    vectors: Sequence[tuple[int, bytes, int, str]],
) -> EmbeddingWriteSummary:
    """Store ``(job_id, vector, dim, model)`` rows, replacing any existing ones.

    Writes to SQL table ``job_embeddings``. Kept here rather than in
    ``ingest.embed`` so the write path has one owner.

    Per-row rather than ``executemany``: one stale ``job_id`` (a job deleted
    between embedding and write) fails its foreign key, and batching would
    discard every other embedding with it. Failed tuples are collected in the
    returned :class:`EmbeddingWriteSummary` instead of raising, so the caller
    can record them and still keep the successes.

    Each row is committed on its own via ``with conn``, so a failure cannot
    roll back earlier writes. This requires that the caller has already
    committed whatever transaction it had open -- typically the job write
    before this call. Otherwise the first row's ``COMMIT`` would commit the
    caller's pending work too, and a later rollback could not undo it.

    Setting ``conn.isolation_level = None`` (autocommit) is deliberately not
    used here: it mutates shared connection state for the duration of the call.
    """
    now = utcnow()
    summary = EmbeddingWriteSummary()

    for job_id, vector, dim, model in vectors:
        try:
            with conn:
                conn.execute(
                    "INSERT INTO job_embeddings"
                    " (job_id, vector, dim, model, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT (job_id) DO UPDATE SET"
                    "   vector = excluded.vector,"
                    "   dim = excluded.dim,"
                    "   model = excluded.model,"
                    "   updated_at = excluded.updated_at",
                    (job_id, vector, dim, model, now),
                )
        except sqlite3.Error as error:
            summary.failed.append((job_id, str(error)))
        else:
            summary.written += 1

    return summary


def record_run(
    conn: sqlite3.Connection,
    source: str,
    summary: UpsertSummary,
    *,
    started_at: str,
    errors: Sequence[tuple[str | None, str]] = (),
    finished_at: str | None = None,
) -> int:
    """Write a row in SQL table ``scrape_runs`` plus its SQL table
    ``scrape_errors`` children.

    Written even when the run failed, which is what makes a silently dead
    scraper visible instead of indistinguishable from a quiet board.
    Returns the new run id.
    """
    finished = finished_at or utcnow()
    with conn:
        cursor = conn.execute(
            "INSERT INTO scrape_runs"
            " (source, started_at, finished_at, fetched, inserted, updated, errors)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                source,
                started_at,
                finished,
                summary.fetched,
                summary.inserted,
                summary.updated,
                len(errors),
            ),
        )
        run_id = int(cursor.lastrowid or 0)

        if errors:
            conn.executemany(
                "INSERT INTO scrape_errors (run_id, tenant, error, occurred_at)"
                " VALUES (?, ?, ?, ?)",
                [(run_id, tenant, message, finished) for tenant, message in errors],
            )
    return run_id


def job_row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a row of SQL table ``jobs`` into a plain dict, for logging or
    diffing."""
    return {name: row[name] for name in JOB_COLUMNS if name in row.keys()}


def dumps_raw(payload: object) -> str | None:
    """Serialize a vendor payload for SQL table ``jobs``'s ``raw_json`` column.

    Best-effort: a payload that will not serialize must not fail the run, so
    this returns ``None`` rather than raising.
    """
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
