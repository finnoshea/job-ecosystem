"""Read and write access to the job store, for the TUI.

Every SQL statement the triage side needs lives here, so the TUI itself contains
none. That is the project's rule -- the interface is a leaf with no business
logic -- and it also means the queries are testable without a terminal.

Three kinds of function:

* **View readers** -- ``jobs_today``, ``jobs_unseen``, and friends, thin wrappers
  over the SQL views in ``core/views.sql``.
* **Search** -- :func:`search_jobs` (keyword, in SQL) and :func:`similar_jobs`
  (embedding, ranked in Python). Deliberately separate: keyword narrowing and
  "more like this" answer different questions, and merging them into one
  relevance score would be uninterpretable.
* **Writers** -- :func:`set_status` and :func:`set_rating`, the only mutations
  triage performs.

Filters
-------

``JobFilter`` is accepted by every reader. It is applied in Python for the view
wrappers (the views already define their own windows) and as SQL for search,
which needs to narrow before ranking. ``Jobs`` comes back as
:class:`~jobecosystem.core.models.Job`, which is a plain data carrier.

Similarity scope
----------------

:func:`similar_jobs` takes ``scope``: ``"filtered"`` ranks within the current
filter, ``"all"`` across the whole table. Both are wanted -- filtering keeps the
list consistent with what is on screen, while ``"all"`` answers "what else
anywhere looks like this".
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..core import similarity
from ..core.models import Job

#: Views the readers are built on. Listed so a typo fails here, not at runtime.
_VIEWS = (
    "jobs_today",
    "jobs_unseen",
    "jobs_reposted",
    "jobs_stale",
    "jobs_needing_descriptions",
    "jobs_with_embeddings",
)

Scope = Literal["filtered", "all"]

VALID_STATUSES = ("new", "seen", "applied", "hidden")


class QueryError(Exception):
    """A caller asked for something the store cannot do."""


@dataclass(frozen=True, slots=True)
class JobFilter:
    """Narrowing options shared by the readers and by search.

    All fields are optional and combine with AND. ``sources`` matches exactly
    (the labels in SQL table ``jobs``'s ``source`` column, e.g.
    ``ashby:ramp``), while ``source_prefixes`` matches the family
    (``ashby`` catches every Ashby board). ``rated_at_least`` is inclusive and
    excludes unrated rows, so ``rated_at_least=0`` means "has a rating".
    """

    sources: tuple[str, ...] = ()
    source_prefixes: tuple[str, ...] = ()
    companies: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    rated_at_least: int | None = None
    min_salary: int | None = None
    only_with_descriptions: bool = False

    def __post_init__(self) -> None:
        if self.rated_at_least is not None and not 0 <= self.rated_at_least <= 5:
            raise QueryError(
                f"rated_at_least must be 0-5, got {self.rated_at_least}"
            )
        unknown = set(self.statuses) - set(VALID_STATUSES)
        if unknown:
            raise QueryError(
                f"unknown status(es) {sorted(unknown)};"
                f" expected one of {list(VALID_STATUSES)}"
            )

    def where(self, *, alias: str = "j") -> tuple[str, list[Any]]:
        """Build a SQL ``WHERE`` fragment and its parameters.

        Written as a fragment rather than a single query so search and the view
        readers share one definition of what a filter means.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if self.sources:
            clauses.append(f"{alias}.source IN ({_placeholders(self.sources)})")
            params.extend(self.sources)
        for prefix in self.source_prefixes:
            clauses.append(f"{alias}.source LIKE ?")
            params.append(f"{prefix}%")
        if self.companies:
            clauses.append(f"{alias}.company IN ({_placeholders(self.companies)})")
            params.extend(self.companies)
        if self.statuses:
            clauses.append(f"{alias}.status IN ({_placeholders(self.statuses)})")
            params.extend(self.statuses)
        if self.rated_at_least is not None:
            clauses.append(f"{alias}.rating >= ?")
            params.append(self.rated_at_least)
        if self.min_salary is not None:
            # A posting with only a ceiling still qualifies if the ceiling
            # clears the bar; a posting with no figure does not.
            clauses.append(f"COALESCE({alias}.salary_max, {alias}.salary_min) >= ?")
            params.append(self.min_salary)
        if self.only_with_descriptions:
            clauses.append(f"{alias}.content_hash IS NOT NULL")

        if not clauses:
            return "", []
        return " AND ".join(clauses), params

    def matches(self, job: Job) -> bool:
        """Apply the same filter to an in-memory job.

        Used for the view readers, whose SQL is fixed by the view definition.
        Kept in step with :meth:`where` by the tests.
        """
        if self.sources and job.source not in self.sources:
            return False
        if self.source_prefixes and not any(
            job.source.startswith(prefix) for prefix in self.source_prefixes
        ):
            return False
        if self.companies and job.company not in self.companies:
            return False
        if self.statuses and job.status not in self.statuses:
            return False
        if self.rated_at_least is not None:
            if job.rating is None or job.rating < self.rated_at_least:
                return False
        if self.min_salary is not None:
            top = job.salary_max if job.salary_max is not None else job.salary_min
            if top is None or top < self.min_salary:
                return False
        if self.only_with_descriptions and job.content_hash is None:
            return False
        return True


def _placeholders(values: Sequence[Any]) -> str:
    return ", ".join("?" * len(values))


# ---------------------------------------------------------------------------
# view readers
# ---------------------------------------------------------------------------

def jobs_today(
    conn: sqlite3.Connection, *, filters: JobFilter | None = None, limit: int | None = None
) -> list[Job]:
    """Jobs first seen within the last day of scrape activity."""
    return _read_view(conn, "jobs_today", filters=filters, limit=limit)


def jobs_unseen(
    conn: sqlite3.Connection, *, filters: JobFilter | None = None, limit: int | None = None
) -> list[Job]:
    """Jobs not yet triaged -- the default list."""
    return _read_view(conn, "jobs_unseen", filters=filters, limit=limit)


def jobs_reposted(
    conn: sqlite3.Connection, *, filters: JobFilter | None = None, limit: int | None = None
) -> list[Job]:
    """Jobs seen more than once, or whose content appears under several rows.

    Rows without a description are excluded by the view itself: without a
    ``content_hash`` a repost cannot be distinguished from a same-titled
    opening.
    """
    return _read_view(conn, "jobs_reposted", filters=filters, limit=limit)


def jobs_stale(
    conn: sqlite3.Connection, *, filters: JobFilter | None = None, limit: int | None = None
) -> list[Job]:
    """Jobs not seen recently, or never re-seen long after posting."""
    return _read_view(conn, "jobs_stale", filters=filters, limit=limit)


def jobs_needing_descriptions(
    conn: sqlite3.Connection, *, filters: JobFilter | None = None, limit: int | None = None
) -> list[Job]:
    """The description work queue, newest first."""
    return _read_view(conn, "jobs_needing_descriptions", filters=filters, limit=limit)


def _read_view(
    conn: sqlite3.Connection,
    view: str,
    *,
    filters: JobFilter | None,
    limit: int | None,
) -> list[Job]:
    """Read one view, applying the filter in Python and the limit in SQL.

    The views carry their own ORDER BY, so the limit is pushed down to avoid
    materialising thousands of rows just to slice them. A filter cannot be
    pushed down (it would fight the view's own WHERE), so it is applied after.
    """
    if view not in _VIEWS:
        raise QueryError(f"unknown view {view!r}")
    if limit is not None and limit <= 0:
        return []

    # Over-fetch when filtering, since some rows will be discarded.
    fetch_limit = limit if filters is None or limit is None else None
    sql = f"SELECT * FROM {view}"
    params: list[Any] = []
    if fetch_limit is not None:
        sql += " LIMIT ?"
        params.append(fetch_limit)

    jobs = [Job.from_row(row) for row in conn.execute(sql, params)]
    if filters is not None:
        jobs = [job for job in jobs if filters.matches(job)]
    return jobs[:limit] if limit is not None else jobs


def get_job(conn: sqlite3.Connection, job_id: int) -> Job | None:
    """One job by id, or ``None``."""
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return Job.from_row(row) if row is not None else None


def get_jobs(conn: sqlite3.Connection, job_ids: Sequence[int]) -> list[Job]:
    """Several jobs by id, preserving the order of ``job_ids``.

    Ids that do not exist are skipped rather than raising, so a stale id in a
    cached list does not break a refresh.
    """
    if not job_ids:
        return []
    rows = conn.execute(
        f"SELECT * FROM jobs WHERE id IN ({_placeholders(job_ids)})",
        list(job_ids),
    ).fetchall()
    by_id = {row["id"]: Job.from_row(row) for row in rows}
    return [by_id[job_id] for job_id in job_ids if job_id in by_id]


# ---------------------------------------------------------------------------
# keyword search
# ---------------------------------------------------------------------------

def search_jobs(
    conn: sqlite3.Connection,
    text: str,
    *,
    filters: JobFilter | None = None,
    limit: int = 200,
    newest_first: bool = True,
) -> list[Job]:
    """Keyword search over title and description.

    A plain substring match over both columns, case-insensitive. FTS5 would rank
    better but needs a virtual table and sync triggers; at tens of thousands of
    rows this is fast enough and needs no machinery.

    Multi-word input is matched as separate terms, all of which must appear, for
    the same reason a search engine does it: "python remote" should not require
    that exact phrase in the text. Each term may appear in either column.

    Note that ``description`` is NULL until the paced fetch has run, so early on
    this effectively searches titles only. Callers may want to say so in the UI
    rather than presenting an empty result as "no matches".
    """
    terms = text.split()
    if not terms:
        return []

    clauses: list[str] = []
    params: list[Any] = []
    for term in terms:
        clauses.append(
            "(LOWER(COALESCE(j.title, '')) LIKE ?"
            " OR LOWER(COALESCE(j.description, '')) LIKE ?)"
        )
        pattern = f"%{term.lower()}%"
        params.extend([pattern, pattern])

    filter_sql, filter_params = (filters or JobFilter()).where(alias="j")
    if filter_sql:
        clauses.append(filter_sql)
        params.extend(filter_params)

    order = "j.first_seen_at DESC, j.id DESC" if newest_first else "j.id DESC"
    sql = (
        f"SELECT j.* FROM jobs j WHERE {' AND '.join(clauses)}"
        f" ORDER BY {order} LIMIT ?"
    )
    params.append(limit)

    return [Job.from_row(row) for row in conn.execute(sql, params)]


# ---------------------------------------------------------------------------
# embedding search
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SimilarJob:
    """A job and its cosine score against the query."""

    job: Job
    score: float


def similar_jobs(
    conn: sqlite3.Connection,
    query: str | Sequence[float],
    *,
    filters: JobFilter | None = None,
    scope: Scope = "filtered",
    limit: int = 50,
    min_score: float = 0.0,
    exclude_job_id: int | None = None,
    embed: Any = None,
) -> list[SimilarJob]:
    """Rank jobs by cosine similarity to ``query``.

    ``query`` is either text -- embedded here, one model call -- or an existing
    vector, which lets "more like this job" reuse a stored vector and skip the
    model entirely.

    ``scope`` decides the candidate set: ``"filtered"`` ranks within ``filters``
    so results agree with what is on screen, ``"all"`` ranks everything.
    ``exclude_job_id`` drops the job being compared against, which is otherwise
    its own best match.

    The embedder is injected rather than imported so that searching does not
    pull torch into the process unless a text query is actually used: pass a
    vector and this function never touches a model. Raises :class:`QueryError`
    if text is given without an embedder.
    """
    vector = _resolve_vector(query, embed=embed)
    if vector is None:
        return []

    where_sql, params = _similar_candidates_where(filters, scope, exclude_job_id)
    rows = conn.execute(
        f"SELECT e.job_id, e.vector, e.dim FROM jobs_with_embeddings e"
        f" JOIN jobs j ON j.id = e.job_id{where_sql}",
        params,
    ).fetchall()
    if not rows:
        return []

    candidates = [(row["job_id"], row["vector"]) for row in rows]
    dim = rows[0]["dim"]
    ranked = similarity.top_k(
        vector, candidates, k=limit, min_score=min_score, dim=dim
    )
    if not ranked:
        return []

    jobs = {job.id: job for job in get_jobs(conn, [job_id for job_id, _ in ranked])}
    return [
        SimilarJob(job=jobs[job_id], score=score)
        for job_id, score in ranked
        if job_id in jobs
    ]


def similar_to_job(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    filters: JobFilter | None = None,
    scope: Scope = "filtered",
    limit: int = 50,
    min_score: float = 0.0,
) -> list[SimilarJob]:
    """"More like this job", reusing the job's stored vector.

    Returns ``[]`` when the job has no embedding yet, rather than embedding the
    description text: the stored vector is what the rest of the table was
    compared against, so mixing the two would make scores incomparable.
    """
    row = conn.execute(
        "SELECT vector FROM job_embeddings WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        return []
    return similar_jobs(
        conn,
        similarity.decode(row["vector"]),
        filters=filters,
        scope=scope,
        limit=limit,
        min_score=min_score,
        exclude_job_id=job_id,
    )


def _resolve_vector(query: str | Sequence[float], *, embed: Any) -> list[float] | None:
    """Turn a query into a vector, embedding text when an embedder is given."""
    if isinstance(query, str):
        if embed is None:
            raise QueryError(
                "a text query needs an embedder; pass embed=embedder.embed_query"
                " (or a vector)"
            )
        if not query.strip():
            return None
        return list(embed(query))
    return list(query)


def _similar_candidates_where(
    filters: JobFilter | None,
    scope: Scope,
    exclude_job_id: int | None,
) -> tuple[str, list[Any]]:
    """SQL for the similarity candidate set."""
    if scope not in ("filtered", "all"):
        raise QueryError(f"scope must be 'filtered' or 'all', got {scope!r}")

    clauses: list[str] = []
    params: list[Any] = []

    # `all` deliberately ignores the filters, which is the point of the option.
    if scope == "filtered" and filters is not None:
        filter_sql, filter_params = filters.where(alias="j")
        if filter_sql:
            clauses.append(filter_sql)
            params.extend(filter_params)
    if exclude_job_id is not None:
        clauses.append("e.job_id <> ?")
        params.append(exclude_job_id)

    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# writers -- the only mutations triage performs
# ---------------------------------------------------------------------------

def set_status(conn: sqlite3.Connection, job_id: int, status: str) -> bool:
    """Set one job's triage status. Returns False when the id is unknown."""
    if status not in VALID_STATUSES:
        raise QueryError(
            f"invalid status {status!r}; expected one of {list(VALID_STATUSES)}"
        )
    with conn:
        cursor = conn.execute(
            "UPDATE jobs SET status = ? WHERE id = ?", (status, job_id)
        )
    return cursor.rowcount > 0


def set_rating(conn: sqlite3.Connection, job_id: int, rating: int | None) -> bool:
    """Set or clear one job's rating. Returns False when the id is unknown."""
    if rating is not None and not 0 <= rating <= 5:
        raise QueryError(f"rating must be 0-5 or None, got {rating!r}")
    with conn:
        cursor = conn.execute(
            "UPDATE jobs SET rating = ? WHERE id = ?", (rating, job_id)
        )
    return cursor.rowcount > 0


def set_status_many(
    conn: sqlite3.Connection, job_ids: Iterable[int], status: str
) -> int:
    """Set the status of several jobs. Returns the number changed."""
    ids = list(job_ids)
    if not ids:
        return 0
    if status not in VALID_STATUSES:
        raise QueryError(
            f"invalid status {status!r}; expected one of {list(VALID_STATUSES)}"
        )
    with conn:
        cursor = conn.execute(
            f"UPDATE jobs SET status = ? WHERE id IN ({_placeholders(ids)})",
            [status, *ids],
        )
    return cursor.rowcount


# ---------------------------------------------------------------------------
# summaries, for headers and footers
# ---------------------------------------------------------------------------

def counts_by_source(conn: sqlite3.Connection) -> dict[str, int]:
    """Job count per source label, biggest first."""
    rows = conn.execute(
        "SELECT source, COUNT(*) AS n FROM jobs GROUP BY source ORDER BY n DESC"
    ).fetchall()
    return {row["source"]: row["n"] for row in rows}


def counts_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    """Job count per triage status."""
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
    ).fetchall()
    return {row["status"]: row["n"] for row in rows}


def companies(conn: sqlite3.Connection, *, limit: int | None = None) -> list[str]:
    """Distinct company names, alphabetically, for a filter picker."""
    sql = "SELECT DISTINCT company FROM jobs ORDER BY company"
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [row[0] for row in conn.execute(sql, params)]


def sources(conn: sqlite3.Connection) -> list[str]:
    """Distinct source labels, alphabetically."""
    return [row[0] for row in conn.execute(
        "SELECT DISTINCT source FROM jobs ORDER BY source"
    )]


def run_summary(conn: sqlite3.Connection, *, limit: int = 20) -> list[dict[str, Any]]:
    """Recent runs from SQL view ``run_stats``, newest first."""
    rows = conn.execute(
        "SELECT * FROM run_stats ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(row) for row in rows]


def description_progress(conn: sqlite3.Connection) -> dict[str, Any]:
    """The single row of SQL view ``description_progress``."""
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    return dict(row) if row is not None else {}


def unresolved_count(conn: sqlite3.Connection) -> int:
    """Jobs still in the description queue, for badge display."""
    return conn.execute(
        "SELECT COUNT(*) FROM jobs_needing_descriptions"
    ).fetchone()[0]
