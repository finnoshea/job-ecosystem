"""Dataclasses mirroring the SQLite rows.

These are plain data carriers: no database access, no business logic. Modules
convert rows to and from these types at their own boundary, which keeps the
schema in one place (``schema.sql``) and the field names in two: here and SQL.

Timestamps are ISO-8601 UTC strings, matching what ``db.utcnow()`` produces and
what every timestamp column stores. They are *not* ``datetime`` objects, so
string comparison and SQLite's date functions stay consistent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from sqlite3 import Row
from typing import Any, Literal, Mapping, Self

# Values allowed by the jobs.status CHECK constraint.
JobStatus = Literal["new", "seen", "applied", "hidden"]

# The columns of `jobs`, in schema order. Used to build INSERT/UPDATE statements
# so the column list cannot drift from the dataclass.
JOB_COLUMNS: tuple[str, ...] = (
    "id",
    "source",
    "external_id",
    "company",
    "title",
    "location",
    "description",
    "url",
    "salary_min",
    "salary_max",
    "posted_at",
    "first_seen_at",
    "last_seen_at",
    "description_fetched_at",
    "description_url",
    "repost_count",
    "status",
    "rating",
    "content_hash",
    "raw_json",
)


def content_hash(
    title: str,
    company: str,
    description: str | None,
) -> str:
    """Stable hash of the fields that identify a posting's content.

    Used as the cross-source repost net: the same job re-listed under a new
    ``external_id`` (or on a different board) yields the same hash, so
    ``views.jobs_reposted`` can spot it.

    Whitespace and case are normalized because boards re-render the same text
    differently: whitespace runs (spaces, tabs, newlines, non-breaking spaces)
    collapse to a single space before hashing, so ``"Senior  Engineer"`` and
    ``"Senior Engineer"`` agree. ``description`` is included because two
    openings at one company often share a title.
    """
    normalized = " ".join(
        " ".join(part.replace("\u00a0", " ").lower().split())
        for part in (title, company, description or "")
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Job:
    """One row of ``jobs``: a deduplicated posting.

    ``id`` is ``None`` until the row is inserted. Scrapers construct these from
    vendor payloads and leave the bookkeeping fields (``first_seen_at``,
    ``last_seen_at``, ``repost_count``, ``status``) to ``ingest.upsert``, which
    owns them.
    """

    source: str
    external_id: str
    company: str
    title: str
    location: str | None = None
    description: str | None = None
    url: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    posted_at: str | None = None
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    description_fetched_at: str | None = None
    description_url: str | None = None
    repost_count: int = 0
    status: JobStatus = "new"
    rating: int | None = None
    content_hash: str | None = None
    raw_json: str | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        """Fill in ``content_hash`` when a description is available.

        Left ``None`` when there is no description: hashing title and company
        alone would collide across distinct openings that share a title, making
        SQL view ``jobs_reposted`` report false positives. ``content_hash`` is
        therefore a completeness marker as well as a fingerprint -- NULL means
        "description not fetched yet".
        """
        if self.content_hash is None and self.description is not None:
            self.content_hash = content_hash(self.title, self.company, self.description)

    @classmethod
    def from_row(cls, row: Row | Mapping[str, Any]) -> Self:
        """Build a :class:`Job` from a ``sqlite3.Row`` or a plain mapping.

        Unknown keys are ignored so a row from a joined view still maps cleanly.
        """
        columns = set(JOB_COLUMNS)
        return cls(**{key: row[key] for key in row.keys() if key in columns})

    def to_dict(self) -> dict[str, Any]:
        """Return the row's column values, excluding an unset ``id``."""
        data = {name: getattr(self, name) for name in JOB_COLUMNS}
        if data["id"] is None:
            data.pop("id")
        return data

    @property
    def has_description(self) -> bool:
        """True when the full posting text has been fetched."""
        return self.description is not None and self.content_hash is not None

    @property
    def salary_range(self) -> str:
        """Human-readable salary, e.g. ``$120k-$160k``, ``$120k+``, or ``''``."""
        if self.salary_min is None and self.salary_max is None:
            return ""
        low = f"${self.salary_min // 1000}k" if self.salary_min is not None else ""
        high = f"${self.salary_max // 1000}k" if self.salary_max is not None else ""
        if low and high:
            return f"{low}-{high}"
        return f"{low or high}+"


@dataclass(slots=True)
class Embedding:
    """One row of ``job_embeddings``: the vector for a job.

    ``vector`` is the float32 BLOB form from ``core.similarity.encode``;
    it is not decoded here so large scans can stay in SQLite's domain.
    """

    job_id: int
    vector: bytes
    dim: int
    model: str
    updated_at: str

    @classmethod
    def from_row(cls, row: Row | Mapping[str, Any]) -> Self:
        """Build an :class:`Embedding` from a database row."""
        return cls(
            job_id=row["job_id"],
            vector=row["vector"],
            dim=row["dim"],
            model=row["model"],
            updated_at=row["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the row's column values."""
        return {
            "job_id": self.job_id,
            "vector": self.vector,
            "dim": self.dim,
            "model": self.model,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class ScrapeRun:
    """One row of ``scrape_runs``: the tally for a single source execution.

    Written even when the run fails, which is what makes silent scraper death
    visible. ``finished_at`` stays ``None`` while a run is in flight.
    """

    source: str
    started_at: str
    finished_at: str | None = None
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    errors: int = 0
    id: int | None = None

    @classmethod
    def from_row(cls, row: Row | Mapping[str, Any]) -> Self:
        """Build a :class:`ScrapeRun` from a database row."""
        return cls(
            id=row["id"],
            source=row["source"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            fetched=row["fetched"],
            inserted=row["inserted"],
            updated=row["updated"],
            errors=row["errors"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the row's column values, excluding an unset ``id``."""
        data = {
            "source": self.source,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "fetched": self.fetched,
            "inserted": self.inserted,
            "updated": self.updated,
            "errors": self.errors,
        }
        if self.id is not None:
            data["id"] = self.id
        return data


@dataclass(slots=True)
class ScrapeError:
    """One row of ``scrape_errors``: detail behind a non-zero error count.

    Separate from :class:`ScrapeRun` so a single dead Workday tenant is
    identifiable rather than collapsed into a number.
    """

    run_id: int
    error: str
    occurred_at: str
    tenant: str | None = None
    id: int | None = None

    @classmethod
    def from_row(cls, row: Row | Mapping[str, Any]) -> Self:
        """Build a :class:`ScrapeError` from a database row."""
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            tenant=row["tenant"],
            error=row["error"],
            occurred_at=row["occurred_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the row's column values, excluding an unset ``id``."""
        data = {
            "run_id": self.run_id,
            "tenant": self.tenant,
            "error": self.error,
            "occurred_at": self.occurred_at,
        }
        if self.id is not None:
            data["id"] = self.id
        return data


@dataclass(slots=True)
class UpsertResult:
    """Outcome of upserting one job: what changed about that row.

    Not a table. Returned by ``ingest.upsert`` so the runner can tally
    ``inserted``/``updated`` for its :class:`ScrapeRun` without re-reading.
    """

    job_id: int
    inserted: bool
    reposted: bool = False
    fields_changed: tuple[str, ...] = field(default_factory=tuple)
