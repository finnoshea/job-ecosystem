"""Running every scraper and recording what happened.

The runner is the only place that touches a connection and a scraper together.
It owns the transaction discipline the rest of ``ingest`` assumes:

* job rows are committed before embeddings are written, because
  :func:`~jobecosystem.ingest.upsert.upsert_embeddings` commits per row and a
  failed row's rollback would otherwise discard uncommitted jobs;
* a failed scraper still produces a row in SQL table ``scrape_runs``, so a
  silently dead source is visible rather than indistinguishable from a board
  with no openings;
* one scraper's failure never prevents the others from running.

Embedding is optional and lazy. The runner takes an embedder callable rather
than importing one, so the daily scrape does not require the embedding client
or its API calls unless embedding is actually requested.
"""

from __future__ import annotations

import sqlite3
import traceback
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ..core import db as core_db
from . import upsert
from .base import Scraper, ScraperError

#: Takes job descriptions and returns one vector BLOB per description, in order.
EmbedFn = Callable[[Sequence[str]], list[bytes]]


@dataclass(slots=True)
class SourceOutcome:
    """What happened for one scraper in one run.

    ``run_id`` is set even when the scraper failed, because the run row is
    always written. ``error`` is a short message; ``exc_text`` keeps the
    traceback for debugging a parser that broke.
    """

    source: str
    run_id: int | None = None
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    reposted: int = 0
    embeddings_written: int = 0
    embedding_failures: int = 0
    errors: list[tuple[str | None, str]] = field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None
    exc_text: str | None = None

    @property
    def ok(self) -> bool:
        """True when the scraper ran to completion without a fatal error."""
        return self.error is None

    def __len__(self) -> int:
        return self.fetched


@dataclass(slots=True)
class RunSummary:
    """Outcomes for every scraper in one invocation."""

    outcomes: list[SourceOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every source completed without a fatal error."""
        return all(outcome.ok for outcome in self.outcomes)

    @property
    def failed_sources(self) -> list[str]:
        """Names of sources that raised."""
        return [o.source for o in self.outcomes if not o.ok]

    def __len__(self) -> int:
        return len(self.outcomes)

    def __iter__(self):
        return iter(self.outcomes)


def run_scrapers(
    conn: sqlite3.Connection,
    scrapers: Iterable[Scraper],
    *,
    embed: EmbedFn | None = None,
    on_outcome: Callable[[SourceOutcome], None] | None = None,
) -> RunSummary:
    """Run each scraper, upserting its jobs and recording a run row.

    Sources run in the order given. Exceptions from one scraper are caught,
    recorded, and do not stop the others -- the whole point of running a
    stable of fragile per-vendor parsers on a timer. KeyboardInterrupt and
    similar ``BaseException``\\ s are not caught.

    ``on_outcome`` is called with each :class:`SourceOutcome` as soon as that
    source finishes and before the next one starts, so a caller can report
    progress while the run is still going rather than only at the end. It is
    not called for a source that has not run yet, and an exception it raises is
    not caught here -- reporting is the caller's job.

    Returns a :class:`RunSummary`; the caller inspects it to decide whether to
    alert. Nothing is raised for a scraper failure.
    """
    summary = RunSummary()

    for scraper in scrapers:
        outcome = _run_one(conn, scraper, embed=embed)
        summary.outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(outcome)

    return summary


def _run_one(
    conn: sqlite3.Connection,
    scraper: Scraper,
    *,
    embed: EmbedFn | None,
) -> SourceOutcome:
    """Run a single scraper and record everything about the attempt."""
    source = _source_of(scraper)
    started_at = core_db.utcnow()
    outcome = SourceOutcome(source=source)

    try:
        result = scraper.scrape()
    except ScraperError as error:
        # Expected vendor failure: worth a run row, not a traceback dump.
        outcome.error = f"{type(error).__name__}: {error}"
        outcome.run_id = upsert.record_run(
            conn, source, upsert.UpsertSummary(),
            started_at=started_at, errors=[(None, outcome.error)],
        )
        return outcome
    except Exception as error:  # noqa: BLE001 - one bad parser must not kill the run
        outcome.error = f"{type(error).__name__}: {error}"
        outcome.exc_text = traceback.format_exc()
        outcome.run_id = upsert.record_run(
            conn, source, upsert.UpsertSummary(),
            started_at=started_at, errors=[(None, outcome.error)],
        )
        return outcome

    outcome.fetched = len(result.jobs)
    outcome.errors = list(result.errors)
    outcome.duration_ms = result.duration_ms

    batch = upsert.upsert_jobs(conn, result.jobs)
    # Commit before embedding: upsert_embeddings commits per row, and a failed
    # row would otherwise roll back these still-uncommitted jobs.
    conn.commit()

    outcome.inserted = batch.inserted
    outcome.updated = batch.updated
    outcome.reposted = batch.reposted

    if embed is not None and batch.results:
        _embed_jobs(conn, batch.results, embed, outcome)

    outcome.run_id = upsert.record_run(
        conn, source, batch,
        started_at=started_at,
        errors=result.errors,
    )
    return outcome


def _embed_jobs(
    conn: sqlite3.Connection,
    results: Sequence[upsert.UpsertResult],
    embed: EmbedFn,
    outcome: SourceOutcome,
) -> None:
    """Embed the freshly written jobs and store the vectors.

    Descriptions come from the database rather than from the scraper's objects,
    so what is embedded is exactly what the next run will read. Jobs with no
    description are skipped: an empty document would otherwise all hash to the
    same empty document and match each other.
    """
    job_ids = [result.job_id for result in results]
    placeholders = ", ".join("?" * len(job_ids))
    rows = conn.execute(
        f"SELECT id, description FROM jobs WHERE id IN ({placeholders})"
        " ORDER BY id",
        job_ids,
    ).fetchall()

    embeddable = [(row["id"], row["description"]) for row in rows if row["description"]]
    if not embeddable:
        return

    try:
        vectors = embed([description for _, description in embeddable])
    except Exception as error:  # noqa: BLE001 - embedding failure is not fatal
        outcome.errors.append((None, f"embedding failed: {type(error).__name__}: {error}"))
        outcome.embedding_failures = len(embeddable)
        return

    if len(vectors) != len(embeddable):
        outcome.errors.append((
            None,
            f"embedder returned {len(vectors)} vectors for {len(embeddable)} jobs",
        ))
        outcome.embedding_failures += len(embeddable)
        return

    written = upsert.upsert_embeddings(conn, [
        (job_id, vector, len(vector) // 4, _model_name(embed))
        for (job_id, _description), vector in zip(embeddable, vectors)
    ])
    outcome.embeddings_written = written.written
    outcome.embedding_failures += len(written.failed)
    outcome.errors.extend((None, f"embedding for job {job_id}: {message}")
                          for job_id, message in written.failed)


def _model_name(embed: EmbedFn) -> str:
    """Best-effort label for the ``model`` column of SQL table ``job_embeddings``.

    Reads ``DEFAULT_MODEL`` off the embedder when it exposes one, so the column
    records the real model rather than a guess. Falls back to ``"unknown"``
    because an unlabeled vector is still more useful than a lost one.
    """
    return getattr(embed, "model_name", None) or "unknown"


def _source_of(scraper: Scraper) -> str:
    """Label a scraper for bookkeeping, surviving a broken ``source_name``."""
    try:
        return scraper.source_name
    except Exception:  # noqa: BLE001 - must still write a run row
        return type(scraper).__name__
