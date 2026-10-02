"""Embedding job descriptions, in batches, for the standalone embed step.

Scraping and description fetching produce rows; this fills ``job_embeddings``
from the descriptions so the triage UI's similarity search has something to rank
against. It is deliberately a separate operation from both:

* It needs the embedding model (torch and friends, the ``embed`` extra), which
  the scrape and describe commands do not.
* The model call is the expensive part, so it is bounded per run and resumable
  from the database.

The work queue is the set of jobs that have a description but no embedding *for
the requested model*. A row embedded with a different model is re-embedded, so
switching models repairs the table instead of mixing incomparable vectors.

Only rows with a description are eligible: an empty document would embed to the
bare ``search_document: `` prefix and match every other empty document. That is
also why this runs after the description fetch, not before it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..core import embedder, similarity
from . import upsert

#: Signature of the injectable embedder: texts -> one float vector per text, in
#: order. This is :func:`jobecosystem.core.embedder.embed_documents`, injected so
#: the batch is testable without a model.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@dataclass(slots=True)
class EmbedSummary:
    """Outcome of one embedding batch."""

    attempted: int = 0
    written: int = 0
    failed: list[tuple[int, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no embedding failed."""
        return not self.failed

    def __len__(self) -> int:
        return self.attempted


def model_name_for(embed: EmbedFn) -> str:
    """The model label to record, from the embedder when it exposes one."""
    return getattr(embed, "model_name", None) or embedder.DEFAULT_MODEL


def pending_jobs(
    conn: sqlite3.Connection,
    *,
    limit: int,
    model: str,
    force: bool = False,
) -> list[tuple[int, str]]:
    """Jobs with a description but no embedding for ``model``, oldest first.

    Ordered by ``id`` so a run is deterministic and a later run resumes exactly
    where this one stopped. ``force`` re-embeds even rows already embedded with
    ``model``, for when the prefixes change but the model id does not.
    """
    where = [
        "j.description IS NOT NULL",
        "j.description <> ''",
    ]
    params: list[object] = []
    if not force:
        where.append("(e.job_id IS NULL OR e.model IS NOT ?)")
        params.append(model)
    params.append(limit)

    sql = (
        "SELECT j.id, j.description FROM jobs j"
        " LEFT JOIN job_embeddings e ON e.job_id = j.id"
        " WHERE " + " AND ".join(where)
        + " ORDER BY j.id LIMIT ?"
    )
    return [(row["id"], row["description"]) for row in conn.execute(sql, params)]


def embed_pending(
    conn: sqlite3.Connection,
    embed: EmbedFn,
    *,
    limit: int = 200,
    batch_size: int = 32,
    model: str | None = None,
    force: bool = False,
) -> EmbedSummary:
    """Embed up to ``limit`` pending descriptions, in chunks of ``batch_size``.

    Each chunk of descriptions is embedded in one model call and then written.
    If the model call raises, or returns the wrong number of vectors, the whole
    chunk is recorded as failed and the run stops: a broken embedder (missing
    model, out of memory) would fail every remaining chunk the same way, so
    there is nothing to gain by continuing.
    """
    model = model or model_name_for(embed)
    rows = pending_jobs(conn, limit=limit, model=model, force=force)

    summary = EmbedSummary()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        summary.attempted += len(chunk)
        texts = [description for _, description in chunk]

        try:
            vectors = embed(texts)
        except Exception as error:  # noqa: BLE001 - surfaced per job, not raised
            summary.failed.extend(
                (job_id, f"{type(error).__name__}: {error}") for job_id, _ in chunk
            )
            break

        if len(vectors) != len(chunk):
            summary.failed.extend(
                (job_id, f"embedder returned {len(vectors)} vectors for {len(chunk)}")
                for job_id, _ in chunk
            )
            break

        written = upsert.upsert_embeddings(conn, [
            (job_id, similarity.encode(vector), len(vector), model)
            for (job_id, _), vector in zip(chunk, vectors)
        ])
        summary.written += written.written
        summary.failed.extend(written.failed)

    return summary
