"""Embedding the resume and ranking jobs against it.

The resume is embedded as a **query** (``input_type="query"``), matching the
``document`` vectors stored for jobs, so the TUI's cosine ranking applies
unchanged. Nothing new is stored: ranking reuses ``triage.queries.similar_jobs``
over the existing ``job_embeddings`` table.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

from ..core import embedder
from .models import Resume
from .resume import query_text

EmbedFn = Callable[[str], Sequence[float]]


def embed_resume(resume: Resume, *, embed: EmbedFn | None = None,
                 model_id: str | None = None) -> list[float]:
    """Embed the resume's query text. ``embed`` is injected for tests."""
    function = embed or embedder.embed_query
    text = query_text(resume)
    if model_id is None:
        return list(function(text))
    return list(function(text, model_id=model_id))  # type: ignore[call-arg]


def similar_jobs(conn: sqlite3.Connection, resume: Resume, *,
                 embed: EmbedFn | None = None, limit: int = 50,
                 min_score: float = 0.0):
    """Rank stored jobs by similarity to the resume.

    Imports the query layer lazily so the resume/overlay code can be used
    without pulling in the triage package.
    """
    from ..triage import queries

    vector = embed_resume(resume, embed=embed)
    return queries.similar_jobs(
        conn, vector, scope="all", limit=limit, min_score=min_score
    )
