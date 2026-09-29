"""Prefix-aware wrapper around the nomic embedding model.

nomic-embed-text-v1.5 is asymmetric: documents and queries are embedded with
different task prefixes, and mixing them up silently degrades search quality
without erroring. This module makes the distinction explicit in the API rather
than leaving it to callers to remember.

Loading happens lazily and is cached per process. SentenceTransformer is
imported inside the loader so the text-only parts of the package (schema,
queries, TUI) do not pay for torch at import time.

The model cache lives in the repo's ``embedders/`` directory via ``HF_HOME``;
see ``config.py``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from . import similarity

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# nomic's documented task prefixes. Anything not in this map is a caller error.
PREFIX_DOCUMENT = "search_document: "
PREFIX_QUERY = "search_query: "
PREFIX_CLUSTERING = "clustering: "
PREFIX_CLASSIFICATION = "classification: "

TASK_PREFIXES = {
    "document": PREFIX_DOCUMENT,
    "query": PREFIX_QUERY,
    "clustering": PREFIX_CLUSTERING,
    "classification": PREFIX_CLASSIFICATION,
}


class EmbedderError(RuntimeError):
    """Raised when the model is missing or a vector cannot be produced."""


def _repo_root() -> Path:
    # src/jobecosystem/core/embedder.py -> core -> jobecosystem -> src -> root
    return Path(__file__).resolve().parents[3]


def configure_hf_home() -> Path:
    """Point Hugging Face's cache at the repo's ``embedders/`` directory.

    Set before the model loads; respects an existing ``HF_HOME`` so tests or an
    unusual install can redirect the cache.
    """
    existing = os.environ.get("HF_HOME")
    if existing:
        return Path(existing).expanduser()
    hf_home = _repo_root() / "embedders"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    return hf_home


@lru_cache(maxsize=1)
def get_model(model_id: str = DEFAULT_MODEL):
    """Load and cache the SentenceTransformer for this process.

    Raises :class:`EmbedderError` if the dependency is missing or the model
    cannot be loaded, so callers get one exception type to handle.
    """
    configure_hf_home()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - environment problem
        raise EmbedderError(
            "sentence-transformers is not installed; run `pip install "
            "sentence-transformers` in the project virtualenv"
        ) from error

    try:
        return SentenceTransformer(model_id)
    except Exception as error:  # noqa: BLE001 - surfaced as one type on purpose
        raise EmbedderError(f"could not load embedding model {model_id!r}: {error}") from error


def embedding_model_name(model_id: str = DEFAULT_MODEL) -> str:
    """Value stored in ``job_embeddings.model``, used to detect model changes."""
    return model_id


def dimension(model_id: str = DEFAULT_MODEL) -> int:
    """Vector width of ``model_id``, or :data:`similarity.NOMIC_DIM` as fallback."""
    try:
        model = get_model(model_id)
        # Renamed in sentence-transformers 6; support both spellings.
        getter = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        return int(getter())
    except EmbedderError:
        return similarity.NOMIC_DIM


def embed_documents(
    texts: Sequence[str],
    *,
    model_id: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed job descriptions for storage, with the document prefix applied.

    Empty or whitespace-only strings are still embedded (as the bare prefix)
    rather than dropped, so result indices line up with the input.
    """
    return _embed(texts, task="document", model_id=model_id, batch_size=batch_size)


def embed_query(
    text: str,
    *,
    model_id: str = DEFAULT_MODEL,
) -> list[float]:
    """Embed a single search query, with the query prefix applied."""
    return _embed([text], task="query", model_id=model_id, batch_size=1)[0]


def embed_document(
    text: str,
    *,
    model_id: str = DEFAULT_MODEL,
) -> list[float]:
    """Embed a single job description, with the document prefix applied."""
    return _embed([text], task="document", model_id=model_id, batch_size=1)[0]


def _embed(
    texts: Sequence[str],
    *,
    task: str,
    model_id: str,
    batch_size: int,
) -> list[list[float]]:
    if task not in TASK_PREFIXES:
        raise ValueError(
            f"unknown task {task!r}; expected one of {sorted(TASK_PREFIXES)}"
        )
    if not texts:
        return []

    prefix = TASK_PREFIXES[task]
    prepared = [f"{prefix}{(text or '').strip()}" for text in texts]

    model = get_model(model_id)
    vectors = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,   # so cosine reduces to a dot product
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_document_blob(text: str, *, model_id: str = DEFAULT_MODEL) -> bytes:
    """Embed a job description straight to the BLOB stored in SQLite."""
    return similarity.encode(embed_document(text, model_id=model_id))


def embed_query_blob(text: str, *, model_id: str = DEFAULT_MODEL) -> bytes:
    """Embed a query straight to the BLOB form used for comparisons."""
    return similarity.encode(embed_query(text, model_id=model_id))
