"""Wrapper around Voyage AI's embedding API.

Voyage is asymmetric: a stored document and a search query are embedded with
different ``input_type`` values (``"document"`` vs ``"query"``), and mixing them
up silently degrades search quality without erroring. This module makes the
distinction explicit in the API rather than leaving it to callers to remember.

Unlike the previous local model, embeddings are a hosted call: the client reads
``VOYAGE_API_KEY`` from the environment, is imported and constructed lazily so
the text-only parts of the package do not pay for it, and a missing key surfaces
as :class:`EmbedderError` rather than an import failure. The client is
injectable so tests never touch the network.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from . import similarity

DEFAULT_MODEL = "voyage-4-lite"

#: Texts per API request. Voyage accepts batches; 128 matches its documented
#: safe maximum and keeps a single request payload small.
DEFAULT_BATCH_SIZE = 128

#: Voyage's ``input_type`` values. Stored documents and search queries must use
#: the matching one.
INPUT_DOCUMENT = "document"
INPUT_QUERY = "query"

#: Known per-model vector widths, so :func:`dimension` needs no API call.
#: Anything not listed falls back to the default width.
MODEL_DIMENSIONS = {
    "voyage-3-lite": 512,
    "voyage-large-2": 1536,
}


class EmbedderError(RuntimeError):
    """Raised when the API key is missing or a vector cannot be produced."""


#: The Voyage client, cached after the first successful construction. Injected
#: by tests via :func:`set_client`.
_client = None

#: Tokens billed across this process, for the CLI to report. Reset per run.
_usage_tokens = 0


def set_client(client) -> None:
    """Install a client (tests) or clear the cache with ``None``."""
    global _client
    _client = client


def get_client():
    """Return the cached Voyage client, constructing it on first use.

    Raises :class:`EmbedderError` when the key is absent or the package is not
    installed, so callers get one exception type to handle.
    """
    global _client
    if _client is not None:
        return _client

    if not os.environ.get("VOYAGE_API_KEY"):
        raise EmbedderError(
            "VOYAGE_API_KEY is not set; export it or pass a client via set_client()"
        )
    try:
        import voyageai
    except ImportError as error:  # pragma: no cover - environment problem
        raise EmbedderError(
            "the voyageai package is not installed; run `pip install voyageai`"
        ) from error

    try:
        _client = voyageai.Client()
    except Exception as error:  # noqa: BLE001 - surfaced as one type on purpose
        raise EmbedderError(f"could not create the Voyage client: {error}") from error
    return _client


def reset_usage() -> None:
    """Zero the token counter, before a batch run."""
    global _usage_tokens
    _usage_tokens = 0


def total_tokens_used() -> int:
    """Tokens billed since the last :func:`reset_usage`."""
    return _usage_tokens


def _add_usage(tokens) -> None:
    global _usage_tokens
    try:
        _usage_tokens += int(tokens or 0)
    except (TypeError, ValueError):
        pass


def embedding_model_name(model_id: str = DEFAULT_MODEL) -> str:
    """Value stored in ``job_embeddings.model``, used to detect model changes."""
    return model_id


def dimension(model_id: str = DEFAULT_MODEL) -> int:
    """Vector width of ``model_id``, defaulting to :data:`similarity.VOYAGE_DIM`."""
    return MODEL_DIMENSIONS.get(model_id, similarity.VOYAGE_DIM)


def embed_documents(
    texts: Sequence[str],
    *,
    model_id: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Embed job descriptions for storage, with ``input_type="document"``.

    Batched into requests of at most ``batch_size`` texts; the returned vectors
    line up one-for-one with the input.
    """
    return _embed(
        texts, input_type=INPUT_DOCUMENT, model_id=model_id, batch_size=batch_size
    )


def embed_query(
    text: str,
    *,
    model_id: str = DEFAULT_MODEL,
) -> list[float]:
    """Embed a single search query, with ``input_type="query"``."""
    return _embed(
        [text], input_type=INPUT_QUERY, model_id=model_id, batch_size=1
    )[0]


def embed_document(
    text: str,
    *,
    model_id: str = DEFAULT_MODEL,
) -> list[float]:
    """Embed a single job description, with ``input_type="document"``."""
    return _embed(
        [text], input_type=INPUT_DOCUMENT, model_id=model_id, batch_size=1
    )[0]


def _embed(
    texts: Sequence[str],
    *,
    input_type: str,
    model_id: str,
    batch_size: int,
) -> list[list[float]]:
    if input_type not in (INPUT_DOCUMENT, INPUT_QUERY):
        raise ValueError(
            f"unknown input_type {input_type!r}; expected {INPUT_DOCUMENT!r}"
            f" or {INPUT_QUERY!r}"
        )
    if not texts:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    client = get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [str(text) for text in texts[start : start + batch_size]]
        try:
            response = client.embed(
                texts=batch, model=model_id, input_type=input_type
            )
        except Exception as error:  # noqa: BLE001 - normalized for callers
            raise EmbedderError(
                f"Voyage embed failed for {model_id!r}: {error}"
            ) from error
        vectors.extend(response.embeddings)
        _add_usage(getattr(response, "total_tokens", 0))
    return vectors


def embed_document_blob(text: str, *, model_id: str = DEFAULT_MODEL) -> bytes:
    """Embed a job description straight to the BLOB stored in SQLite."""
    return similarity.encode(embed_document(text, model_id=model_id))


def embed_query_blob(text: str, *, model_id: str = DEFAULT_MODEL) -> bytes:
    """Embed a query straight to the BLOB form used for comparisons."""
    return similarity.encode(embed_query(text, model_id=model_id))
