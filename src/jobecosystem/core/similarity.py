"""Vector encoding and cosine similarity for job embeddings.

Vectors are stored in SQLite as little-endian float32 BLOBs (see
``job_embeddings.vector``). Similarity is computed in Python: SQLite has no
vector type, and this is fine for tens of thousands of jobs. If it ever gets
slow, revisit with the ``sqlite-vec`` extension -- do not add a separate
vector database.

nomic-embed-text-v1.5 vectors are already unit length after normalization, but
nothing here assumes that; every comparison normalizes.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

# Matches the dim column used by core.embedder for nomic-embed-text-v1.5.
NOMIC_DIM = 768

_FLOAT32 = array  # alias for readability below


def encode(vector: Sequence[float]) -> bytes:
    """Pack a vector into the BLOB form stored in ``job_embeddings.vector``."""
    floats = _FLOAT32("f", vector)
    if _sys_byteorder_is_big():
        floats.byteswap()
    return floats.tobytes()


def decode(blob: bytes, dim: int | None = None) -> list[float]:
    """Unpack a BLOB from the database back into a list of floats.

    ``dim`` is checked when supplied, which turns a truncated or wrong-model
    BLOB into an immediate error instead of a quietly wrong score.
    """
    floats = _FLOAT32("f")
    if len(blob) % floats.itemsize:
        raise ValueError(
            f"vector BLOB length {len(blob)} is not a multiple of "
            f"{floats.itemsize} bytes"
        )
    floats.frombytes(blob)
    if _sys_byteorder_is_big():
        floats.byteswap()

    if dim is not None and len(floats) != dim:
        raise ValueError(f"expected {dim} dimensions, got {len(floats)}")
    return list(floats)


def _sys_byteorder_is_big() -> bool:
    import sys

    return sys.byteorder == "big"


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(vector: Sequence[float]) -> float:
    """Euclidean length of a vector."""
    return math.sqrt(sum(x * x for x in vector))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector is all zeros.

    Returns a plain float rather than raising on a zero vector because callers
    rank long lists and one empty description should not abort the sort.
    """
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def cosine_blob(a: bytes, b: bytes, dim: int | None = None) -> float:
    """Cosine similarity between two stored vectors."""
    return cosine(decode(a, dim), decode(b, dim))


def top_k(
    query: Sequence[float],
    candidates: Iterable[tuple[int, bytes]],
    *,
    k: int = 10,
    min_score: float = 0.0,
    dim: int | None = None,
) -> list[tuple[int, float]]:
    """Rank ``(job_id, vector_blob)`` pairs against ``query``.

    Returns at most ``k`` ``(job_id, score)`` pairs, highest score first,
    dropping anything below ``min_score``. This is the brute-force path behind
    "jobs like this one"; it assumes the caller already limited ``candidates``.
    """
    if k <= 0:
        return []
    scored = [
        (job_id, cosine_blob(encode(query), blob, dim))
        for job_id, blob in candidates
    ]
    scored = [(job_id, score) for job_id, score in scored if score >= min_score]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


@dataclass(frozen=True)
class Neighbor:
    """A ranked match, for callers that prefer a named shape over a tuple."""

    job_id: int
    score: float


def top_k_neighbors(
    query: Sequence[float],
    candidates: Iterable[tuple[int, bytes]],
    *,
    k: int = 10,
    min_score: float = 0.0,
    dim: int | None = None,
) -> list[Neighbor]:
    """Same as :func:`top_k`, returning :class:`Neighbor` records."""
    return [
        Neighbor(job_id=job_id, score=score)
        for job_id, score in top_k(
            query, candidates, k=k, min_score=min_score, dim=dim
        )
    ]
