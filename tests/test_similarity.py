"""Tests for jobecosystem.core.similarity: vector encoding and cosine ranking."""

from __future__ import annotations

import math

import pytest

from jobecosystem.core import similarity as sim


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------

def test_encode_decode_round_trip_exactly():
    # Values chosen to be exactly representable in float32.
    values = [1.0, 0.0, -0.5, 2.25]
    assert sim.decode(sim.encode(values)) == values


def test_encode_decode_round_trip_within_float32_precision():
    values = [0.1, 0.2, -0.3]
    got = sim.decode(sim.encode(values))
    assert all(abs(a - b) < 1e-6 for a, b in zip(got, values))


def test_encode_produces_four_bytes_per_dimension():
    assert len(sim.encode([0.0] * 768)) == 768 * 4


def test_encode_of_empty_vector_is_empty_bytes():
    assert sim.encode([]) == b""


def test_decode_accepts_matching_dim():
    assert len(sim.decode(sim.encode([1.0] * 768), 768)) == 768


def test_decode_rejects_mismatched_dim():
    with pytest.raises(ValueError, match="expected 768 dimensions"):
        sim.decode(sim.encode([1.0, 2.0, 3.0]), 768)


def test_decode_rejects_truncated_blob():
    with pytest.raises(ValueError, match="not a multiple"):
        sim.decode(b"\x00\x00\x00")


def test_voyage_dim_constant_matches_model():
    assert sim.VOYAGE_DIM == 1024


# ---------------------------------------------------------------------------
# dot / norm / cosine
# ---------------------------------------------------------------------------

def test_dot_basic():
    assert sim.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0


def test_dot_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        sim.dot([1.0, 2.0], [1.0])


def test_norm_of_unit_vector():
    assert math.isclose(sim.norm([1.0, 0.0, 0.0]), 1.0)


def test_norm_of_three_four_five_triangle():
    assert math.isclose(sim.norm([3.0, 4.0]), 5.0)


def test_cosine_of_identical_vectors_is_one():
    assert math.isclose(sim.cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, abs_tol=1e-9)


def test_cosine_is_scale_invariant():
    assert math.isclose(sim.cosine([1.0, 2.0], [10.0, 20.0]), 1.0, abs_tol=1e-9)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert math.isclose(sim.cosine([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_of_opposed_vectors_is_minus_one():
    assert math.isclose(sim.cosine([1.0, 0.0], [-1.0, 0.0]), -1.0, abs_tol=1e-9)


def test_cosine_of_zero_vector_is_zero_not_an_error():
    # Ranking a long list must not abort because one description was empty.
    assert sim.cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert sim.cosine([1.0, 1.0], [0.0, 0.0]) == 0.0
    assert sim.cosine([0.0], [0.0]) == 0.0


def test_cosine_blob_matches_cosine_of_decoded():
    a, b = [0.1, 0.2, 0.3], [0.4, -0.5, 0.6]
    assert math.isclose(
        sim.cosine_blob(sim.encode(a), sim.encode(b)),
        sim.cosine(a, b),
        rel_tol=1e-6,
    )


def test_cosine_blob_validates_dim():
    with pytest.raises(ValueError):
        sim.cosine_blob(sim.encode([1.0, 2.0]), sim.encode([1.0, 2.0]), dim=768)


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------

@pytest.fixture
def candidates():
    return [
        (1, sim.encode([1.0, 0.0, 0.0])),    # identical direction
        (2, sim.encode([0.9, 0.1, 0.0])),    # close
        (3, sim.encode([0.0, 1.0, 0.0])),    # orthogonal
        (4, sim.encode([-1.0, 0.0, 0.0])),   # opposed
    ]


def test_top_k_orders_by_descending_score(candidates):
    ranked = sim.top_k([1.0, 0.0, 0.0], candidates, k=4, min_score=-1.0)
    assert [job_id for job_id, _ in ranked] == [1, 2, 3, 4]


def test_top_k_limits_to_k(candidates):
    assert len(sim.top_k([1.0, 0.0, 0.0], candidates, k=2)) == 2


def test_top_k_filters_by_min_score(candidates):
    ranked = sim.top_k([1.0, 0.0, 0.0], candidates, k=10, min_score=0.5)
    assert {job_id for job_id, _ in ranked} == {1, 2}


def test_top_k_with_k_zero_returns_nothing(candidates):
    assert sim.top_k([1.0, 0.0, 0.0], candidates, k=0) == []


def test_top_k_with_negative_k_returns_nothing(candidates):
    assert sim.top_k([1.0, 0.0, 0.0], candidates, k=-1) == []


def test_top_k_with_no_candidates_returns_nothing():
    assert sim.top_k([1.0, 0.0], [], k=5) == []


def test_top_k_default_min_score_drops_negatively_correlated(candidates):
    # The default min_score=0.0 is what makes "jobs like this one" meaningful:
    # an opposed vector is not a match.
    assert {job_id for job_id, _ in sim.top_k([1.0, 0.0, 0.0], candidates, k=99)} == {1, 2, 3}


def test_top_k_k_larger_than_candidate_count(candidates):
    assert len(sim.top_k([1.0, 0.0, 0.0], candidates, k=99, min_score=-1.0)) == 4


def test_top_k_scores_are_floats(candidates):
    for _, score in sim.top_k([1.0, 0.0, 0.0], candidates, k=4, min_score=-1.0):
        assert isinstance(score, float)


def test_top_k_breakdown_of_scores(candidates):
    ranked = dict(sim.top_k([1.0, 0.0, 0.0], candidates, k=4, min_score=-1.0))
    assert math.isclose(ranked[1], 1.0, abs_tol=1e-6)
    assert math.isclose(ranked[3], 0.0, abs_tol=1e-6)
    assert math.isclose(ranked[4], -1.0, abs_tol=1e-6)


def test_top_k_validates_dimension_when_dim_given(candidates):
    with pytest.raises(ValueError):
        sim.top_k([1.0, 0.0, 0.0], candidates, k=1, dim=768)


# ---------------------------------------------------------------------------
# top_k_neighbors
# ---------------------------------------------------------------------------

def test_top_k_neighbors_returns_named_records(candidates):
    neighbors = sim.top_k_neighbors([1.0, 0.0, 0.0], candidates, k=2, min_score=-1.0)
    assert [n.job_id for n in neighbors] == [1, 2]
    assert all(isinstance(n.score, float) for n in neighbors)


def test_top_k_neighbors_is_frozen(candidates):
    neighbor = sim.top_k_neighbors([1.0, 0.0, 0.0], candidates, k=1, min_score=-1.0)[0]
    with pytest.raises(Exception):
        neighbor.job_id = 99  # type: ignore[misc]


def test_top_k_neighbors_agrees_with_top_k(candidates):
    query = [0.5, 0.5, 0.0]
    assert [(n.job_id, n.score) for n in sim.top_k_neighbors(query, candidates, k=3)] == \
           sim.top_k(query, candidates, k=3)


# ---------------------------------------------------------------------------
# integration: BLOB round-trip through SQLite
# ---------------------------------------------------------------------------

def test_vector_survives_sqlite_blob_round_trip(conn):
    from jobecosystem.core import db as core_db

    now = core_db.utcnow()
    conn.execute(
        "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
        " last_seen_at, content_hash) VALUES ('ashby', '1', 'Acme', 'Eng', ?, ?, 'h')",
        (now, now),
    )
    original = [0.1, 0.2, 0.3]
    conn.execute(
        "INSERT INTO job_embeddings (job_id, vector, dim, model, updated_at)"
        " VALUES (1, ?, 3, 'test-model', ?)",
        (sim.encode(original), now),
    )
    row = conn.execute("SELECT vector, dim FROM job_embeddings WHERE job_id = 1").fetchone()
    decoded = sim.decode(row["vector"], row["dim"])
    assert all(abs(a - b) < 1e-6 for a, b in zip(decoded, original))
