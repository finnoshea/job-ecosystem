"""Tests for jobecosystem.ingest.embed: the standalone embedding batch.

No model is loaded: the embedder is injected, so these exercise the queue, the
chunking, and the write without torch.
"""

from __future__ import annotations

from jobecosystem.core import similarity
from jobecosystem.core.models import Job
from jobecosystem.ingest import embed as batch
from jobecosystem.ingest import upsert


def add_job(conn, external_id, description="body text", **overrides):
    fields = {
        "source": "ashby:ramp",
        "external_id": external_id,
        "company": "Acme",
        "title": "Engineer",
        "description": description,
    }
    fields.update(overrides)
    result = upsert.upsert_job(conn, Job(**fields))
    conn.commit()
    return result.job_id


def constant_embed(dim=3):
    calls: list[list[str]] = []

    def embed(texts):
        calls.append(list(texts))
        return [[1.0] + [0.0] * (dim - 1) for _ in texts]

    embed.calls = calls
    return embed


# ---------------------------------------------------------------------------
# pending_jobs
# ---------------------------------------------------------------------------

def test_pending_includes_described_unembedded_jobs(conn):
    job_id = add_job(conn, "1")
    assert batch.pending_jobs(conn, limit=10, model="m") == [(job_id, "body text")]


def test_pending_skips_jobs_without_a_description(conn):
    add_job(conn, "1", description=None)
    assert batch.pending_jobs(conn, limit=10, model="m") == []


def test_pending_skips_jobs_already_embedded_with_the_same_model(conn):
    job_id = add_job(conn, "1")
    upsert.upsert_embeddings(conn, [(job_id, similarity.encode([1.0, 0.0]), 2, "m")])
    assert batch.pending_jobs(conn, limit=10, model="m") == []


def test_pending_re_embeds_jobs_stored_with_another_model(conn):
    job_id = add_job(conn, "1")
    upsert.upsert_embeddings(conn, [(job_id, similarity.encode([1.0, 0.0]), 2, "old")])
    assert batch.pending_jobs(conn, limit=10, model="new") == [(job_id, "body text")]


def test_pending_force_includes_already_embedded_jobs(conn):
    job_id = add_job(conn, "1")
    upsert.upsert_embeddings(conn, [(job_id, similarity.encode([1.0, 0.0]), 2, "m")])
    assert batch.pending_jobs(conn, limit=10, model="m", force=True) == [
        (job_id, "body text")
    ]


def test_pending_respects_the_limit_in_id_order(conn):
    ids = [add_job(conn, str(i)) for i in range(5)]
    assert [job_id for job_id, _ in batch.pending_jobs(
        conn, limit=2, model="m"
    )] == ids[:2]


# ---------------------------------------------------------------------------
# embed_pending
# ---------------------------------------------------------------------------

def test_embed_pending_writes_vectors_with_dim_and_model(conn):
    job_id = add_job(conn, "1")
    summary = batch.embed_pending(conn, constant_embed(3), model="m")

    assert summary.attempted == 1
    assert summary.written == 1
    assert summary.ok
    row = conn.execute(
        "SELECT vector, dim, model FROM job_embeddings WHERE job_id = ?", (job_id,)
    ).fetchone()
    assert row["dim"] == 3
    assert row["model"] == "m"
    assert len(similarity.decode(row["vector"])) == 3


def test_embed_pending_embeds_in_chunks(conn):
    for i in range(5):
        add_job(conn, str(i))
    embed = constant_embed()
    summary = batch.embed_pending(conn, embed, batch_size=2, model="m")

    assert summary.attempted == 5
    assert summary.written == 5
    # ceil(5 / 2) = 3 model calls
    assert [len(call) for call in embed.calls] == [2, 2, 1]


def test_embed_pending_records_a_model_failure_and_stops(conn):
    for i in range(3):
        add_job(conn, str(i))

    def boom(texts):
        raise RuntimeError("no model")

    summary = batch.embed_pending(conn, boom, batch_size=2, model="m")
    assert summary.written == 0
    assert summary.attempted == 2                 # only the first chunk was tried
    assert all("no model" in message for _, message in summary.failed)
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 0


def test_embed_pending_records_a_vector_count_mismatch(conn):
    add_job(conn, "1")
    summary = batch.embed_pending(conn, lambda texts: [], model="m")
    assert summary.written == 0
    assert "0 vectors" in summary.failed[0][1]


def test_embed_pending_is_a_no_op_when_nothing_is_pending(conn):
    add_job(conn, "1", description=None)
    summary = batch.embed_pending(conn, constant_embed(), model="m")
    assert summary.attempted == 0
    assert summary.ok


def test_embed_pending_uses_the_limit(conn):
    for i in range(5):
        add_job(conn, str(i))
    summary = batch.embed_pending(conn, constant_embed(), limit=2, model="m")
    assert summary.written == 2
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 2


def test_model_name_for_reads_the_embedder_attribute():
    def embed(texts):
        return []

    assert batch.model_name_for(embed)  # falls back to the default model id

    embed.model_name = "custom"
    assert batch.model_name_for(embed) == "custom"
