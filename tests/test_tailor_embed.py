"""Tests for jobecosystem.tailor.embed: resume query text and ranking."""

from __future__ import annotations

from jobecosystem.core import similarity
from jobecosystem.core.models import Job
from jobecosystem.ingest import upsert
from jobecosystem.tailor import embed as embed_mod


def add_job(conn, external_id, **overrides):
    fields = {"source": "ashby:ramp", "external_id": external_id,
              "company": "Acme", "title": "Engineer", "description": "body"}
    fields.update(overrides)
    result = upsert.upsert_job(conn, Job(**fields))
    conn.commit()
    return result.job_id


def test_embed_resume_uses_the_query_text(sample_resume):
    seen = {}

    def embed(text):
        seen["text"] = text
        return [1.0, 0.0, 0.0]

    vector = embed_mod.embed_resume(sample_resume, embed=embed)
    assert vector == [1.0, 0.0, 0.0]
    assert "Cut latency 40% using caches." in seen["text"]


def test_similar_jobs_ranks_by_the_resume_vector(conn, sample_resume):
    matching = add_job(conn, "1", title="Python Platform Engineer")
    other = add_job(conn, "2", title="Nurse")
    upsert.upsert_embeddings(conn, [
        (matching, similarity.encode([1.0, 0.0, 0.0]), 3, "test"),
        (other, similarity.encode([0.0, 1.0, 0.0]), 3, "test"),
    ])

    hits = embed_mod.similar_jobs(
        conn, sample_resume, embed=lambda text: [1.0, 0.0, 0.0], limit=5
    )
    assert hits[0].job.id == matching
    assert hits[0].score > hits[1].score
