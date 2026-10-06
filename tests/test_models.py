"""Tests for jobecosystem.core.models: dataclasses and content hashing."""

from __future__ import annotations

import sqlite3

import pytest

from jobecosystem.core import db as core_db
from jobecosystem.core import models


# ---------------------------------------------------------------------------
# content_hash
# ---------------------------------------------------------------------------

def test_content_hash_is_deterministic():
    args = ("Senior Engineer", "Acme", "Python and SQLite")
    assert models.content_hash(*args) == models.content_hash(*args)


def test_content_hash_is_hex_sha256():
    digest = models.content_hash("t", "c", "d")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


@pytest.mark.parametrize(
    "title, company, description",
    [
        ("Senior  Engineer", "Acme", "Python and SQLite"),      # doubled space
        ("Senior\tEngineer", "Acme", "Python and SQLite"),      # tab
        ("Senior\nEngineer", "Acme", "Python and SQLite"),      # newline
        ("Senior\u00a0Engineer", "Acme", "Python and SQLite"),  # non-breaking space
        ("  Senior Engineer  ", "Acme", "Python and SQLite"),   # surrounding space
        ("SENIOR ENGINEER", "ACME", "PYTHON AND SQLITE"),       # case
        ("Senior Engineer", "  Acme ", " Python and SQLite "),  # company/desc padding
        ("Senior Engineer", "Acme", "Python\r\nand SQLite"),    # CRLF
        ("Senior Engineer", "Acme", "Python   and   SQLite"),   # multi-space desc
        ("Senior\t Engineer\n", "  ACME\u00a0", "Python \r\n  and  SQLite\t"),
    ],
)
def test_content_hash_normalizes_whitespace_and_case(title, company, description):
    # The whole point of the hash: the same posting re-rendered differently, or
    # re-listed on another board, must land on the same value or repost
    # detection misses it.
    canonical = models.content_hash("Senior Engineer", "Acme", "Python and SQLite")
    assert models.content_hash(title, company, description) == canonical


def test_content_hash_distinguishes_different_titles():
    a = models.content_hash("Senior Engineer", "Acme", "Python")
    b = models.content_hash("Junior Engineer", "Acme", "Python")
    assert a != b


def test_content_hash_distinguishes_different_companies():
    a = models.content_hash("Engineer", "Acme", "Python")
    b = models.content_hash("Engineer", "Beta", "Python")
    assert a != b


def test_content_hash_distinguishes_different_descriptions():
    # Two openings at one company often share a title; the description separates.
    a = models.content_hash("Engineer", "Acme", "Python")
    b = models.content_hash("Engineer", "Acme", "Go")
    assert a != b


def test_content_hash_preserves_field_boundaries():
    assert models.content_hash("AB", "C", "D") != models.content_hash("A", "BC", "D")


def test_content_hash_treats_none_and_empty_description_alike():
    assert models.content_hash("t", "c", None) == models.content_hash("t", "c", "")


def test_content_hash_handles_empty_title_and_company():
    assert models.content_hash("", "", "") == models.content_hash("", "", None)


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def test_job_fills_content_hash_when_a_description_is_present():
    job = models.Job(source="ashby", external_id="1", company="Acme",
                     title="Eng", description="Body text")
    assert job.content_hash == models.content_hash("Eng", "Acme", "Body text")


def test_job_has_no_content_hash_without_a_description():
    # NULL is deliberate: hashing title and company alone would collide across
    # distinct openings that share a title, creating false reposts.
    job = models.Job(source="ashby", external_id="1", company="Acme", title="Eng")
    assert job.content_hash is None


def test_has_description_reflects_the_pairing():
    with_description = models.Job(
        source="a", external_id="1", company="c", title="t", description="body"
    )
    without = models.Job(source="a", external_id="2", company="c", title="t")
    assert with_description.has_description is True
    assert without.has_description is False


def test_job_keeps_supplied_content_hash():
    job = models.Job(
        source="ashby", external_id="1", company="Acme", title="Eng",
        content_hash="deadbeef",
    )
    assert job.content_hash == "deadbeef"


def test_supplied_content_hash_survives_without_a_description():
    # The ``content_hash`` column is the completeness marker, but an explicit
    # value is still honoured rather than overwritten.
    job = models.Job(
        source="ashby", external_id="1", company="Acme", title="Eng",
        content_hash="deadbeef",
    )
    assert job.content_hash == "deadbeef"


def test_job_defaults_match_schema_defaults():
    job = models.Job(source="a", external_id="1", company="c", title="t")
    assert job.repost_count == 0
    assert job.status == "new"
    assert job.rating is None
    assert job.id is None
    assert job.first_seen_at is None
    assert job.posted_at is None


def test_jobs_with_same_content_but_different_source_hash_alike():
    # The cross-source repost net: same posting, different board, different id.
    a = models.Job(source="workday", external_id="99", company="  ACME ",
                   title="Senior\tEngineer", description="Python and SQLite")
    b = models.Job(source="ashby", external_id="1", company="Acme",
                   title="Senior Engineer", description="Python and SQLite")
    assert a.content_hash == b.content_hash


def test_job_to_dict_omits_unset_id():
    job = models.Job(source="a", external_id="1", company="c", title="t")
    assert "id" not in job.to_dict()


def test_job_to_dict_includes_set_id():
    job = models.Job(source="a", external_id="1", company="c", title="t", id=7)
    assert job.to_dict()["id"] == 7


def test_job_to_dict_covers_every_schema_column():
    job = models.Job(source="a", external_id="1", company="c", title="t")
    data = job.to_dict()
    # id is omitted only because it is unset; every other column is present.
    assert set(data) | {"id"} == set(models.JOB_COLUMNS)
    assert "description_fetched_at" in data


def test_job_from_row_round_trips_through_sqlite(conn):
    now = core_db.utcnow()
    job = models.Job(
        source="ashby", external_id="42", company="Acme", title="Senior Engineer",
        description="Python and SQLite", url="https://example.test/j",
        salary_min=150000, salary_max=190000, posted_at=now,
        first_seen_at=now, last_seen_at=now, raw_json="{}",
    )
    data = job.to_dict()
    conn.execute(
        f"INSERT INTO jobs ({', '.join(data)}) VALUES ({', '.join('?' * len(data))})",
        list(data.values()),
    )
    restored = models.Job.from_row(conn.execute("SELECT * FROM jobs").fetchone())

    assert restored.id == 1
    assert restored.title == "Senior Engineer"
    assert restored.salary_min == 150000
    assert restored.salary_max == 190000
    assert restored.status == "new"
    assert restored.repost_count == 0
    assert restored.content_hash == job.content_hash


def test_job_from_row_maps_a_view_row(conn):
    now = core_db.utcnow()
    conn.execute(
        "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
        " last_seen_at, content_hash) VALUES ('ashby','1','Acme','Eng',?,?,'h')",
        (now, now),
    )
    row = conn.execute("SELECT * FROM jobs_unseen").fetchone()
    assert models.Job.from_row(row).title == "Eng"


def test_job_from_row_ignores_unknown_columns():
    row = {"id": 7, "source": "a", "external_id": "b", "company": "c",
           "title": "t", "unexpected": "ignored"}
    assert models.Job.from_row(row).id == 7


def test_job_from_row_accepts_plain_mapping():
    row = dict.fromkeys(models.JOB_COLUMNS)
    row.update({"source": "a", "external_id": "1", "company": "c", "title": "t",
                "content_hash": "h"})
    assert models.Job.from_row(row).title == "t"


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, ""),
        ({"salary_min": 150000, "salary_max": 190000}, "$150k-$190k"),
        ({"salary_min": 100000}, "$100k+"),
        ({"salary_max": 90000}, "$90k+"),
        ({"salary_min": 100000, "salary_max": 100000}, "$100k-$100k"),
        ({"salary_min": 99500, "salary_max": 100499}, "$99k-$100k"),  # truncates
    ],
)
def test_job_salary_range_formatting(kwargs, expected):
    job = models.Job(source="a", external_id="1", company="c", title="t", **kwargs)
    assert job.salary_range == expected


def test_job_slots_prevent_unknown_attributes():
    job = models.Job(source="a", external_id="1", company="c", title="t")
    with pytest.raises(AttributeError):
        job.salary_currency = "USD"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def test_embedding_round_trips_through_sqlite(conn):
    now = core_db.utcnow()
    conn.execute(
        "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
        " last_seen_at, content_hash) VALUES ('ashby','1','Acme','Eng',?,?,'h')",
        (now, now),
    )
    conn.execute(
        "INSERT INTO job_embeddings (job_id, vector, dim, model, updated_at)"
        " VALUES (1, ?, 768, 'nomic-ai/nomic-embed-text-v1.5', ?)",
        (b"\x00" * 3072, now),
    )
    embedding = models.Embedding.from_row(
        conn.execute("SELECT * FROM job_embeddings").fetchone()
    )
    assert embedding.job_id == 1
    assert embedding.dim == 768
    assert len(embedding.vector) == 3072
    assert embedding.to_dict() == {
        "job_id": 1, "vector": b"\x00" * 3072, "dim": 768,
        "model": "nomic-ai/nomic-embed-text-v1.5", "updated_at": now,
    }


def test_deleting_a_job_cascades_to_its_embedding(conn):
    # Depends on PRAGMA foreign_keys=ON, which db.connect sets per connection.
    now = core_db.utcnow()
    conn.execute(
        "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
        " last_seen_at, content_hash) VALUES ('ashby','1','Acme','Eng',?,?,'h')",
        (now, now),
    )
    conn.execute(
        "INSERT INTO job_embeddings (job_id, vector, dim, model, updated_at)"
        " VALUES (1, ?, 3, 'm', ?)", (b"\x00" * 12, now),
    )
    conn.execute("DELETE FROM jobs WHERE id = 1")
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# ScrapeRun / ScrapeError
# ---------------------------------------------------------------------------

def test_scrape_run_round_trips(conn):
    conn.execute(
        "INSERT INTO scrape_runs (source, started_at) VALUES ('ashby', ?)",
        (core_db.utcnow(),),
    )
    run = models.ScrapeRun.from_row(conn.execute("SELECT * FROM scrape_runs").fetchone())
    assert run.source == "ashby"
    assert run.finished_at is None      # still in flight
    assert (run.fetched, run.inserted, run.updated, run.errors) == (0, 0, 0, 0)
    assert run.id == 1
    assert run.to_dict()["id"] == 1


def test_scrape_run_to_dict_omits_unset_id():
    run = models.ScrapeRun(source="ashby", started_at=core_db.utcnow())
    assert "id" not in run.to_dict()
    assert run.to_dict()["fetched"] == 0


def test_scrape_error_round_trips(conn):
    conn.execute("INSERT INTO scrape_runs (source, started_at) VALUES ('workday', ?)",
                 (core_db.utcnow(),))
    conn.execute(
        "INSERT INTO scrape_errors (run_id, tenant, error, occurred_at)"
        " VALUES (1, 'acme', 'timeout after 30s', ?)", (core_db.utcnow(),),
    )
    error = models.ScrapeError.from_row(
        conn.execute("SELECT * FROM scrape_errors").fetchone()
    )
    assert error.run_id == 1
    assert error.tenant == "acme"
    assert error.error == "timeout after 30s"
    assert error.to_dict()["tenant"] == "acme"


def test_scrape_error_allows_null_tenant(conn):
    conn.execute("INSERT INTO scrape_runs (source, started_at) VALUES ('ashby', ?)",
                 (core_db.utcnow(),))
    conn.execute(
        "INSERT INTO scrape_errors (run_id, error, occurred_at) VALUES (1, 'boom', ?)",
        (core_db.utcnow(),),
    )
    error = models.ScrapeError.from_row(
        conn.execute("SELECT * FROM scrape_errors").fetchone()
    )
    assert error.tenant is None


def test_deleting_a_run_cascades_to_its_errors(conn):
    conn.execute("INSERT INTO scrape_runs (source, started_at) VALUES ('workday', ?)",
                 (core_db.utcnow(),))
    conn.execute(
        "INSERT INTO scrape_errors (run_id, error, occurred_at) VALUES (1, 'boom', ?)",
        (core_db.utcnow(),),
    )
    conn.execute("DELETE FROM scrape_runs WHERE id = 1")
    assert conn.execute("SELECT COUNT(*) FROM scrape_errors").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# UpsertResult
# ---------------------------------------------------------------------------

def test_upsert_result_defaults():
    result = models.UpsertResult(job_id=1, inserted=True)
    assert result.reposted is False
    assert result.fields_changed == ()


def test_upsert_result_carries_field_changes():
    result = models.UpsertResult(
        job_id=1, inserted=False, reposted=True, fields_changed=("title", "url")
    )
    assert result.reposted is True
    assert result.fields_changed == ("title", "url")


# ---------------------------------------------------------------------------
# schema drift
# ---------------------------------------------------------------------------

def test_job_columns_match_the_schema(conn):
    # Catches the dataclass and schema.sql drifting apart.
    actual = tuple(r[1] for r in conn.execute("PRAGMA table_info(jobs)"))
    assert set(actual) == set(models.JOB_COLUMNS)


def test_status_check_constraint_matches_job_status_alias(conn):
    now = core_db.utcnow()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
            " last_seen_at, content_hash, status) VALUES ('a','1','c','t',?,?,'h','bogus')",
            (now, now),
        )


@pytest.mark.parametrize("rating", [0, 1, 2, 3, 4, 5])
def test_rating_accepts_the_inclusive_range(conn, rating):
    now = core_db.utcnow()
    conn.execute(
        "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
        " last_seen_at, content_hash, rating) VALUES ('a','1','c','t',?,?,'h',?)",
        (now, now, rating),
    )
    assert conn.execute("SELECT rating FROM jobs").fetchone()[0] == rating


@pytest.mark.parametrize("rating", [-1, 6, 99])
def test_rating_rejects_out_of_range_values(conn, rating):
    now = core_db.utcnow()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
            " last_seen_at, content_hash, rating) VALUES ('a','1','c','t',?,?,'h',?)",
            (now, now, rating),
        )


def test_rating_may_be_null(conn):
    # NULL means unrated, which is the state of every freshly scraped job.
    now = core_db.utcnow()
    conn.execute(
        "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
        " last_seen_at, content_hash) VALUES ('a','1','c','t',?,?,'h')",
        (now, now),
    )
    assert conn.execute("SELECT rating FROM jobs").fetchone()[0] is None
