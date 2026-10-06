"""Tests for jobecosystem.ingest.upsert: dedup, repost, and run bookkeeping.

The repost tests drive the gap against an explicit anchor rather than wall
clock, because ``_is_repost`` measures scrape activity, not real time.
"""

from __future__ import annotations

import json

import pytest

from jobecosystem.core import db as core_db
from jobecosystem.core.models import Job
from jobecosystem.ingest import upsert

# The repost tests drive gaps against a reference point derived from the current
# time, because ``upsert`` compares against ``utcnow()``. A hardcoded constant
# here would drift and silently change every relative offset as real time passed.
def anchor_time() -> str:
    """A stable timestamp for this test, near but not after now."""
    return core_db.utcnow()


def make_job(external_id="1", description="A job description", **overrides):
    """A Job with a description by default, so ``content_hash`` is populated.

    Rows without a description are a deliberate case (see the
    no-description tests) rather than the default, because a description is what
    makes the hash -- and therefore repost detection -- meaningful.
    """
    fields = {
        "source": "ashby",
        "external_id": external_id,
        "company": "Acme",
        "title": "Engineer",
        "description": description,
    }
    fields.update(overrides)
    return Job(**fields)


def commit_jobs(conn, *jobs):
    """Write jobs and commit them.

    ``upsert_job``/``upsert_jobs`` leave the transaction open for the caller.
    Any test that then writes embeddings must commit first, because
    ``upsert_embeddings`` commits per row and a rollback would otherwise take
    the uncommitted jobs with it.
    """
    upsert.upsert_jobs(conn, list(jobs))
    conn.commit()


def rewind(conn, external_id, days):
    """Set last_seen_at exactly ``days`` before the run's reference point."""
    conn.execute(
        "UPDATE jobs SET last_seen_at = datetime(?, ?) WHERE external_id = ?",
        (anchor_time(), f"-{days} days", external_id),
    )
    conn.commit()


def pin_anchor(conn, days_before=1):
    """Create/refresh a throwaway row so the table anchor sits near the run point.

    Without this the anchor is whatever the freshest row happens to be, and
    every rewind would look like a same-run sighting.
    """
    reference = anchor_time()
    conn.execute(
        "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
        " last_seen_at, content_hash) VALUES ('anchor','anchor','A','T', ?, ?, 'h')"
        " ON CONFLICT (source, external_id) DO UPDATE SET last_seen_at = excluded.last_seen_at",
        (reference, reference),
    )
    conn.execute(
        "UPDATE jobs SET last_seen_at = datetime(?, ?) WHERE source = 'anchor'",
        (reference, f"-{days_before} days"),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------

def test_insert_stamps_bookkeeping_fields(conn):
    result = upsert.upsert_job(conn, make_job())
    assert result.inserted is True
    assert result.reposted is False

    row = conn.execute("SELECT * FROM jobs").fetchone()
    assert row["first_seen_at"] is not None
    assert row["last_seen_at"] == row["first_seen_at"]
    assert row["repost_count"] == 0
    assert row["status"] == "new"


def test_insert_fills_content_hash(conn):
    job = make_job()
    upsert.upsert_job(conn, job)
    row = conn.execute("SELECT content_hash FROM jobs").fetchone()
    assert row["content_hash"] == job.content_hash
    assert len(row["content_hash"]) == 64


def test_insert_without_a_description_leaves_content_hash_null(conn):
    # NULL is the marker for "description not fetched yet", which
    # jobs_needing_descriptions selects on.
    upsert.upsert_job(conn, make_job(description=None))
    row = conn.execute(
        "SELECT content_hash, description_fetched_at FROM jobs"
    ).fetchone()
    assert row["content_hash"] is None
    assert row["description_fetched_at"] is None


def test_insert_with_a_description_stamps_description_fetched_at(conn):
    upsert.upsert_job(conn, make_job())
    row = conn.execute("SELECT description_fetched_at FROM jobs").fetchone()
    assert row["description_fetched_at"] is not None


def test_no_description_row_appears_in_the_work_queue(conn):
    upsert.upsert_jobs(conn, [make_job("1", description=None), make_job("2")])
    ids = [r[0] for r in conn.execute("SELECT external_id FROM jobs_needing_descriptions")]
    assert ids == ["1"]


def test_a_later_description_is_written(conn):
    # The scrape stores the listing; a paced fetch fills the text in later.
    upsert.upsert_job(conn, make_job("1", description=None, url="https://x.test/j"))
    assert conn.execute("SELECT content_hash FROM jobs").fetchone()[0] is None

    result = upsert.upsert_job(conn, make_job("1", description="Fetched later"))
    row = conn.execute(
        "SELECT description, content_hash, description_fetched_at, url FROM jobs"
    ).fetchone()
    assert row["description"] == "Fetched later"
    assert row["content_hash"] is not None
    assert row["description_fetched_at"] is not None
    assert row["url"] == "https://x.test/j"      # untouched by the update
    assert "description_fetched_at" in result.fields_changed


def test_a_second_description_update_does_not_restamp(conn):
    upsert.upsert_job(conn, make_job("1"))
    first = conn.execute("SELECT description_fetched_at FROM jobs").fetchone()[0]
    upsert.upsert_job(conn, make_job("1", title="Renamed"))
    assert conn.execute("SELECT description_fetched_at FROM jobs").fetchone()[0] == first


def test_insert_overwrites_scraper_supplied_bookkeeping(conn):
    # upsert owns these fields; a scraper setting them must not win.
    job = make_job(
        first_seen_at="1999-01-01T00:00:00Z",
        last_seen_at="1999-01-01T00:00:00Z",
        repost_count=42,
        status="applied",
        rating=5,
    )
    upsert.upsert_job(conn, job)
    row = conn.execute("SELECT * FROM jobs").fetchone()
    assert row["first_seen_at"] != "1999-01-01T00:00:00Z"
    assert row["repost_count"] == 0
    assert row["status"] == "new"
    assert row["rating"] is None       # triage judgment, never a scraper's


def test_update_preserves_a_user_rating(conn):
    upsert.upsert_job(conn, make_job())
    conn.execute("UPDATE jobs SET rating = 4, status = 'seen'")
    conn.commit()

    upsert.upsert_job(conn, make_job(title="Renamed"))
    row = conn.execute("SELECT rating, status, title FROM jobs").fetchone()
    assert row["rating"] == 4
    assert row["status"] == "seen"
    assert row["title"] == "Renamed"


def test_rating_survives_the_description_fetch_update(conn):
    upsert.upsert_job(conn, make_job("1", description=None))
    conn.execute("UPDATE jobs SET rating = 3")
    conn.commit()

    upsert.upsert_job(conn, make_job("1", description="Fetched later"))
    row = conn.execute("SELECT rating, content_hash FROM jobs").fetchone()
    assert row["rating"] == 3
    assert row["content_hash"] is not None


def test_insert_returns_the_new_job_id(conn):
    result = upsert.upsert_job(conn, make_job())
    assert result.job_id == conn.execute("SELECT id FROM jobs").fetchone()[0]


def test_insert_preserves_optional_fields(conn):
    upsert.upsert_job(
        conn,
        make_job(location="Remote", description="Python", url="https://x.test/j",
                 salary_min=150000, salary_max=190000),
    )
    row = conn.execute("SELECT * FROM jobs").fetchone()
    assert (row["location"], row["description"], row["url"]) == (
        "Remote", "Python", "https://x.test/j")
    assert (row["salary_min"], row["salary_max"]) == (150000, 190000)


def test_second_identical_write_is_unchanged_not_inserted(conn):
    upsert.upsert_job(conn, make_job())
    result = upsert.upsert_job(conn, make_job())
    assert result.inserted is False
    assert result.fields_changed == ()
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_update_advances_last_seen_but_not_first_seen(conn):
    upsert.upsert_job(conn, make_job())
    first = conn.execute("SELECT first_seen_at FROM jobs").fetchone()[0]

    # Rewind so the refresh has something to advance past. utcnow() has
    # one-second resolution, so re-inserting immediately might not move it.
    rewind(conn, "1", 1)
    rewound = conn.execute("SELECT last_seen_at FROM jobs").fetchone()[0]
    assert rewound < first

    upsert.upsert_job(conn, make_job())

    row = conn.execute("SELECT first_seen_at, last_seen_at FROM jobs").fetchone()
    assert row["first_seen_at"] == first
    assert row["last_seen_at"] > rewound


def test_update_reports_changed_fields(conn):
    upsert.upsert_job(conn, make_job())
    result = upsert.upsert_job(conn, make_job(title="Senior Engineer"))
    assert result.inserted is False
    assert "title" in result.fields_changed
    assert conn.execute("SELECT title FROM jobs").fetchone()[0] == "Senior Engineer"


def test_update_does_not_clear_fields_with_none(conn):
    # A sparse listing must not erase data a fuller listing already provided.
    upsert.upsert_job(conn, make_job(location="Remote", description="Python"))
    upsert.upsert_job(conn, make_job(location=None, description=None))
    row = conn.execute("SELECT location, description FROM jobs").fetchone()
    assert row["location"] == "Remote"
    assert row["description"] == "Python"


def test_update_ignores_bookkeeping_columns_in_fields_changed(conn):
    upsert.upsert_job(conn, make_job())
    rewind(conn, "1", 1)
    result = upsert.upsert_job(conn, make_job())
    assert "last_seen_at" not in result.fields_changed
    assert "repost_count" not in result.fields_changed
    assert "status" not in result.fields_changed


def test_update_does_not_reset_triage_status(conn):
    upsert.upsert_job(conn, make_job())
    conn.execute("UPDATE jobs SET status = 'applied' WHERE external_id = '1'")
    conn.commit()
    upsert.upsert_job(conn, make_job(title="Renamed"))
    assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "applied"


def test_different_sources_with_same_external_id_are_separate_rows(conn):
    # The unique constraint is (source, external_id), not external_id alone.
    upsert.upsert_job(conn, make_job())
    upsert.upsert_job(conn, make_job(source="workday"))
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# repost detection
# ---------------------------------------------------------------------------

def test_first_insert_is_never_a_repost(conn):
    assert upsert.upsert_job(conn, make_job()).reposted is False


def test_short_absence_is_not_a_repost(conn):
    upsert.upsert_job(conn, make_job())
    pin_anchor(conn, days_before=1)
    rewind(conn, "1", 7)
    assert upsert.upsert_job(conn, make_job()).reposted is False
    assert conn.execute("SELECT repost_count FROM jobs WHERE external_id='1'").fetchone()[0] == 0


@pytest.mark.parametrize("days", [14, 15, 30, 90, 365])
def test_long_absence_is_a_repost(conn, days):
    upsert.upsert_job(conn, make_job())
    pin_anchor(conn, days_before=1)
    rewind(conn, "1", days)
    assert upsert.upsert_job(conn, make_job()).reposted is True


def test_just_under_the_gap_is_not_a_repost(conn):
    upsert.upsert_job(conn, make_job())
    pin_anchor(conn, days_before=1)
    rewind(conn, "1", upsert.REPOST_GAP_DAYS - 1)
    assert upsert.upsert_job(conn, make_job()).reposted is False


def test_repost_increments_the_counter(conn):
    upsert.upsert_job(conn, make_job())
    pin_anchor(conn, days_before=1)
    rewind(conn, "1", 30)
    upsert.upsert_job(conn, make_job())
    assert conn.execute("SELECT repost_count FROM jobs WHERE external_id='1'").fetchone()[0] == 1


def test_repeated_reposts_accumulate(conn):
    upsert.upsert_job(conn, make_job())
    pin_anchor(conn, days_before=1)
    for _ in range(3):
        rewind(conn, "1", 30)
        upsert.upsert_job(conn, make_job())
    assert conn.execute("SELECT repost_count FROM jobs WHERE external_id='1'").fetchone()[0] == 3


def test_a_job_seen_in_the_same_run_is_not_a_repost(conn):
    # The same-run guard: a row whose last_seen_at equals the table anchor is
    # simply still posted, no matter how old it is.
    upsert.upsert_job(conn, make_job())
    conn.execute("UPDATE jobs SET last_seen_at = ? WHERE external_id = '1'", (anchor_time(),))
    conn.commit()
    assert upsert.upsert_job(conn, make_job()).reposted is False


def test_repost_count_shows_up_in_the_jobs_reposted_view(conn):
    # End-to-end: the bookkeeping upsert writes is what the view selects on.
    upsert.upsert_job(conn, make_job())
    assert conn.execute("SELECT COUNT(*) FROM jobs_reposted").fetchone()[0] == 0
    pin_anchor(conn, days_before=1)
    rewind(conn, "1", 30)
    upsert.upsert_job(conn, make_job())
    assert conn.execute("SELECT COUNT(*) FROM jobs_reposted").fetchone()[0] == 1


def test_repost_detection_does_not_depend_on_batch_order(conn):
    # Regression: the anchor was read per row, so the first row written moved it
    # to "now" and every later row looked like a sighting of itself. A repost
    # was only ever detected if it happened to be written first.
    for order in (("1", "2"), ("2", "1")):
        conn.execute("DELETE FROM jobs")
        conn.commit()

        upsert.upsert_jobs(conn, [make_job(e) for e in order])
        conn.execute(
            "UPDATE jobs SET last_seen_at = datetime('now','-1 day')"
            " WHERE external_id <> '2'"
        )
        conn.execute(
            "UPDATE jobs SET last_seen_at = datetime('now','-30 days')"
            " WHERE external_id = '2'"
        )
        conn.commit()

        summary = upsert.upsert_jobs(conn, [make_job(e) for e in order])
        assert summary.reposted == 1, f"order {order} lost the repost"
        counts = dict(conn.execute(
            "SELECT external_id, repost_count FROM jobs"
        ).fetchall())
        assert counts["2"] == 1
        assert counts["1"] == 0


def test_repost_anchor_is_the_newest_last_seen(conn):
    upsert.upsert_jobs(conn, [make_job("1")])
    conn.execute("UPDATE jobs SET last_seen_at = '2026-01-05T00:00:00Z'")
    conn.commit()
    assert upsert.repost_anchor(conn) == "2026-01-05T00:00:00Z"


def test_repost_anchor_is_none_on_an_empty_table(conn):
    assert upsert.repost_anchor(conn) is None


def test_repost_gap_default_is_fourteen_days():
    assert upsert.REPOST_GAP_DAYS == 14


# ---------------------------------------------------------------------------
# upsert_jobs (batch)
# ---------------------------------------------------------------------------

def test_batch_insert_tallies(conn):
    summary = upsert.upsert_jobs(conn, [make_job("1"), make_job("2")])
    assert (summary.fetched, summary.inserted, summary.updated, summary.unchanged) == (2, 2, 0, 0)
    assert len(summary) == 2


def test_batch_of_identical_jobs_is_all_unchanged(conn):
    upsert.upsert_jobs(conn, [make_job("1"), make_job("2")])
    summary = upsert.upsert_jobs(conn, [make_job("1"), make_job("2")])
    assert (summary.inserted, summary.updated, summary.unchanged) == (0, 0, 2)


def test_batch_tallies_changed_rows(conn):
    upsert.upsert_jobs(conn, [make_job("1"), make_job("2")])
    summary = upsert.upsert_jobs(conn, [make_job("1", title="New"), make_job("2")])
    assert (summary.inserted, summary.updated, summary.unchanged) == (0, 1, 1)


def test_batch_counts_reposts(conn):
    upsert.upsert_jobs(conn, [make_job("1")])
    pin_anchor(conn, days_before=1)
    rewind(conn, "1", 30)
    summary = upsert.upsert_jobs(conn, [make_job("1")])
    assert summary.reposted == 1


def test_batch_empty_is_an_empty_summary(conn):
    summary = upsert.upsert_jobs(conn, [])
    assert (summary.fetched, len(summary)) == (0, 0)


def test_batch_accumulates_results(conn):
    summary = upsert.upsert_jobs(conn, [make_job("1"), make_job("2"), make_job("3")])
    assert [r.job_id for r in summary.results] == sorted(r.job_id for r in summary.results)
    assert all(isinstance(r.inserted, bool) for r in summary.results)


def test_batch_accepts_a_generator(conn):
    summary = upsert.upsert_jobs(conn, (make_job(str(i)) for i in range(3)))
    assert summary.inserted == 3


def test_batch_is_atomic_on_failure(conn):
    # A bad row must not leave earlier rows half-written.
    class Exploding:
        source = "ashby"

        def __getattr__(self, name):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        upsert.upsert_jobs(conn, [make_job("1"), Exploding()])
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# upsert_embeddings
# ---------------------------------------------------------------------------

def test_embedding_insert_and_replace(conn):
    commit_jobs(conn, make_job())
    summary = upsert.upsert_embeddings(conn, [(1, b"\x00" * 12, 3, "m")])
    assert summary.written == 1
    assert summary.ok

    upsert.upsert_embeddings(conn, [(1, b"\x01" * 12, 3, "m2")])
    rows = conn.execute("SELECT vector, model FROM job_embeddings").fetchall()
    assert len(rows) == 1
    assert rows[0]["vector"] == b"\x01" * 12
    assert rows[0]["model"] == "m2"


def test_embedding_batch_writes_all_rows(conn):
    commit_jobs(conn, make_job("1"), make_job("2"))
    summary = upsert.upsert_embeddings(conn, [(1, b"\x00" * 12, 3, "m"),
                                              (2, b"\x00" * 12, 3, "m")])
    assert summary.written == 2
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 2


def test_embedding_updated_at_is_set(conn):
    commit_jobs(conn, make_job())
    upsert.upsert_embeddings(conn, [(1, b"\x00" * 12, 3, "m")])
    assert conn.execute("SELECT updated_at FROM job_embeddings").fetchone()[0] is not None


def test_embedding_batch_keeps_successes_when_one_row_fails(conn):
    # The reason for the per-row loop: a stale job_id fails its foreign key and
    # must not discard the rest of the batch.
    commit_jobs(conn, make_job("1"), make_job("2"))
    summary = upsert.upsert_embeddings(conn, [
        (1, b"\x00" * 12, 3, "m"),
        (999, b"\x00" * 12, 3, "m"),
        (2, b"\x00" * 12, 3, "m"),
    ])
    assert summary.written == 2
    assert summary.ok is False
    assert [job_id for job_id, _ in summary.failed] == [999]
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 2


def test_embedding_failure_records_the_database_message(conn):
    commit_jobs(conn, make_job("1"))
    summary = upsert.upsert_embeddings(conn, [(404, b"\x00" * 12, 3, "m")])
    (_job_id, message), = summary.failed
    assert "FOREIGN KEY" in message.upper()


def test_embedding_failure_does_not_roll_back_committed_jobs(conn):
    # Regression: wrapping the per-row write in `with conn` around a swallowed
    # exception left the connection mid-transaction, and the next commit
    # discarded the job rows too.
    commit_jobs(conn, make_job("1"), make_job("2"))
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2

    upsert.upsert_embeddings(conn, [
        (1, b"\x00" * 12, 3, "m"),
        (999, b"\x00" * 12, 3, "m"),
    ])

    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 1


def test_embedding_uncommitted_jobs_are_lost_if_the_caller_does_not_commit(conn):
    # Documents the contract: upsert_embeddings commits, so the caller must
    # commit its own work first or a per-row rollback takes it along.
    upsert.upsert_job(conn, make_job("1"))  # deliberately not committed
    upsert.upsert_embeddings(conn, [(999, b"\x00" * 12, 3, "m")])
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_embedding_failure_leaves_no_open_transaction(conn):
    commit_jobs(conn, make_job("1"))
    upsert.upsert_embeddings(conn, [(999, b"\x00" * 12, 3, "m")])
    # A caller beginning its own transaction must not inherit an aborted one.
    with conn:
        conn.execute("UPDATE jobs SET title = 'Renamed' WHERE external_id = '1'")
    assert conn.execute("SELECT title FROM jobs").fetchone()[0] == "Renamed"


def test_embedding_connection_is_usable_after_a_failure(conn):
    commit_jobs(conn, make_job("1"))
    upsert.upsert_embeddings(conn, [(999, b"\x00" * 12, 3, "m")])
    # A failed row must not poison the connection for later writes.
    summary = upsert.upsert_embeddings(conn, [(1, b"\x00" * 12, 3, "m")])
    assert summary.ok
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 1


def test_embedding_empty_batch(conn):
    summary = upsert.upsert_embeddings(conn, [])
    assert (summary.written, summary.failed, summary.ok, len(summary)) == (0, [], True, 0)


def test_embedding_write_summary_len_counts_both_outcomes():
    summary = upsert.EmbeddingWriteSummary(written=2, failed=[(1, "boom")])
    assert len(summary) == 3


def test_embedding_write_summary_reports_not_ok_with_failures():
    assert upsert.EmbeddingWriteSummary(written=1).ok
    assert not upsert.EmbeddingWriteSummary(written=1, failed=[(2, "x")]).ok


# ---------------------------------------------------------------------------
# record_run
# ---------------------------------------------------------------------------

def test_record_run_writes_summary(conn):
    summary = upsert.upsert_jobs(conn, [make_job("1")])
    run_id = upsert.record_run(conn, "ashby", summary, started_at=core_db.utcnow())

    row = conn.execute("SELECT * FROM scrape_runs").fetchone()
    assert run_id == row["id"]
    assert (row["source"], row["fetched"], row["inserted"], row["updated"]) == (
        "ashby", 1, 1, 0)
    assert row["finished_at"] is not None
    assert row["errors"] == 0


def test_record_run_writes_error_children(conn):
    summary = upsert.UpsertSummary(fetched=5)
    upsert.record_run(
        conn, "workday", summary,
        started_at=core_db.utcnow(),
        errors=[("acme", "timeout"), ("globex", "HTTP 503")],
    )
    row = conn.execute("SELECT errors FROM scrape_runs").fetchone()
    assert row["errors"] == 2
    tenants = [r[0] for r in conn.execute("SELECT tenant FROM scrape_errors ORDER BY tenant")]
    assert tenants == ["acme", "globex"]


def test_record_run_accepts_tenantless_errors(conn):
    upsert.record_run(
        conn, "ashby", upsert.UpsertSummary(),
        started_at=core_db.utcnow(), errors=[(None, "not JSON")],
    )
    assert conn.execute("SELECT tenant FROM scrape_errors").fetchone()[0] is None


def test_record_run_is_written_even_when_nothing_was_fetched(conn):
    # A failed run must be visible, not indistinguishable from no run.
    upsert.record_run(
        conn, "ashby", upsert.UpsertSummary(),
        started_at=core_db.utcnow(), errors=[(None, "connection refused")],
    )
    row = conn.execute("SELECT * FROM scrape_runs").fetchone()
    assert row is not None
    assert (row["fetched"], row["errors"]) == (0, 1)


def test_record_run_uses_supplied_finished_at(conn):
    upsert.record_run(
        conn, "ashby", upsert.UpsertSummary(),
        started_at="2026-01-01T00:00:00Z", finished_at="2026-01-01T00:01:00Z",
    )
    assert conn.execute("SELECT finished_at FROM scrape_runs").fetchone()[0] == "2026-01-01T00:01:00Z"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_job_row_to_dict_covers_schema_columns(conn):
    upsert.upsert_job(conn, make_job())
    data = upsert.job_row_to_dict(conn.execute("SELECT * FROM jobs").fetchone())
    assert len(data) == len(Job.__dataclass_fields__)
    assert data["source"] == "ashby"


def test_dumps_raw_serializes_payloads():
    assert json.loads(upsert.dumps_raw({"a": 1})) == {"a": 1}


def test_dumps_raw_returns_none_for_unserializable_input():
    # A weird payload must not fail the scrape.
    assert upsert.dumps_raw(object()) is None


def test_raw_json_round_trips_through_a_job(conn):
    payload = {"id": 42, "title": "Engineer"}
    upsert.upsert_job(conn, make_job(raw_json=upsert.dumps_raw(payload)))
    stored = conn.execute("SELECT raw_json FROM jobs").fetchone()[0]
    assert json.loads(stored) == payload
