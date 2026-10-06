"""Tests for the reporting views: run_stats, run_errors_by_source,
description_progress.

These views exist to be queried from outside the codebase, so the tests assert
the shape an external tool would rely on: column names, and numbers that add up.
"""

from __future__ import annotations

from jobecosystem.core import db as core_db
from jobecosystem.core.models import Job
from jobecosystem.ingest import upsert


def add_run(conn, source, *, started, finished=None, fetched=0, inserted=0,
            updated=0, errors=0, run_errors=()):
    cursor = conn.execute(
        "INSERT INTO scrape_runs (source, started_at, finished_at, fetched,"
        " inserted, updated, errors) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source, started, finished, fetched, inserted, updated, errors),
    )
    for tenant, message in run_errors:
        conn.execute(
            "INSERT INTO scrape_errors (run_id, tenant, error, occurred_at)"
            " VALUES (?, ?, ?, ?)",
            (cursor.lastrowid, tenant, message, finished or started),
        )
    conn.commit()
    return cursor.lastrowid


def add_job(conn, external_id, *, description=None):
    job = Job(
        source="ashby:ramp",
        external_id=external_id,
        company="Acme",
        title="Engineer",
        description=description,
    )
    result = upsert.upsert_job(conn, job)
    conn.commit()
    return result.job_id


# ---------------------------------------------------------------------------
# run_stats
# ---------------------------------------------------------------------------

def test_run_stats_shape(conn):
    add_run(conn, "ashby:ramp", started="2026-09-30T10:00:00Z",
            finished="2026-09-30T10:02:00Z", fetched=100, inserted=90)
    row = conn.execute("SELECT * FROM run_stats").fetchone()
    assert set(row.keys()) == {
        "id", "source", "started_at", "finished_at", "status",
        "duration_seconds", "minutes_since_start", "fetched", "inserted",
        "updated", "errors", "inserted_percent",
    }


def test_finished_run_reports_duration_in_seconds(conn):
    add_run(conn, "a", started="2026-09-30T10:00:00Z", finished="2026-09-30T10:02:00Z")
    row = conn.execute("SELECT * FROM run_stats").fetchone()
    assert row["status"] == "finished"
    assert row["duration_seconds"] == 120.0


def test_sub_minute_duration_is_fractional(conn):
    add_run(conn, "a", started="2026-09-30T10:00:00Z", finished="2026-09-30T10:00:45Z")
    assert conn.execute("SELECT duration_seconds FROM run_stats").fetchone()[0] == 45.0


def test_in_flight_run_is_marked_running(conn):
    add_run(conn, "a", started="2026-09-30T10:00:00Z")
    row = conn.execute("SELECT * FROM run_stats").fetchone()
    assert row["status"] == "running"
    assert row["finished_at"] is None


def test_running_row_has_no_duration_but_does_report_elapsed(conn):
    # The elapsed figure is the whole point of the row for an in-flight run.
    add_run(conn, "a", started=core_db.utcnow())
    row = conn.execute("SELECT * FROM run_stats").fetchone()
    assert row["duration_seconds"] is None
    assert row["minutes_since_start"] is not None


def test_minutes_since_start_is_populated_for_finished_runs_too(conn):
    add_run(conn, "a", started="2026-09-01T10:00:00Z", finished="2026-09-01T10:01:00Z")
    assert conn.execute("SELECT minutes_since_start FROM run_stats").fetchone()[0] > 0


def test_inserted_percent_is_computed(conn):
    add_run(conn, "a", started="x", finished="y", fetched=200, inserted=50)
    assert conn.execute("SELECT inserted_percent FROM run_stats").fetchone()[0] == 25.0


def test_inserted_percent_is_null_when_nothing_was_fetched(conn):
    # Avoids a division by zero on a failed run.
    add_run(conn, "a", started="x", finished="y", fetched=0, inserted=0)
    assert conn.execute("SELECT inserted_percent FROM run_stats").fetchone()[0] is None


def test_inserted_percent_can_exceed_zero_for_a_rerun(conn):
    add_run(conn, "a", started="x", finished="y", fetched=100, inserted=0, updated=40)
    assert conn.execute("SELECT inserted_percent FROM run_stats").fetchone()[0] == 0.0


def test_run_stats_covers_every_run(conn):
    add_run(conn, "a", started="x", finished="y")
    add_run(conn, "b", started="x", finished="y")
    add_run(conn, "c", started="x")
    assert conn.execute("SELECT COUNT(*) FROM run_stats").fetchone()[0] == 3


def test_run_stats_matches_scrape_runs(conn):
    add_run(conn, "a", started="x", finished="y", fetched=7, inserted=3, errors=2)
    row = conn.execute("SELECT * FROM run_stats").fetchone()
    assert (row["fetched"], row["inserted"], row["errors"]) == (7, 3, 2)


def test_run_stats_orders_by_nothing_in_particular(conn):
    # A view, not a report: callers apply their own ORDER BY. Asserting that
    # keeps the view free of an implied sort.
    add_run(conn, "b", started="x", finished="y")
    add_run(conn, "a", started="x", finished="y")
    sources = [r[0] for r in conn.execute("SELECT source FROM run_stats ORDER BY source")]
    assert sources == ["a", "b"]


# ---------------------------------------------------------------------------
# run_errors_by_source
# ---------------------------------------------------------------------------

def test_errors_roll_up_per_source(conn):
    add_run(conn, "nvidia", started="x", finished="y", errors=3)
    add_run(conn, "nvidia", started="x", finished="y", errors=2)
    add_run(conn, "disney", started="x", finished="y")
    rows = {r["source"]: r for r in conn.execute("SELECT * FROM run_errors_by_source")}

    assert rows["nvidia"]["runs"] == 2
    assert rows["nvidia"]["error_count"] == 5
    assert rows["nvidia"]["runs_with_errors"] == 2
    assert rows["disney"]["runs"] == 1
    assert rows["disney"]["error_count"] == 0
    assert rows["disney"]["runs_with_errors"] == 0


def test_errors_by_source_tracks_the_latest_run(conn):
    add_run(conn, "a", started="2026-01-01T00:00:00Z", finished="2026-01-01T00:01:00Z")
    add_run(conn, "a", started="2026-06-01T00:00:00Z", finished="2026-06-01T00:01:00Z")
    row = conn.execute("SELECT * FROM run_errors_by_source").fetchone()
    assert row["last_run_at"] == "2026-06-01T00:00:00Z"


def test_errors_by_source_is_empty_without_runs(conn):
    assert conn.execute("SELECT COUNT(*) FROM run_errors_by_source").fetchone()[0] == 0


def test_errors_by_source_shape(conn):
    add_run(conn, "a", started="x", finished="y")
    row = conn.execute("SELECT * FROM run_errors_by_source").fetchone()
    assert set(row.keys()) == {
        "source", "runs", "error_count", "runs_with_errors", "last_run_at"
    }


def test_error_detail_is_available_alongside_the_totals(conn):
    # The rollup says how many; scrape_errors says what.
    add_run(conn, "nvidia", started="x", finished="y", errors=1,
            run_errors=[("nvidia", "timeout after 30s")])
    row = conn.execute("SELECT error FROM scrape_errors").fetchone()
    assert row["error"] == "timeout after 30s"


# ---------------------------------------------------------------------------
# description_progress
# ---------------------------------------------------------------------------

def test_description_progress_shape(conn):
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    assert set(row.keys()) == {
        "total_jobs", "described", "pending", "described_percent",
        "first_fetched_at", "last_fetched_at",
    }


def test_description_progress_on_an_empty_database(conn):
    # A single row of NULLs, not zero rows: it is an aggregate, and external
    # tools should not have to handle a missing row.
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    assert row is not None
    assert row["total_jobs"] == 0
    assert row["described"] is None
    assert row["described_percent"] is None


def test_description_progress_counts_described_and_pending(conn):
    add_job(conn, "1", description="body")
    add_job(conn, "2", description="body")
    add_job(conn, "3")
    add_job(conn, "4")
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    assert (row["total_jobs"], row["described"], row["pending"]) == (4, 2, 2)
    assert row["described_percent"] == 50.0


def test_description_progress_is_full_when_everything_is_fetched(conn):
    for i in range(3):
        add_job(conn, str(i), description="body")
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    assert row["described_percent"] == 100.0
    assert row["pending"] == 0


def test_description_progress_is_zero_when_nothing_is_fetched(conn):
    for i in range(3):
        add_job(conn, str(i))
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    assert row["described_percent"] == 0.0
    assert row["pending"] == 3


def test_pending_matches_the_work_queue(conn):
    # The number an external poller sees must agree with the view the batch uses.
    for i in range(5):
        add_job(conn, str(i), description="body" if i < 2 else None)
    progress = conn.execute("SELECT pending FROM description_progress").fetchone()[0]
    queue = conn.execute(
        "SELECT COUNT(*) FROM jobs_needing_descriptions"
    ).fetchone()[0]
    assert progress == queue == 3


def test_exhausted_rows_are_not_pending(conn):
    # A row that has hit the retry cap leaves the queue and stops counting as
    # pending work, but nothing marks it -- it is simply never described.
    add_job(conn, "1")
    exhausted = add_job(conn, "2")
    conn.execute("UPDATE jobs SET description_attempts = 5 WHERE id = ?", (exhausted,))
    conn.commit()
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    assert row["pending"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM jobs_needing_descriptions"
    ).fetchone()[0] == 1


def test_description_progress_records_the_fetch_window(conn):
    add_job(conn, "1", description="body")
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    assert row["first_fetched_at"] == row["last_fetched_at"]
    assert row["first_fetched_at"] is not None


def test_description_progress_ignores_jobs_without_fetch_times(conn):
    add_job(conn, "1")
    row = conn.execute("SELECT * FROM description_progress").fetchone()
    assert row["first_fetched_at"] is None


def test_description_progress_advances_as_fetches_land(conn):
    ids = [add_job(conn, str(i)) for i in range(4)]
    assert conn.execute("SELECT described FROM description_progress").fetchone()[0] == 0

    conn.execute("UPDATE jobs SET description='x', content_hash='h' WHERE id=?", (ids[0],))
    conn.commit()
    assert conn.execute("SELECT described FROM description_progress").fetchone()[0] == 1

    conn.execute("UPDATE jobs SET description='x', content_hash='h' WHERE id=?", (ids[1],))
    conn.commit()
    assert conn.execute("SELECT described FROM description_progress").fetchone()[0] == 2
