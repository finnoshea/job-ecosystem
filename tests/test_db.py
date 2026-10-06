"""Tests for jobecosystem.core.db: connection setup, migrations, views wiring."""

from __future__ import annotations

import sqlite3

import pytest

from jobecosystem.core import db


# ---------------------------------------------------------------------------
# utcnow
# ---------------------------------------------------------------------------

def test_utcnow_format_is_iso8601_z():
    value = db.utcnow()
    assert len(value) == 20
    assert value.endswith("Z")
    assert value[10] == "T"
    # Round-trips through SQLite's own datetime parser.
    assert sqlite3.connect(":memory:").execute(
        "SELECT datetime(?)", (value,)
    ).fetchone()[0] is not None


def test_utcnow_is_lexicographically_sortable():
    # The schema relies on string comparison for recency windows.
    assert db.utcnow() <= db.utcnow()


# ---------------------------------------------------------------------------
# resolve_db_path
# ---------------------------------------------------------------------------

def test_resolve_db_path_prefers_explicit_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "from-env.db"))
    assert db.resolve_db_path(tmp_path / "explicit.db") == tmp_path / "explicit.db"


def test_resolve_db_path_uses_env_when_no_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "from-env.db"))
    assert db.resolve_db_path() == tmp_path / "from-env.db"


def test_resolve_db_path_falls_back_to_repo_default(monkeypatch):
    monkeypatch.delenv("DB_PATH", raising=False)
    assert db.resolve_db_path() == db.DEFAULT_DB_PATH
    assert db.DEFAULT_DB_PATH.name == "jobs.db"


def test_resolve_db_path_expands_user(monkeypatch):
    monkeypatch.delenv("DB_PATH", raising=False)
    assert "~" not in str(db.resolve_db_path("~/x.db"))


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------

def test_connect_creates_the_database_file(db_path):
    assert not db_path.exists()
    conn = db.connect(db_path)
    conn.close()
    # -wal/-shm appear only once something is written; the main file is enough.
    assert db_path.exists()


def test_connect_creates_missing_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "jobs.db"
    conn = db.connect(nested)
    conn.close()
    assert nested.exists()


def test_connect_enables_wal(db_path):
    conn = db.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_connect_enables_foreign_keys(db_path):
    conn = db.connect(db_path)
    try:
        # Per-connection pragma; must be on for ON DELETE CASCADE to fire.
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_connect_sets_row_factory_to_row(db_path):
    conn = db.connect(db_path)
    try:
        row = conn.execute("SELECT 1 AS one").fetchone()
        assert row["one"] == 1
    finally:
        conn.close()


def test_connect_applies_views_by_default(db_path):
    conn = db.connect(db_path)
    try:
        views = _view_names(conn)
    finally:
        conn.close()
    assert {"jobs_today", "jobs_unseen", "jobs_reposted", "jobs_stale"} <= views


def test_connect_skips_schema_and_views_when_disabled(tmp_path):
    path = tmp_path / "bare.db"
    conn = db.connect(path, run_migration_and_update_views=False)
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert "jobs" not in tables
    assert "schema_migrations" not in tables


def test_connect_is_idempotent(db_path):
    first = db.connect(db_path)
    first.close()
    second = db.connect(db_path)
    try:
        assert db.schema_version(second) == db.LATEST_VERSION
    finally:
        second.close()


def test_migration_run_parameter_rejects_old_name(db_path):
    with pytest.raises(TypeError):
        db.connect(db_path, run_migrations=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------

def test_migrate_creates_all_tables(conn):
    tables = {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"jobs", "job_embeddings", "scrape_runs", "scrape_errors"} <= tables


def test_migrate_records_version(conn):
    assert db.schema_version(conn) == db.LATEST_VERSION


def test_migrate_is_idempotent(conn):
    # Second call in the same process must be a no-op.
    assert db.migrate(conn) == []
    assert db.schema_version(conn) == db.LATEST_VERSION


def test_migrate_is_independent_of_connect(db_path):
    raw = sqlite3.connect(db_path)
    try:
        assert db.migrate(raw) == [v for v, _, _ in db._MIGRATIONS]
        assert db.migrate(raw) == []
    finally:
        raw.close()


# ---------------------------------------------------------------------------
# schema_version
# ---------------------------------------------------------------------------

def test_schema_version_zero_for_uninitialized_database(tmp_path):
    raw = sqlite3.connect(tmp_path / "empty.db")
    raw.row_factory = sqlite3.Row
    try:
        assert db.schema_version(raw) == 0
    finally:
        raw.close()


# ---------------------------------------------------------------------------
# apply_views
# ---------------------------------------------------------------------------

def test_apply_views_is_reapplyable(conn):
    before = _view_names(conn)
    db.apply_views(conn)
    assert _view_names(conn) == before


def test_apply_views_picks_up_a_changed_definition(conn, monkeypatch, tmp_path):
    # Views are deliberately outside the migration ledger, so re-applying the
    # file must replace an existing definition rather than skip it.
    replacement = tmp_path / "views.sql"
    replacement.write_text(
        "DROP VIEW IF EXISTS jobs_unseen;\n"
        "CREATE VIEW jobs_unseen AS SELECT id FROM jobs WHERE 0;\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(db, "VIEWS_PATH", replacement)
    db.apply_views(conn)
    columns = [r[1] for r in conn.execute("PRAGMA table_info(jobs_unseen)")]
    assert columns == ["id"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _view_names(conn) -> set[str]:
    return {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
    }


# ---------------------------------------------------------------------------
# migrations
# ---------------------------------------------------------------------------

def test_latest_version_matches_the_migration_list():
    assert db.LATEST_VERSION == max(v for v, _, _ in db._MIGRATIONS)


def test_every_migration_script_exists():
    # A typo in a path would otherwise fail only on a real upgrade.
    for version, name, path in db._MIGRATIONS:
        assert path.exists(), f"migration {version} ({name}) missing: {path}"


def test_migration_versions_are_unique_and_ordered():
    versions = [v for v, _, _ in db._MIGRATIONS]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


def test_a_fresh_database_reaches_the_latest_version(db_path):
    conn = db.connect(db_path)
    try:
        assert db.schema_version(conn) == db.LATEST_VERSION
    finally:
        conn.close()


def test_migration_2_creates_the_composite_index(db_path):
    conn = db.connect(db_path)
    try:
        names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    finally:
        conn.close()
    assert "idx_jobs_status_seen" in names


def test_upgrading_an_older_database_applies_only_what_is_missing(db_path):
    # Simulate a database created before migration 2, then upgrade it.
    raw = sqlite3.connect(db_path)
    raw.executescript(db.SCHEMA_PATH.read_text(encoding="utf-8"))
    raw.execute(db._CREATE_MIGRATIONS)
    raw.execute(
        "INSERT INTO schema_migrations (version, name, applied_at)"
        " VALUES (1, 'schema', '2026-01-01T00:00:00Z')"
    )
    raw.commit()
    assert db.schema_version(raw) == 1
    raw.close()

    conn = db.connect(db_path)
    try:
        assert db.schema_version(conn) == db.LATEST_VERSION
        # Every migration ran, in order, and nothing ran twice.
        applied = [
            row[0] for row in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ]
        assert applied == [v for v, _, _ in db._MIGRATIONS]
    finally:
        conn.close()


def test_upgrading_is_idempotent(db_path):
    first = db.connect(db_path)
    first.close()
    second = db.connect(db_path)
    try:
        assert db.migrate(second) == []
    finally:
        second.close()


def test_the_composite_index_serves_the_unseen_read(db_path):
    # The reason the index exists: without it SQLite sorts every matching row in
    # a temp B-tree before applying LIMIT, which took seconds at ~14k rows.
    conn = db.connect(db_path)
    try:
        plan = "\n".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM jobs_unseen LIMIT 300"
            )
        )
    finally:
        conn.close()
    assert "idx_jobs_status_seen" in plan
    assert "TEMP B-TREE" not in plan.upper()


def test_stale_view_uses_indexes_not_a_full_scan(db_path):
    # Regression: SQLite 3.45.1 stopped applying the "OR optimization" to the
    # single-OR form of this view and fell back to a full index scan evaluating
    # julianday() per row -- 20 seconds at 14k rows, versus 29 ms on 3.40.1 with
    # the same database. The view is written as a UNION now; both branches must
    # search an index.
    conn = db.connect(db_path)
    try:
        plan = [
            str(row[3])
            for row in conn.execute("EXPLAIN QUERY PLAN SELECT id FROM jobs_stale")
        ]
    finally:
        conn.close()

    # The anchor subquery legitimately scans jobs once to find MAX(last_seen_at);
    # what must not happen is the *filter itself* scanning. So look for a SEARCH
    # of the jobs table by each branch's index.
    assert "MULTI-INDEX OR" not in "\n".join(plan).upper()
    assert any(
        "SEARCH j USING INDEX idx_jobs_last_seen_at" in line for line in plan
    ), plan
    assert any(
        "SEARCH j USING INDEX idx_jobs_posted_at" in line for line in plan
    ), plan


def test_stale_view_returns_each_job_once(conn):
    # UNION ALL with a guard rather than UNION, so duplicates would show up as
    # repeated rows rather than being silently collapsed.
    now = db.utcnow()
    conn.execute(
        "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
        " last_seen_at, content_hash) VALUES ('a','1','Acme','Eng', ?, ?, 'h')",
        (now, now),
    )
    conn.commit()
    ids = [row[0] for row in conn.execute("SELECT id FROM jobs_stale")]
    assert len(ids) == len(set(ids))


def test_stale_view_is_ordered_by_last_seen(conn):
    # A limit without an order would return arbitrary rows, and callers do pass
    # a limit.
    now = db.utcnow()
    for i, seen in enumerate(("2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z")):
        conn.execute(
            "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
            " last_seen_at, content_hash) VALUES ('a',?,'Acme','Eng', ?, ?, 'h')",
            (str(i), now, seen),
        )
    conn.commit()
    conn.execute("UPDATE jobs SET last_seen_at = '2099-01-01T00:00:00Z' WHERE external_id='0'")
    conn.commit()
    seen_order = [
        row[0] for row in conn.execute("SELECT last_seen_at FROM jobs_stale")
    ]
    assert seen_order == sorted(seen_order)


def test_stale_view_matches_the_documented_rule(conn):
    # Asserted against a hand-written equivalent of the original OR, so a
    # rewrite that changes which rows qualify fails here.
    now = db.utcnow()
    rows = [
        # (external_id, status, last_seen_at, first_seen_at, posted_at)
        ("fresh", "new", now, now, now),
        ("old", "new", "2020-01-01T00:00:00Z", now, None),
        ("applied", "applied", "2020-01-01T00:00:00Z", now, None),
        ("never_reseen", "new", "2020-01-01T00:00:00Z",
         "2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),
    ]
    for external_id, status, seen, first, posted in rows:
        conn.execute(
            "INSERT INTO jobs (source, external_id, company, title, first_seen_at,"
            " last_seen_at, posted_at, status, content_hash)"
            " VALUES ('a', ?, 'Acme', 'Eng', ?, ?, ?, ?, 'h')",
            (external_id, first, seen, posted, status),
        )
    conn.commit()
    conn.execute("UPDATE jobs SET last_seen_at = ? WHERE external_id = 'fresh'", (now,))
    conn.commit()

    got = {
        row[0] for row in conn.execute(
            "SELECT external_id FROM jobs_stale"
        )
    }
    expected = {"old", "never_reseen"}
    assert got == expected, f"got {got}, expected {expected}"


def test_migration_3_creates_the_first_seen_index(db_path):
    conn = db.connect(db_path)
    try:
        names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    finally:
        conn.close()
    assert "idx_jobs_first_seen_at" in names


def test_today_view_is_served_by_an_index(db_path):
    # Without it SQLite sorted every matching row in a temp B-tree before
    # applying LIMIT: 590 ms to return 300 rows on 14k rows, 1.8 s with an
    # offset.
    conn = db.connect(db_path)
    try:
        plan = "\n".join(
            str(row[3])
            for row in conn.execute("EXPLAIN QUERY PLAN SELECT * FROM jobs_today LIMIT 300")
        )
    finally:
        conn.close()
    assert "idx_jobs_first_seen_at" in plan
    # The anchor subquery still scans once to find MAX(last_seen_at); what must
    # not happen is the ordered read scanning the jobs table.
    assert "SCAN j" not in plan.replace("SCAN jobs USING COVERING INDEX", "")
