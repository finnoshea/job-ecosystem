"""SQLite connection, configuration, and migration handling.

Single source of truth for where the database lives and how connections are
opened. ``ingest``, ``triage``, and ``tailor`` all obtain connections from
:func:`connect`; nothing outside this module imports :mod:`sqlite3` directly.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# src/jobecosystem/core/db.py -> core -> jobecosystem -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = _REPO_ROOT / "jobs.db"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
VIEWS_PATH = Path(__file__).with_name("views.sql")
_MIGRATIONS_DIR = Path(__file__).with_name("migrations")

# Ordered (version, name, script) migrations. Append new migrations here and
# never edit an entry that has already shipped.
#
# Migration 1 is schema.sql, which uses IF NOT EXISTS throughout so it is
# idempotent. Later migrations live in ``migrations/`` and must be additive:
# adding a column, adding an index, backfilling -- never a destructive change
# that would break a scraper running against the older shape.
_MIGRATIONS: tuple[tuple[int, str, Path], ...] = (
    (1, "schema", SCHEMA_PATH),
    (2, "triage_indexes", _MIGRATIONS_DIR / "002_triage_indexes.sql"),
    (3, "first_seen_index", _MIGRATIONS_DIR / "003_first_seen_index.sql"),
    (4, "description_attempts", _MIGRATIONS_DIR / "004_description_attempts.sql"),
)

#: Highest migration version this code knows about. Tests compare against it so
#: adding a migration does not require editing them.
LATEST_VERSION: int = _MIGRATIONS[-1][0]

# Bootstrapped in Python so the very first migration has a ledger to write to.
_CREATE_MIGRATIONS = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


def utcnow() -> str:
    """Current UTC time as ISO-8601 (``YYYY-MM-DDTHH:MM:SSZ``).

    Every timestamp column stores exactly this format, so both string
    comparison and SQLite's ``date()``/``datetime()`` functions behave.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_db_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the database file: explicit argument > ``DB_PATH`` env > default."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_DB_PATH


def connect(
    path: str | os.PathLike[str] | None = None,
    *,
    run_migration_and_update_views: bool = True,
) -> sqlite3.Connection:
    """Open the database with the project's pragmas applied.

    WAL lets the daily scraper write while the TUI reads. ``foreign_keys`` is
    per-connection, so it must be set here (it is what makes the ``ON DELETE
    CASCADE`` on embeddings/errors actually fire).

    With ``run_migration_and_update_views`` the connection is also brought up
    to the current schema and view definitions; see :func:`migrate` and
    :func:`apply_views` for what that entails. Pass ``False`` only when the
    caller has already done so on this database (or opened it read-only), and
    be aware that subsequent reads may then reference missing columns or
    views.

    **The connection belongs to the thread that opened it.** ``sqlite3``
    enforces this by default and raises ``ProgrammingError: SQLite objects
    created in a thread can only be used in that same thread`` if another thread
    touches it. Do not pass a connection to a worker thread, and do not set
    ``check_same_thread=False`` to work around it: that only silences the check,
    leaving genuine concurrent use to fail later as ``database is locked``.

    The pattern to use instead: do the database work on the owning thread, run
    only the slow, non-database part (an HTTP request, a model call) on the
    worker, and hand the result back for the owning thread to store. See
    :func:`jobecosystem.ingest.sources.description.fetch_description_from_known_url`
    for a fetch that takes no connection for exactly this reason, and the TUI's
    ``d`` binding for the round trip.

    In practice this means one connection per process, with the exception of
    tests. Opening a second connection in another thread is fine -- WAL allows
    one writer alongside readers -- it is *sharing* one connection that is not.
    """
    db_path = resolve_db_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")    # persistent, but cheap to re-assert
    conn.execute("PRAGMA foreign_keys=ON")     # per-connection; required for cascades
    conn.execute("PRAGMA busy_timeout=5000")   # wait on lock contention instead of raising
    conn.execute("PRAGMA synchronous=NORMAL")  # safe under WAL, faster writes

    if run_migration_and_update_views:
        migrate(conn)
        apply_views(conn)
    return conn


def apply_views(conn: sqlite3.Connection) -> None:
    """(Re-)apply ``views.sql``, defining every query view.

    Deliberately outside the versioned migration ledger: views hold no state,
    and re-applying the file on each connect means an edited view definition
    takes effect without a migration entry. ``views.sql`` drops each view
    before recreating it, so this always converges on the current text.

    Runs after :func:`migrate` because a view can reference a column that a
    pending migration adds.

    Note: ``executescript`` issues an implicit ``COMMIT`` before executing the
    script, which is unconditional here (unlike :func:`migrate`, which returns
    early when nothing is pending). Consequences:

    * Never call this on a connection with an open transaction -- the caller's
      uncommitted writes are committed and the transaction is ended, so a
      subsequent ``rollback()`` will not undo them.
    * DDL is transactional in SQLite, but the leading ``COMMIT`` means the
      script does not run as part of whatever transaction was open, and the
      ``DROP``/``CREATE`` pairs are not atomic with each other.
    * On a locked or read-only database this write fails; opened read-only
      connections must skip it (``run_migration_and_update_views=False``).

    In practice ``connect`` calls this once per process before any work
    begins, so the implicit ``COMMIT`` has nothing to commit and the failure
    mode does not arise.
    """
    conn.executescript(VIEWS_PATH.read_text(encoding="utf-8"))


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply every migration not yet recorded; return the versions applied.

    Idempotent: a second call in the same process returns an empty list.
    """
    conn.execute(_CREATE_MIGRATIONS)
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}

    newly_applied: list[int] = []
    for version, name, script in _MIGRATIONS:
        if version in applied:
            continue
        conn.executescript(script.read_text(encoding="utf-8"))
        with conn:
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at)"
                " VALUES (?, ?, ?)",
                (version, name, utcnow()),
            )
        newly_applied.append(version)
    return newly_applied


def schema_version(conn: sqlite3.Connection) -> int:
    """Highest applied migration version, or 0 if the database is uninitialized."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table'"
        " AND name = 'schema_migrations'"
    ).fetchone()
    if exists is None:
        return 0
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    ).fetchone()
    return int(row[0])


def main() -> None:
    """``python -m jobecosystem.core.db`` -- create/upgrade the database."""
    conn = connect()
    try:
        print(f"database: {resolve_db_path()}")
        print(f"schema version: {schema_version(conn)}")
        views = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"
            )
        ]
        print(f"views: {', '.join(views) if views else '(none)'}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
