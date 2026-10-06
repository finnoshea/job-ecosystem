"""Tests for jobecosystem.ingest.describe_cli.

The CLI is a cron interface, so the tests care about three things: it never
prompts, its exit code reflects failure, and its output stays quiet on success.
Everything runs through ``run(args)`` with a stubbed HTTP layer, so no network
is involved.
"""

from __future__ import annotations

import io

import pytest

from jobecosystem.core.models import Job
from jobecosystem.ingest import describe_cli, upsert
from jobecosystem.ingest.sources import description as desc

WORKDAY_DETAIL = {"jobPostingInfo": {"jobDescription": "<p>Body</p>"}}


def store(conn, external_id="1", source="workday:x:S", url="https://x.test/1"):
    """Insert a description-less row of a source that has a parser."""
    result = upsert.upsert_job(
        conn,
        Job(
            source=source,
            external_id=external_id,
            company="Acme",
            title="Engineer",
            description=None,
            description_url=url,
        ),
    )
    conn.commit()
    return result.job_id


def parse(argv):
    return describe_cli.build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------

def test_parser_defaults():
    args = parse([])
    assert args.limit == 200
    assert args.db is None
    assert args.delay == desc.DEFAULT_DELAY
    assert args.jitter == desc.DEFAULT_JITTER
    assert args.force is False
    assert args.quiet is False


def test_limit_accepts_the_long_and_short_form():
    assert parse(["--limit", "5"]).limit == 5
    assert parse(["-n", "7"]).limit == 7


def test_help_mentions_the_database_env_var():
    assert "DB_PATH" in describe_cli.build_parser().format_help()


def test_unknown_argument_is_rejected():
    with pytest.raises(SystemExit):
        parse(["--nope"])


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def test_nothing_to_do_exits_ok_with_a_zero_summary(conn, db_path):
    out, err = io.StringIO(), io.StringIO()
    code = describe_cli.run(parse(["--db", str(db_path)]), out=out, err=err)
    assert code == describe_cli.EXIT_OK
    assert err.getvalue() == ""
    assert "described 0 of 0" in out.getvalue()


def test_fetches_and_reports_the_counts(conn, db_path, monkeypatch):
    monkeypatch.setattr(desc, "_http_get_json", lambda url, **kw: WORKDAY_DETAIL)
    store(conn, "1")
    store(conn, "2", url="https://x.test/2")

    out = io.StringIO()
    code = describe_cli.run(parse(["--db", str(db_path)]), out=out, err=io.StringIO())

    assert code == describe_cli.EXIT_OK
    assert "described 2 of 2 (2 written, 0 skipped, 0 failed)" in out.getvalue()
    described = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE description IS NOT NULL"
    ).fetchone()[0]
    assert described == 2


def test_limit_bounds_the_run(conn, db_path, monkeypatch):
    monkeypatch.setattr(desc, "_http_get_json", lambda url, **kw: WORKDAY_DETAIL)
    for i in range(3):
        store(conn, str(i), url=f"https://x.test/{i}")

    describe_cli.run(
        parse(["--db", str(db_path), "--limit", "2"]),
        out=io.StringIO(), err=io.StringIO(),
    )

    described = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE description IS NOT NULL"
    ).fetchone()[0]
    assert described == 2


def test_a_failure_exits_one_and_reports_to_stderr(conn, db_path, monkeypatch):
    def boom(url, **kw):
        raise desc.DescriptionError("HTTP 503")

    monkeypatch.setattr(desc, "_http_get_json", boom)
    store(conn)

    out, err = io.StringIO(), io.StringIO()
    code = describe_cli.run(parse(["--db", str(db_path)]), out=out, err=err)

    assert code == describe_cli.EXIT_DESCRIPTION_FAILED
    assert "1 description(s) failed" in err.getvalue()
    assert "job 1" in err.getvalue()
    assert "1 failed" in out.getvalue()


def test_quiet_prints_nothing_on_success(conn, db_path, monkeypatch):
    monkeypatch.setattr(desc, "_http_get_json", lambda url, **kw: WORKDAY_DETAIL)
    store(conn)
    out = io.StringIO()
    code = describe_cli.run(
        parse(["--db", str(db_path), "--quiet"]), out=out, err=io.StringIO()
    )
    assert code == describe_cli.EXIT_OK
    assert out.getvalue() == ""


def test_quiet_still_reports_failures_to_stderr(conn, db_path, monkeypatch):
    def boom(url, **kw):
        raise desc.DescriptionError("HTTP 503")

    monkeypatch.setattr(desc, "_http_get_json", boom)
    store(conn)
    err = io.StringIO()
    code = describe_cli.run(
        parse(["--db", str(db_path), "--quiet"]),
        out=io.StringIO(), err=err,
    )
    assert code == describe_cli.EXIT_DESCRIPTION_FAILED
    assert "1 description(s) failed" in err.getvalue()


# ---------------------------------------------------------------------------
# setup failures
# ---------------------------------------------------------------------------

def test_a_negative_limit_is_a_setup_failure(db_path):
    err = io.StringIO()
    code = describe_cli.run(
        parse(["--db", str(db_path), "--limit", "-1"]),
        out=io.StringIO(), err=err,
    )
    assert code == describe_cli.EXIT_SETUP_FAILED
    assert "--limit must be" in err.getvalue()


def test_a_bad_database_path_is_a_setup_failure(tmp_path):
    blocked = tmp_path / "adir"
    blocked.mkdir()
    err = io.StringIO()
    code = describe_cli.run(parse(["--db", str(blocked)]), out=io.StringIO(), err=err)
    assert code == describe_cli.EXIT_SETUP_FAILED
    assert "could not open" in err.getvalue()


def test_database_path_argument_wins_over_the_environment(
    conn, tmp_path, monkeypatch
):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "from-env.db"))
    explicit = tmp_path / "explicit.db"
    describe_cli.run(
        parse(["--db", str(explicit)]), out=io.StringIO(), err=io.StringIO()
    )
    assert explicit.exists()
    assert not (tmp_path / "from-env.db").exists()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_parses_argv_and_returns_an_exit_code(db_path):
    assert describe_cli.main(["--db", str(db_path)]) == describe_cli.EXIT_OK


def test_main_is_callable_without_arguments():
    import inspect

    assert not [
        p for p in inspect.signature(describe_cli.main).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind is p.POSITIONAL_OR_KEYWORD
    ]
