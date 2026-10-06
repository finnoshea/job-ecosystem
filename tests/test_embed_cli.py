"""Tests for jobecosystem.ingest.embed_cli.

The CLI is a cron interface, so the tests care about the exit code and that
success stays quiet. The embedder is injected, so no model is loaded.
"""

from __future__ import annotations

import io

import pytest

from jobecosystem.core import embedder
from jobecosystem.core.models import Job
from jobecosystem.ingest import embed_cli, upsert


def add_job(conn, external_id, description="body text"):
    result = upsert.upsert_job(
        conn,
        Job(
            source="ashby:ramp",
            external_id=external_id,
            company="Acme",
            title="Engineer",
            description=description,
        ),
    )
    conn.commit()
    return result.job_id


def vector_embed(texts):
    return [[1.0, 0.0, 0.0] for _ in texts]


def parse(argv):
    return embed_cli.build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------

def test_parser_defaults():
    args = parse([])
    assert args.limit == 200
    assert args.db is None
    assert args.batch_size == 32
    assert args.model == embedder.DEFAULT_MODEL
    assert args.force is False
    assert args.quiet is False


def test_limit_accepts_the_long_and_short_form():
    assert parse(["--limit", "5"]).limit == 5
    assert parse(["-n", "7"]).limit == 7


def test_help_mentions_the_database_env_var():
    assert "DB_PATH" in embed_cli.build_parser().format_help()


def test_unknown_argument_is_rejected():
    with pytest.raises(SystemExit):
        parse(["--nope"])


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def test_nothing_to_do_exits_ok_with_a_zero_summary(conn, db_path):
    out, err = io.StringIO(), io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(db_path)]), out=out, err=err, embed=vector_embed
    )
    assert code == embed_cli.EXIT_OK
    assert err.getvalue() == ""
    assert "embedded 0 of 0" in out.getvalue()


def test_embeds_and_reports_the_counts(conn, db_path):
    for i in range(3):
        add_job(conn, str(i))

    out = io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(db_path)]), out=out, err=io.StringIO(), embed=vector_embed
    )

    assert code == embed_cli.EXIT_OK
    assert "embedded 3 of 3 (3 written, 0 failed, 0 tokens)" in out.getvalue()
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 3


def test_records_the_requested_model(conn, db_path):
    add_job(conn, "1")
    embed_cli.run(
        parse(["--db", str(db_path), "--model", "custom-model"]),
        out=io.StringIO(), err=io.StringIO(), embed=vector_embed,
    )
    row = conn.execute("SELECT model FROM job_embeddings").fetchone()
    assert row["model"] == "custom-model"


def test_limit_bounds_the_run(conn, db_path):
    for i in range(5):
        add_job(conn, str(i))
    embed_cli.run(
        parse(["--db", str(db_path), "--limit", "2"]),
        out=io.StringIO(), err=io.StringIO(), embed=vector_embed,
    )
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 2


def test_a_failure_exits_one_and_reports_to_stderr(conn, db_path):
    add_job(conn, "1")

    def boom(texts):
        raise RuntimeError("model exploded")

    out, err = io.StringIO(), io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(db_path)]), out=out, err=err, embed=boom
    )

    assert code == embed_cli.EXIT_EMBEDDING_FAILED
    assert "1 embedding(s) failed" in err.getvalue()
    assert "model exploded" in err.getvalue()
    assert "0 failed" not in out.getvalue()


def test_quiet_prints_nothing_on_success(conn, db_path):
    add_job(conn, "1")
    out = io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(db_path), "--quiet"]),
        out=out, err=io.StringIO(), embed=vector_embed,
    )
    assert code == embed_cli.EXIT_OK
    assert out.getvalue() == ""


def test_quiet_still_reports_failures(conn, db_path):
    add_job(conn, "1")

    def boom(texts):
        raise RuntimeError("nope")

    err = io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(db_path), "--quiet"]),
        out=io.StringIO(), err=err, embed=boom,
    )
    assert code == embed_cli.EXIT_EMBEDDING_FAILED
    assert "1 embedding(s) failed" in err.getvalue()


# ---------------------------------------------------------------------------
# setup failures
# ---------------------------------------------------------------------------

def test_a_negative_limit_is_a_setup_failure(db_path):
    err = io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(db_path), "--limit", "-1"]),
        out=io.StringIO(), err=err, embed=vector_embed,
    )
    assert code == embed_cli.EXIT_SETUP_FAILED
    assert "--limit must be" in err.getvalue()


def test_a_zero_batch_size_is_a_setup_failure(db_path):
    err = io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(db_path), "--batch-size", "0"]),
        out=io.StringIO(), err=err, embed=vector_embed,
    )
    assert code == embed_cli.EXIT_SETUP_FAILED
    assert "--batch-size must be" in err.getvalue()


def test_a_bad_database_path_is_a_setup_failure(tmp_path):
    blocked = tmp_path / "adir"
    blocked.mkdir()
    err = io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(blocked)]), out=io.StringIO(), err=err, embed=vector_embed
    )
    assert code == embed_cli.EXIT_SETUP_FAILED
    assert "could not open" in err.getvalue()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_missing_api_key_is_a_setup_failure(db_path, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    err = io.StringIO()
    code = embed_cli.run(
        parse(["--db", str(db_path)]), out=io.StringIO(), err=err
    )
    assert code == embed_cli.EXIT_SETUP_FAILED
    assert "VOYAGE_API_KEY" in err.getvalue()


def test_main_parses_argv_and_returns_an_exit_code(db_path, monkeypatch):
    # Nothing pending, so no API call happens even on the real path.
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    assert embed_cli.main(["--db", str(db_path)]) == embed_cli.EXIT_OK


def test_main_is_callable_without_arguments():
    import inspect

    assert not [
        p for p in inspect.signature(embed_cli.main).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind is inspect.POSITIONAL_OR_KEYWORD
    ]
