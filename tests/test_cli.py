"""Tests for jobecosystem.ingest.cli.

The CLI is the cron interface, so the tests care about three things: it never
prompts, its exit code reflects failure, and it does not need a real terminal.
Everything runs through ``run(args)`` rather than a subprocess, with a stub
poster so no network is involved.
"""

from __future__ import annotations

import io
import sqlite3

import pytest

from jobecosystem.ingest import cli


@pytest.fixture(autouse=True)
def _isolate_smartrecruiters_config(tmp_path, monkeypatch):
    """Keep the repo's real company lists out of every CLI test.

    These tests pass explicit board/tenant files, so without this the default
    smartrecruiters_companies.txt and greenhouse_companies.txt would still be
    read from the repo root and add unexpected sources to otherwise-unrelated
    selections.
    """
    monkeypatch.setenv(
        "SMARTRECRUITERS_COMPANIES_FILE", str(tmp_path / "no-companies.txt")
    )
    monkeypatch.setenv(
        "GREENHOUSE_BOARDS_FILE", str(tmp_path / "no-boards.txt")
    )
    monkeypatch.setenv("LEVER_COMPANIES_FILE", str(tmp_path / "no-sites.txt"))


@pytest.fixture(autouse=True)
def _offline_workable(monkeypatch):
    """Keep the Workable feed off the network in every CLI test.

    Workable has no config file, so it is on by default; a test that means to
    exercise Ashby or Workday must not reach the marketplace through it. Tests
    about Workable override this stub themselves.
    """
    from jobecosystem.ingest.sources import workable

    monkeypatch.setattr(
        workable, "_http_get_json", lambda url, **kw: {"jobs": []}
    )


def make_args(**overrides):
    """Parsed-argument namespace with everything defaulted to 'off'."""
    defaults = {
        "db": None,
        "source": None,
        "ashby_boards": None,
        "workday_tenants": None,
        "smartrecruiters_companies": None,
        "greenhouse_companies": None,
        "lever_companies": None,
        "workable_pages": cli.WORKABLE_DEFAULT_PAGES,
        "workable_query": None,
        "workable_location": None,
        "workable_workplace": None,
        "quiet": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return cli.build_parser().parse_args([]) if False else _namespace(**defaults)


def _namespace(**values):
    class NS:
        pass

    ns = NS()
    for key, value in values.items():
        setattr(ns, key, value)
    return ns


def write_configs(tmp_path, *, ashby="", workday=""):
    """Write board/tenant files and return their paths."""
    boards = tmp_path / "ashby_boards.txt"
    tenants = tmp_path / "workday_tenants.txt"
    boards.write_text(ashby, encoding="utf-8")
    tenants.write_text(workday, encoding="utf-8")
    return boards, tenants


ASHBY_PAYLOAD = {
    "jobs": [
        {"id": "a1", "title": "Engineer", "isListed": True,
         "descriptionPlain": "Body", "jobUrl": "https://x.test/a1"},
    ]
}

WORKDAY_PAGE = {
    "total": 1,
    "jobPostings": [
        {"title": "Engineer", "externalPath": "/job/X/E_1", "bulletFields": ["J-1"]},
    ],
}


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------

def test_parser_defaults_are_all_off():
    args = cli.build_parser().parse_args([])
    assert args.db is None
    assert args.source is None
    assert args.quiet is False
    assert args.dry_run is False
    assert args.smartrecruiters_companies is None
    assert args.greenhouse_companies is None
    assert args.lever_companies is None
    assert args.workable_pages == cli.WORKABLE_DEFAULT_PAGES


def test_parser_accepts_workable_pages_and_filters():
    args = cli.build_parser().parse_args([
        "--workable-pages", "5",
        "--workable-query", "engineer",
        "--workable-location", "Remote",
        "--workable-workplace", "remote",
    ])
    assert args.workable_pages == 5
    assert args.workable_query == "engineer"
    assert args.workable_location == "Remote"
    assert args.workable_workplace == "remote"


def test_parser_accepts_a_database_path():
    assert cli.build_parser().parse_args(["--db", "/tmp/x.db"]).db == "/tmp/x.db"


def test_source_filter_is_repeatable():
    args = cli.build_parser().parse_args(["--source", "ashby", "--source", "workday"])
    assert args.source == ["ashby", "workday"]


def test_unknown_argument_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--nope"])


def test_help_mentions_the_database_env_var():
    # The default resolution order should be discoverable from --help.
    text = cli.build_parser().format_help()
    assert "DB_PATH" in text


# ---------------------------------------------------------------------------
# source selection
# ---------------------------------------------------------------------------

def test_selects_every_source_by_default(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    selection = cli.select_scrapers(
        make_args(ashby_boards=str(boards), workday_tenants=str(tenants))
    )
    assert [s.source for s in selection.scrapers] == [
        "ashby:ramp", "workable", "workday:t:S"
    ]
    assert selection.errors == []


def test_source_filter_keeps_only_ashby(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    selection = cli.select_scrapers(
        make_args(source=["ashby"], ashby_boards=str(boards),
                  workday_tenants=str(tenants))
    )
    assert [s.source for s in selection.scrapers] == ["ashby:ramp"]


def test_source_filter_keeps_only_workday(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    selection = cli.select_scrapers(
        make_args(source=["workday"], ashby_boards=str(boards),
                  workday_tenants=str(tenants))
    )
    assert [s.source for s in selection.scrapers] == ["workday:t:S"]


def test_selects_smartrecruiters_companies(tmp_path):
    boards, tenants = write_configs(tmp_path)
    companies = tmp_path / "smartrecruiters_companies.txt"
    companies.write_text("RedBull us Red Bull\n", encoding="utf-8")
    selection = cli.select_scrapers(
        make_args(ashby_boards=str(boards), workday_tenants=str(tenants),
                  smartrecruiters_companies=str(companies))
    )
    assert [s.source for s in selection.scrapers] == [
        "smartrecruiters:RedBull", "workable"
    ]


def test_selects_greenhouse_companies(tmp_path):
    boards, tenants = write_configs(tmp_path)
    companies = tmp_path / "greenhouse_companies.txt"
    companies.write_text("motional Motional\n", encoding="utf-8")
    selection = cli.select_scrapers(
        make_args(ashby_boards=str(boards), workday_tenants=str(tenants),
                  greenhouse_companies=str(companies))
    )
    assert [s.source for s in selection.scrapers] == [
        "greenhouse:motional", "workable"
    ]


def test_workday_runs_last(tmp_path):
    # Fetch order is Ashby, SmartRecruiters, Greenhouse, Workday -- Workday is
    # slowest and most fragile, so it must not delay the other sources'
    # streaming lines.
    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    companies = tmp_path / "smartrecruiters_companies.txt"
    companies.write_text("RedBull us Red Bull\n", encoding="utf-8")
    gh = tmp_path / "greenhouse_companies.txt"
    gh.write_text("motional Motional\n", encoding="utf-8")
    selection = cli.select_scrapers(
        make_args(ashby_boards=str(boards), workday_tenants=str(tenants),
                  smartrecruiters_companies=str(companies),
                  greenhouse_companies=str(gh))
    )
    assert [s.source for s in selection.scrapers] == [
        "ashby:ramp", "smartrecruiters:RedBull", "greenhouse:motional",
        "workable", "workday:t:S",
    ]


def test_source_filter_keeps_only_greenhouse(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    gh = tmp_path / "greenhouse_companies.txt"
    gh.write_text("motional Motional\n", encoding="utf-8")
    selection = cli.select_scrapers(
        make_args(source=["greenhouse"], ashby_boards=str(boards),
                  workday_tenants=str(tenants), greenhouse_companies=str(gh))
    )
    assert [s.source for s in selection.scrapers] == ["greenhouse:motional"]


def test_source_filter_keeps_only_smartrecruiters(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    companies = tmp_path / "smartrecruiters_companies.txt"
    companies.write_text("RedBull us Red Bull\n", encoding="utf-8")
    selection = cli.select_scrapers(
        make_args(source=["smartrecruiters"], ashby_boards=str(boards),
                  workday_tenants=str(tenants),
                  smartrecruiters_companies=str(companies))
    )
    assert [s.source for s in selection.scrapers] == ["smartrecruiters:RedBull"]


def test_source_filter_matches_a_full_prefix(tmp_path):
    boards, tenants = write_configs(tmp_path, workday="h.wd1 t Site N\nh.wd1 u Other M\n")
    selection = cli.select_scrapers(
        make_args(source=["workday:t:Site"], ashby_boards=str(boards),
                  workday_tenants=str(tenants))
    )
    assert [s.source for s in selection.scrapers] == ["workday:t:Site"]


def test_a_filter_matching_nothing_is_reported(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    selection = cli.select_scrapers(
        make_args(source=["bogus"], ashby_boards=str(boards),
                  workday_tenants=str(tenants))
    )
    assert selection.scrapers == []
    assert len(selection.errors) == 1
    assert "bogus" in selection.errors[0]


def test_empty_config_files_yield_only_the_workable_aggregate(tmp_path):
    boards, tenants = write_configs(tmp_path)
    selection = cli.select_scrapers(
        make_args(ashby_boards=str(boards), workday_tenants=str(tenants))
    )
    # Workable needs no config file, so it is the one source still present.
    assert [s.source for s in selection.scrapers] == ["workable"]
    assert selection.errors == []


def test_selects_lever_companies(tmp_path):
    boards, tenants = write_configs(tmp_path)
    sites = tmp_path / "lever_companies.txt"
    sites.write_text("acme Acme Corp\n", encoding="utf-8")
    selection = cli.select_scrapers(
        make_args(source=["lever"], ashby_boards=str(boards),
                  workday_tenants=str(tenants), lever_companies=str(sites))
    )
    assert [s.source for s in selection.scrapers] == ["lever:acme"]


def test_source_filter_keeps_only_workable(tmp_path):
    boards, tenants = write_configs(tmp_path)
    selection = cli.select_scrapers(
        make_args(source=["workable"], ashby_boards=str(boards),
                  workday_tenants=str(tenants))
    )
    assert [s.source for s in selection.scrapers] == ["workable"]


def test_workable_filters_and_page_budget_reach_the_scraper(tmp_path):
    boards, tenants = write_configs(tmp_path)
    selection = cli.select_scrapers(
        make_args(source=["workable"], ashby_boards=str(boards),
                  workday_tenants=str(tenants), workable_pages=3,
                  workable_query="engineer", workable_workplace="remote")
    )
    (scraper,) = selection.scrapers
    assert scraper.pages == 3
    assert scraper.query == "engineer"
    assert scraper.workplace == "remote"
    assert "query=engineer" in scraper.page_url()
    assert "workplace=remote" in scraper.page_url()


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------

def test_dry_run_lists_sources_and_exits_ok(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(
        make_args(db=str(tmp_path / "jobs.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants), dry_run=True),
        out=out, err=err,
    )
    assert code == cli.EXIT_OK
    assert "would scrape 2 source(s)" in out.getvalue()
    assert "ashby:ramp" in out.getvalue()


def test_dry_run_creates_the_database(tmp_path):
    # It opens a connection before deciding, which is harmless and means a
    # permissions problem surfaces here rather than mid-scrape.
    path = tmp_path / "jobs.db"
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    cli.run(
        make_args(db=str(path), ashby_boards=str(boards),
                  workday_tenants=str(tenants), dry_run=True),
        out=io.StringIO(), err=io.StringIO(),
    )
    assert path.exists()


def test_dry_run_reports_a_bad_filter(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    err = io.StringIO()
    code = cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants), dry_run=True, source=["bogus"]),
        out=io.StringIO(), err=err,
    )
    assert code == cli.EXIT_SETUP_FAILED
    assert "no sources matched" in err.getvalue()


def test_dry_run_quiet_prints_nothing_on_success(tmp_path):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    out = io.StringIO()
    cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants), dry_run=True, quiet=True),
        out=out, err=io.StringIO(),
    )
    assert out.getvalue() == ""


# ---------------------------------------------------------------------------
# no sources
# ---------------------------------------------------------------------------

def test_no_sources_configured_is_a_setup_failure(tmp_path):
    boards, tenants = write_configs(tmp_path)
    err = io.StringIO()
    # Restrict to families with no config; otherwise the config-free Workable
    # source would keep the run alive.
    code = cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants), source=["ashby", "workday"]),
        out=io.StringIO(), err=err,
    )
    assert code == cli.EXIT_SETUP_FAILED
    assert "no sources configured" in err.getvalue()


def test_a_bad_database_path_is_a_setup_failure(tmp_path):
    err = io.StringIO()
    # A directory where the file should be: sqlite cannot open it.
    blocked = tmp_path / "adir"
    blocked.mkdir()
    code = cli.run(
        make_args(db=str(blocked), ashby_boards=str(tmp_path / "n.txt"),
                  workday_tenants=str(tmp_path / "n2.txt")),
        out=io.StringIO(), err=err,
    )
    assert code == cli.EXIT_SETUP_FAILED
    assert "could not open" in err.getvalue()


# ---------------------------------------------------------------------------
# a real run, against stubbed HTTP
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_network(monkeypatch):
    """Replace the network layer for both sources."""
    from jobecosystem.ingest.sources import ashby, workday

    monkeypatch.setattr(ashby, "_http_get_json", lambda url, **kw: ASHBY_PAYLOAD)
    monkeypatch.setattr(workday, "_http_post_json", lambda url, body, **kw: WORKDAY_PAGE)


def test_run_scrapes_and_exits_zero(tmp_path, stub_network):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=out, err=err,
    )
    assert code == cli.EXIT_OK
    assert err.getvalue() == ""
    assert "ashby:ramp: ok" in out.getvalue()
    assert "3 source(s)" in out.getvalue()


def test_per_source_lines_are_printed_before_the_totals(tmp_path, stub_network):
    # The streaming reporter's point: a source's line lands as it finishes,
    # ahead of the run-wide totals line that only prints at the end.
    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    out = io.StringIO()
    cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=out, err=io.StringIO(),
    )
    text = out.getvalue()
    assert text.index("ashby:ramp: ok") < text.index("source(s):")


def test_run_stores_jobs(tmp_path, stub_network):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    db_path = tmp_path / "j.db"
    cli.run(
        make_args(db=str(db_path), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=io.StringIO(), err=io.StringIO(),
    )
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        # One run per source: Ashby, the Workable aggregate, and (empty) Workday
        # is not configured here, so two.
        assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 2
    finally:
        conn.close()


def test_run_summary_reports_new_jobs(tmp_path, stub_network):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    out = io.StringIO()
    cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=out, err=io.StringIO(),
    )
    assert "1 fetched, 1 new" in out.getvalue()


def test_summary_counts_failed_sources_separately(tmp_path, monkeypatch):
    # A fetch failure is not a listing error, and reporting "0 errors" beside a
    # FAILED line would be misleading.
    from jobecosystem.ingest.base import FetchError
    from jobecosystem.ingest.sources import ashby

    def boom(url, **kwargs):
        raise FetchError("refused")

    monkeypatch.setattr(ashby, "_http_get_json", boom)
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    out = io.StringIO()
    cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=out, err=io.StringIO(),
    )
    summary = out.getvalue()
    assert "1 failed source(s)" in summary
    assert "0 listing error(s)" in summary


def test_quiet_suppresses_the_summary(tmp_path, stub_network):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    out = io.StringIO()
    cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants), quiet=True),
        out=out, err=io.StringIO(),
    )
    assert out.getvalue() == ""


def test_a_failing_source_exits_one(tmp_path, monkeypatch):
    from jobecosystem.ingest.sources import ashby
    from jobecosystem.ingest.base import FetchError

    def boom(url, **kwargs):
        raise FetchError("connection refused")

    monkeypatch.setattr(ashby, "_http_get_json", boom)
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    out, err = io.StringIO(), io.StringIO()
    code = cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=out, err=err,
    )
    assert code == cli.EXIT_SOURCE_FAILED
    assert "FAILED" in out.getvalue()
    assert "1 source(s) failed" in err.getvalue()


def test_a_failing_source_still_records_a_run(tmp_path, monkeypatch):
    # The scheduler must be able to see that it tried.
    from jobecosystem.ingest.sources import ashby
    from jobecosystem.ingest.base import FetchError

    monkeypatch.setattr(
        ashby, "_http_get_json",
        lambda url, **kw: (_ for _ in ()).throw(FetchError("refused")),
    )
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    db_path = tmp_path / "j.db"
    cli.run(
        make_args(db=str(db_path), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=io.StringIO(), err=io.StringIO(),
    )
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT source, errors FROM scrape_runs").fetchone()
        assert row[0] == "ashby:ramp"
        assert row[1] == 1
    finally:
        conn.close()


def test_one_failing_source_does_not_stop_the_other(tmp_path, monkeypatch):
    from jobecosystem.ingest.base import FetchError
    from jobecosystem.ingest.sources import ashby, workday

    def boom(url, **kwargs):
        raise FetchError("refused")

    monkeypatch.setattr(ashby, "_http_get_json", boom)
    monkeypatch.setattr(workday, "_http_post_json", lambda url, body, **kw: WORKDAY_PAGE)

    boards, tenants = write_configs(tmp_path, ashby="ramp\n", workday="h.wd1 t S N\n")
    code = cli.run(
        make_args(db=str(tmp_path / "j.db"), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=io.StringIO(), err=io.StringIO(),
    )
    assert code == cli.EXIT_SOURCE_FAILED
    conn = sqlite3.connect(tmp_path / "j.db")
    try:
        # The Workday job still landed.
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert conn.execute(
            "SELECT source FROM jobs"
        ).fetchone()[0].startswith("workday:")
    finally:
        conn.close()


def test_database_path_argument_wins_over_the_environment(tmp_path, monkeypatch, stub_network):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "from-env.db"))
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    explicit = tmp_path / "explicit.db"
    cli.run(
        make_args(db=str(explicit), ashby_boards=str(boards),
                  workday_tenants=str(tenants)),
        out=io.StringIO(), err=io.StringIO(),
    )
    assert explicit.exists()
    assert not (tmp_path / "from-env.db").exists()


def test_rerunning_is_idempotent(tmp_path, stub_network):
    boards, tenants = write_configs(tmp_path, ashby="ramp\n")
    db_path = tmp_path / "j.db"
    for _ in range(2):
        cli.run(
            make_args(db=str(db_path), ashby_boards=str(boards),
                      workday_tenants=str(tenants)),
            out=io.StringIO(), err=io.StringIO(),
        )
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        # Two sources (Ashby + Workable) across two runs.
        assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 4
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_parses_argv_and_returns_an_exit_code(tmp_path):
    boards, tenants = write_configs(tmp_path)
    code = cli.main([
        "--db", str(tmp_path / "j.db"),
        "--ashby-boards", str(boards),
        "--workday-tenants", str(tenants),
        "--source", "ashby",
    ])
    assert code == cli.EXIT_SETUP_FAILED     # nothing configured


def test_main_is_callable_without_arguments():
    # The generated console script calls it bare; --help must be the only way
    # to get a SystemExit out of it in that case.
    import inspect

    assert not [
        p for p in inspect.signature(cli.main).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind is p.POSITIONAL_OR_KEYWORD
    ]
