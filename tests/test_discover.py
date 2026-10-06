"""Tests for jobecosystem.ingest.discover and its CLI.

The harvesters are pure over injected payloads, so nothing here touches the
network; the fixtures are the shapes HN actually returns.
"""

from __future__ import annotations

import io

import pytest

from jobecosystem.ingest import discover
from jobecosystem.ingest import discover_cli as cli


def fake_fetch(routes):
    """A ``fetch_json`` replacement that serves saved payloads by URL substring."""
    calls = []

    def fetch(url, user_agent=None):
        calls.append((url, user_agent))
        for needle, payload in routes.items():
            if needle in url:
                return payload
        raise AssertionError(f"unexpected URL: {url}")

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


# ---------------------------------------------------------------------------
# recognising URLs
# ---------------------------------------------------------------------------

def test_extract_slugs_finds_every_family():
    text = (
        "https://jobs.lever.co/acme and https://apply.workable.com/globex "
        "and https://job-boards.greenhouse.io/initech "
        "and https://jobs.ashbyhq.com/umbrella "
        "and https://jobs.smartrecruiters.com/wayne/12345-something"
    )
    assert discover.extract_slugs(text) == {
        "lever": {"acme"},
        "workable": {"globex"},
        "greenhouse": {"initech"},
        "ashby": {"umbrella"},
        "smartrecruiters": {"wayne"},
    }


def test_extract_slugs_understands_both_workable_hosts():
    found = discover.extract_slugs(
        "https://apply.workable.com/globex/ and https://acme.workable.com/jobs"
    )
    assert found == {"workable": {"globex", "acme"}}


def test_extract_slugs_ignores_lookalike_hosts():
    # A prefix match, a real ATS host with no company, and the aggregator's own
    # site must all yield nothing.
    assert discover.extract_slugs("https://jobs.lever.co.evil.com/acme") == {}
    assert discover.extract_slugs("https://jobs.workable.com/") == {}
    assert discover.extract_slugs("https://www.workable.com") == {}


def test_extract_slugs_decodes_percent_encoding_and_entities():
    assert discover.extract_slugs(
        "https%3A%2F%2Fjobs.lever.co%2Facme"
    ) == {"lever": {"acme"}}
    assert discover.extract_slugs(
        "https://jobs.lever.co/acme&amp;utm=1"
    ) == {"lever": {"acme"}}


def test_extract_slugs_lowercases_so_case_cannot_duplicate():
    assert discover.extract_slugs("https://jobs.lever.co/Acme") == {"lever": {"acme"}}


def test_extract_slugs_handles_empty_input():
    assert discover.extract_slugs(None) == {}
    assert discover.extract_slugs("") == {}


# ---------------------------------------------------------------------------
# HN
# ---------------------------------------------------------------------------

HN_ROUTES = {
    "search?tags=story": {
        "hits": [
            {"objectID": "1", "created_at": "2025-12-01T00:00:00Z",
             "title": "Ask HN: Who is hiring? (December 2025)"},
            {"objectID": "2", "created_at": "2026-01-01T00:00:00Z",
             "title": "Ask HN: Who is hiring? (January 2026)"},
            {"objectID": "3", "created_at": "2026-01-02T00:00:00Z",
             "title": "Ask HN: Who wants to be hired?"},
        ]
    },
    "items/2": {
        "children": [
            {"text": '<p>Acme Robotics | Senior SWE | Remote<br>'
                     '<a href="https://jobs.lever.co/acme-robotics">apply</a></p>'},
            {"text": '<p>Globex | Data | NYC<br>'
                     'apply at https://apply.workable.com/globex/</p>'},
            {"text": None},
            {"text": "<p>No ATS here: https://example.com/careers</p>"},
        ]
    },
}


def test_latest_thread_id_picks_the_newest_hiring_story():
    assert discover.latest_thread_id(HN_ROUTES["search?tags=story"]) == "2"


def test_latest_thread_id_is_none_without_stories():
    assert discover.latest_thread_id({"hits": []}) is None
    assert discover.latest_thread_id("not a dict") is None


def test_harvest_hn_reads_the_newest_thread():
    harvest = discover.harvest_hn(fake_fetch(HN_ROUTES))
    assert harvest.slugs == {"lever": {"acme-robotics"}, "workable": {"globex"}}
    assert harvest.names["lever"]["acme-robotics"] == "Acme Robotics"
    assert harvest.names["workable"]["globex"] == "Globex"


def test_harvest_hn_raises_without_a_thread():
    with pytest.raises(discover.FetchError):
        discover.harvest_hn(fake_fetch({"search?tags=story": {"hits": []}}))


def test_harvest_hn_thread_skips_empty_comments():
    harvest = discover.harvest_hn_thread({"children": [{"text": None}, "junk"]})
    assert not harvest


def test_company_from_comment_strips_a_yc_tag():
    name = discover._company_from_comment("<p>Acme (YC W22) | SWE | Remote</p>")
    assert name == "Acme"


# ---------------------------------------------------------------------------
# source dispatch
# ---------------------------------------------------------------------------

def test_harvest_dispatches_hn_and_rejects_unknown():
    assert discover.harvest("hn", fake_fetch(HN_ROUTES)).slugs
    with pytest.raises(discover.FetchError):
        discover.harvest("nope", fake_fetch({}))


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------

def test_merge_into_file_appends_and_preserves_comments(tmp_path):
    path = tmp_path / "lever_companies.txt"
    path.write_text("# keep me\nacme  Acme\n", encoding="utf-8")

    outcome = discover.merge_into_file(
        path, {"acme", "globex"}, names={"globex": "Globex"}
    )

    assert outcome.added == ["globex"]
    assert outcome.existing == 1
    assert path.read_text(encoding="utf-8") == (
        "# keep me\nacme  Acme\nglobex  Globex\n"
    )


def test_merge_into_file_is_idempotent(tmp_path):
    path = tmp_path / "lever_companies.txt"
    discover.merge_into_file(path, {"acme"})
    assert discover.merge_into_file(path, {"acme"}).added == []


def test_merge_into_file_dry_run_writes_nothing(tmp_path):
    path = tmp_path / "lever_companies.txt"
    outcome = discover.merge_into_file(path, {"acme"}, dry_run=True)
    assert outcome.added == ["acme"]
    assert not path.exists()


def test_merge_into_file_creates_a_missing_file_with_a_header(tmp_path):
    path = tmp_path / "new.txt"
    discover.merge_into_file(path, {"acme"}, header="# discovered\n")
    assert path.read_text(encoding="utf-8").startswith("# discovered\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.sources == "all"
    assert args.family == "all"
    assert args.dry_run is False


def test_run_writes_discovered_slugs_to_the_output_dir(tmp_path):
    args = cli.build_parser().parse_args(
        ["--from", "hn", "--family", "lever", "--output-dir", str(tmp_path)]
    )
    out = io.StringIO()

    code = cli.run(args, out=out, fetch_json=fake_fetch(HN_ROUTES))

    assert code == cli.EXIT_OK
    written = (tmp_path / "lever_companies.txt").read_text(encoding="utf-8")
    assert "acme-robotics  Acme Robotics" in written
    assert "added 1" in out.getvalue()


def test_run_dry_run_writes_nothing(tmp_path):
    args = cli.build_parser().parse_args(
        ["--from", "hn", "--family", "lever", "--dry-run",
         "--output-dir", str(tmp_path)]
    )
    code = cli.run(args, out=io.StringIO(), fetch_json=fake_fetch(HN_ROUTES))
    assert code == cli.EXIT_OK
    assert not (tmp_path / "lever_companies.txt").exists()


def test_run_reports_a_dead_source_but_keeps_going(tmp_path):
    def failing(url, user_agent=None):
        raise discover.FetchError("boom")

    args = cli.build_parser().parse_args(
        ["--from", "hn", "--family", "lever", "--output-dir", str(tmp_path)]
    )
    err = io.StringIO()
    code = cli.run(args, out=io.StringIO(), err=err, fetch_json=failing)

    assert code == cli.EXIT_SOURCE_FAILED
    assert "hn: boom" in err.getvalue()


def test_run_rejects_unknown_sources_and_families(tmp_path):
    bad_source = cli.build_parser().parse_args(["--from", "nope"])
    assert cli.run(bad_source, out=io.StringIO(), err=io.StringIO(),
                   fetch_json=fake_fetch({})) == cli.EXIT_SETUP_FAILED

    bad_family = cli.build_parser().parse_args(["--family", "nope"])
    assert cli.run(bad_family, out=io.StringIO(), err=io.StringIO(),
                   fetch_json=fake_fetch({})) == cli.EXIT_SETUP_FAILED
