"""Tests for jobecosystem.ingest.sources.greenhouse.

Parser tests run against stubbed JSON shaped like a live response for
``motional`` and ``reddit``: listings carry metadata only, never a description.
"""

from __future__ import annotations

import pytest

from jobecosystem.ingest import runner
from jobecosystem.ingest.base import FetchError, ParseError
from jobecosystem.ingest.sources import greenhouse as gh
from jobecosystem.ingest.sources.greenhouse import GreenhouseScraper


def listing(**overrides) -> dict:
    """A listing with only the fields the parser requires."""
    item = {
        "id": 7855051003,
        "title": "Associate AV Test Engineer",
        "company_name": "Motional",
        "requisition_id": "MOTI-675",
        "first_published": "2026-08-11T19:32:34-04:00",
        "updated_at": "2026-10-01T15:28:36-04:00",
        "location": {"name": "Pittsburgh, Pennsylvania, United States"},
        "absolute_url": "https://motional.com/open-positions/?gh_jid=7855051003",
    }
    item.update(overrides)
    return item


def board(items: list[dict], total: int | None = None) -> dict:
    return {"jobs": items, "meta": {"total": len(items) if total is None else total}}


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_listing_url():
    assert GreenhouseScraper("motional").listing_url == (
        "https://boards-api.greenhouse.io/v1/boards/motional/jobs"
    )


def test_source_defaults_to_the_board():
    assert GreenhouseScraper("motional").source == "greenhouse:motional"


def test_detail_url():
    assert GreenhouseScraper("motional").detail_url("42") == (
        "https://boards-api.greenhouse.io/v1/boards/motional/jobs/42"
    )


def test_company_defaults_to_the_board():
    assert GreenhouseScraper("motional").company == "motional"


def test_company_can_be_relabelled():
    assert GreenhouseScraper("motional", name="Motional").company == "Motional"


def test_empty_board_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        GreenhouseScraper("")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def test_fetch_parses_every_posting():
    payload = board([listing(), listing(id=2)])
    scraper = GreenhouseScraper("motional", fetch_json=lambda url: payload)
    jobs = scraper.fetch()
    assert [j.external_id for j in jobs] == ["7855051003", "2"]
    assert {j.source for j in jobs} == {"greenhouse:motional"}


def test_fetch_returns_empty_for_an_empty_board():
    scraper = GreenhouseScraper("motional", fetch_json=lambda url: board([]))
    assert scraper.fetch() == []


def test_response_without_jobs_list_raises_fetch_error():
    with pytest.raises(FetchError, match="no 'jobs' list"):
        GreenhouseScraper("motional", fetch_json=lambda url: {"meta": {}}).fetch()


def test_http_failure_is_normalized_to_fetch_error():
    def boom(url):
        raise RuntimeError("connection reset")

    with pytest.raises(FetchError, match="could not read"):
        GreenhouseScraper("motional", fetch_json=boom).fetch()


def test_fetch_error_is_not_rewrapped():
    def missing(url):
        raise FetchError("board not found (404)")

    with pytest.raises(FetchError, match="404"):
        GreenhouseScraper("motional", fetch_json=missing).fetch()


def test_a_short_board_is_reported_against_meta_total():
    scraper = GreenhouseScraper(
        "motional", fetch_json=lambda url: board([listing()], total=5)
    )
    result = scraper.scrape()
    assert len(result.jobs) == 1
    assert any("collected 1 of 5" in message for _, message in result.errors)


def test_a_malformed_listing_is_reported_not_fatal():
    items = [listing(), {"id": 2}, listing(id=3)]
    scraper = GreenhouseScraper("motional", fetch_json=lambda url: board(items))
    result = scraper.scrape()
    assert [j.external_id for j in result.jobs] == ["7855051003", "3"]
    assert any("job 2" in message for _, message in result.errors)


def test_duplicate_ids_are_collapsed():
    payload = board([listing(id=1), listing(id=1, title="Again")])
    jobs = GreenhouseScraper("motional", fetch_json=lambda url: payload).fetch()
    assert [j.external_id for j in jobs] == ["1"]


# ---------------------------------------------------------------------------
# parse_listing
# ---------------------------------------------------------------------------

def test_parse_listing_has_no_description_but_records_where_to_get_it():
    job = GreenhouseScraper("motional").parse_listing(listing())
    assert job.description is None
    assert job.content_hash is None
    assert job.description_url == (
        "https://boards-api.greenhouse.io/v1/boards/motional/jobs/7855051003"
    )


def test_parse_listing_accepts_an_integer_id():
    job = GreenhouseScraper("motional").parse_listing(listing(id=42))
    assert job.external_id == "42"


def test_parse_listing_requires_id_and_title():
    scraper = GreenhouseScraper("motional")
    with pytest.raises(ParseError, match="'id'"):
        scraper.parse_listing({"title": "Engineer"})
    with pytest.raises(ParseError, match="'title'"):
        scraper.parse_listing({"id": 1})


def test_parse_listing_normalizes_the_first_published_date():
    job = GreenhouseScraper("motional").parse_listing(listing())
    assert job.posted_at == "2026-08-11T23:32:34Z"


def test_parse_listing_uses_the_nested_location_name():
    job = GreenhouseScraper("motional").parse_listing(listing())
    assert job.location == "Pittsburgh, Pennsylvania, United States"


def test_parse_listing_carries_the_absolute_url():
    job = GreenhouseScraper("motional").parse_listing(listing())
    assert job.url == "https://motional.com/open-positions/?gh_jid=7855051003"


def test_parse_listing_falls_back_to_the_api_company_name():
    # No configured name, so the listing's company_name is used.
    job = GreenhouseScraper("motional").parse_listing(listing())
    assert job.company == "Motional"


def test_configured_name_wins_over_the_api_name():
    job = GreenhouseScraper("motional", name="Motional Inc").parse_listing(listing())
    assert job.company == "Motional Inc"


def test_unparseable_date_is_none():
    job = GreenhouseScraper("motional").parse_listing(listing(first_published="?"))
    assert job.posted_at is None


# ---------------------------------------------------------------------------
# parse_boards
# ---------------------------------------------------------------------------

def test_parses_a_board_alone():
    (spec,) = gh.parse_boards("motional").specs
    assert (spec.board, spec.name) == ("motional", None)


def test_parses_a_display_name():
    (spec,) = gh.parse_boards("motional Motional").specs
    assert spec.name == "Motional"


def test_multi_word_display_name_is_joined():
    (spec,) = gh.parse_boards("acme Acme Corporation Ltd").specs
    assert spec.name == "Acme Corporation Ltd"


def test_blank_and_comment_lines_are_ignored():
    assert gh.parse_boards("\n   \n# comment\n").specs == []


def test_trailing_comments_are_stripped():
    (spec,) = gh.parse_boards("motional Motional  # the AV company").specs
    assert spec.name == "Motional"


def test_duplicate_boards_collapse_to_one():
    text = "motional Motional\nmotional Motional Inc"
    (spec,) = gh.parse_boards(text).specs
    assert spec.name == "Motional"                 # first line wins


def test_order_is_preserved():
    assert [s.board for s in gh.parse_boards("c\na\nb").specs] == ["c", "a", "b"]


def test_source_label_is_built():
    (spec,) = gh.parse_boards("reddit").specs
    assert spec.source == "greenhouse:reddit"


def test_unbalanced_quotes_are_reported():
    outcome = gh.parse_boards('motional "Unclosed')
    assert outcome.specs == []
    (number, message), = outcome.errors
    assert number == 1
    assert "parse" in message.lower() or "quotation" in message.lower()


# ---------------------------------------------------------------------------
# resolve / load / build
# ---------------------------------------------------------------------------

def test_resolve_prefers_an_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setenv("GREENHOUSE_BOARDS_FILE", str(tmp_path / "env.txt"))
    assert gh.resolve_companies_file(tmp_path / "given.txt") == tmp_path / "given.txt"


def test_resolve_uses_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GREENHOUSE_BOARDS_FILE", str(tmp_path / "env.txt"))
    assert gh.resolve_companies_file() == tmp_path / "env.txt"


def test_resolve_falls_back_to_the_repo_default(monkeypatch):
    monkeypatch.delenv("GREENHOUSE_BOARDS_FILE", raising=False)
    assert gh.resolve_companies_file() == gh.DEFAULT_COMPANIES_FILE
    assert gh.DEFAULT_COMPANIES_FILE.name == "greenhouse_companies.txt"


def test_default_file_lives_at_the_repo_root():
    assert gh.DEFAULT_COMPANIES_FILE.parent.name == "jobecosystem"
    assert gh.DEFAULT_COMPANIES_FILE.parent.name != "src"


def test_load_reads_a_file(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("motional Motional\n", encoding="utf-8")
    (spec,) = gh.load_companies(path).specs
    assert spec.board == "motional"


def test_load_returns_empty_for_a_missing_file(tmp_path):
    outcome = gh.load_companies(tmp_path / "nope.txt")
    assert outcome.specs == []
    assert outcome.errors == []


def test_the_shipped_file_lists_reddit_and_motional():
    outcome = gh.load_companies()
    assert outcome.errors == []
    assert {s.board for s in outcome.specs} >= {"reddit", "motional"}


def test_build_creates_one_scraper_per_board(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("reddit Reddit\nmotional Motional\n", encoding="utf-8")
    scrapers = gh.build_scrapers(path)
    assert [s.source for s in scrapers] == [
        "greenhouse:reddit", "greenhouse:motional"
    ]
    assert scrapers[0].company == "Reddit"


def test_build_returns_empty_for_a_missing_file(tmp_path):
    assert gh.build_scrapers(tmp_path / "nope.txt") == []


def test_build_passes_fetch_json_through(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("motional\n", encoding="utf-8")
    (scraper,) = gh.build_scrapers(path, fetch_json=lambda url: board([]))
    assert scraper.fetch() == []


# ---------------------------------------------------------------------------
# integration with the runner
# ---------------------------------------------------------------------------

def test_scrapers_from_the_file_run_through_the_runner(conn, tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("motional Motional\n", encoding="utf-8")
    payload = board([listing()])

    summary = runner.run_scrapers(
        conn, gh.build_scrapers(path, fetch_json=lambda url: payload)
    )
    assert [o.source for o in summary] == ["greenhouse:motional"]
    assert summary.ok

    row = conn.execute(
        "SELECT source, title, description, description_url, content_hash"
        " FROM jobs"
    ).fetchone()
    assert row["source"] == "greenhouse:motional"
    assert row["title"] == "Associate AV Test Engineer"
    assert row["description"] is None
    assert row["content_hash"] is None
    assert row["description_url"].endswith("/jobs/7855051003")


# ---------------------------------------------------------------------------
# description parsing
# ---------------------------------------------------------------------------

def test_parse_description_unescapes_then_strips_html():
    payload = {"content": "&lt;h3&gt;Mission&lt;/h3&gt;&lt;p&gt;Drive safely&lt;/p&gt;"}
    assert gh.parse_description(payload) == "Mission\nDrive safely"


def test_parse_description_requires_content():
    with pytest.raises(gh.DescriptionError, match="'content'"):
        gh.parse_description({"content": ""})


def test_parse_description_rejects_markup_only():
    with pytest.raises(gh.DescriptionError, match="no text"):
        gh.parse_description({"content": "&lt;p&gt;&lt;/p&gt;"})
