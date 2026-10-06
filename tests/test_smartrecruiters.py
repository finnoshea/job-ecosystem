"""Tests for jobecosystem.ingest.sources.smartrecruiters.

Parser and pagination tests run against stubbed JSON, so they need no network.
The shapes come from a live response for ``RedBull`` (see the module docstring):
listings carry metadata only, never a description.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from jobecosystem.ingest import runner
from jobecosystem.ingest.base import FetchError, ParseError
from jobecosystem.ingest.sources import smartrecruiters as sr
from jobecosystem.ingest.sources.smartrecruiters import SmartRecruitersScraper


def listing(**overrides) -> dict:
    """A listing with only the fields the parser requires."""
    item = {
        "id": "744000153217222",
        "name": "Office Administrator (Part Time)",
        "refNumber": "REF32740A",
        "releasedDate": "2026-10-02T15:22:56.691Z",
        "company": {"identifier": "RedBull", "name": "Red Bull"},
        "location": {
            "city": "Springfield",
            "region": "IL",
            "country": "us",
            "remote": False,
            "hybrid": False,
            "fullLocation": "Springfield, IL, United States",
        },
        "visibility": "PUBLIC",
    }
    item.update(overrides)
    return item


def page(offset: int, items: list[dict], total: int, limit: int = 100) -> dict:
    return {"offset": offset, "limit": limit, "totalFound": total, "content": items}


def paged_stub(pages: dict[int, dict], calls: list[str] | None = None):
    """A fetch_json that answers by the ``offset`` in the URL."""

    def fetch(url: str) -> dict:
        if calls is not None:
            calls.append(url)
        offset = int(parse_qs(urlparse(url).query)["offset"][0])
        return pages.get(offset, page(offset, [], total=0))

    return fetch


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_listing_url():
    assert SmartRecruitersScraper("RedBull").listing_url == (
        "https://api.smartrecruiters.com/v1/companies/RedBull/postings"
    )


def test_source_defaults_to_the_identifier():
    assert SmartRecruitersScraper("RedBull").source == "smartrecruiters:RedBull"


def test_page_url_has_limit_and_offset():
    url = SmartRecruitersScraper("RedBull").page_url(200)
    params = parse_qs(urlparse(url).query)
    assert params["limit"] == ["100"]
    assert params["offset"] == ["200"]
    assert "country" not in params


def test_page_url_includes_country_when_set():
    url = SmartRecruitersScraper("RedBull", country="us").page_url(0)
    assert parse_qs(urlparse(url).query)["country"] == ["us"]


def test_detail_and_public_urls():
    scraper = SmartRecruitersScraper("RedBull")
    assert scraper.detail_url("42") == (
        "https://api.smartrecruiters.com/v1/companies/RedBull/postings/42"
    )
    assert scraper.public_url("42") == "https://jobs.smartrecruiters.com/RedBull/42"


def test_company_defaults_to_the_identifier():
    assert SmartRecruitersScraper("RedBull").company == "RedBull"


def test_company_can_be_relabelled():
    assert SmartRecruitersScraper("RedBull", name="Red Bull").company == "Red Bull"


def test_empty_company_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        SmartRecruitersScraper("")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def test_fetch_parses_every_posting():
    scraper = SmartRecruitersScraper("RedBull", fetch_json=paged_stub({
        0: page(0, [listing(), listing(id="744000153196890")], total=2),
    }))
    jobs = scraper.fetch()
    assert [j.external_id for j in jobs] == ["744000153217222", "744000153196890"]
    assert {j.source for j in jobs} == {"smartrecruiters:RedBull"}


def test_fetch_paginates_until_total_is_reached():
    calls: list[str] = []
    stub = paged_stub(
        {
            0: page(0, [listing(id="1"), listing(id="2")], total=3, limit=2),
            2: page(2, [listing(id="3")], total=3, limit=2),
        },
        calls,
    )
    jobs = SmartRecruitersScraper("RedBull", fetch_json=stub, page_size=2).fetch()
    assert [j.external_id for j in jobs] == ["1", "2", "3"]
    offsets = [parse_qs(urlparse(u).query)["offset"][0] for u in calls]
    assert offsets == ["0", "2"]


def test_fetch_stops_on_an_empty_page():
    jobs = SmartRecruitersScraper("RedBull", fetch_json=paged_stub({
        0: page(0, [], total=0),
    })).fetch()
    assert jobs == []


def test_fetch_dedupes_repeats_across_pages():
    # A company that ignores offset must not loop or duplicate.
    stub = paged_stub({
        0: page(0, [listing(id="1")], total=2, limit=1),
        1: page(1, [listing(id="1")], total=2, limit=1),
    })
    assert [j.external_id for j in SmartRecruitersScraper(
        "RedBull", fetch_json=stub, page_size=1
    ).fetch()] == ["1"]


def test_non_public_postings_are_skipped():
    stub = paged_stub({
        0: page(0, [listing(id="1"), listing(id="2", visibility="INTERNAL")], total=2),
    })
    jobs = SmartRecruitersScraper("RedBull", fetch_json=stub).fetch()
    assert [j.external_id for j in jobs] == ["1"]


def test_missing_visibility_is_kept():
    item = listing()
    del item["visibility"]
    stub = paged_stub({0: page(0, [item], total=1)})
    assert len(SmartRecruitersScraper("RedBull", fetch_json=stub).fetch()) == 1


def test_response_without_content_list_raises_fetch_error():
    scraper = SmartRecruitersScraper("RedBull", fetch_json=lambda url: {"totalFound": 0})
    with pytest.raises(FetchError, match="no 'content' list"):
        scraper.fetch()


def test_http_failure_is_normalized_to_fetch_error():
    def boom(url):
        raise RuntimeError("connection reset")

    with pytest.raises(FetchError, match="could not read"):
        SmartRecruitersScraper("RedBull", fetch_json=boom).fetch()


def test_a_later_page_failure_is_reported():
    def flaky(url: str) -> dict:
        if "offset=0" in url:
            return page(0, [listing(id="1")], total=50, limit=1)
        raise FetchError("HTTP 503")

    scraper = SmartRecruitersScraper("RedBull", fetch_json=flaky, page_size=1)
    result = scraper.scrape()
    assert len(result.jobs) == 1
    assert any("pagination stopped" in message for _, message in result.errors)


def test_first_page_failure_propagates():
    def boom(url):
        raise FetchError("HTTP 500")

    with pytest.raises(FetchError, match="HTTP 500"):
        SmartRecruitersScraper("RedBull", fetch_json=boom).fetch()


def test_a_malformed_listing_is_reported_not_fatal():
    items = [listing(id="1"), {"id": "2"}, listing(id="3")]
    stub = paged_stub({0: page(0, items, total=3)})
    result = SmartRecruitersScraper("RedBull", fetch_json=stub).scrape()
    assert [j.external_id for j in result.jobs] == ["1", "3"]
    assert any("offset 0 item 1" in message for _, message in result.errors)


# ---------------------------------------------------------------------------
# parse_listing
# ---------------------------------------------------------------------------

def test_parse_listing_has_no_description_but_records_where_to_get_it():
    job = SmartRecruitersScraper("RedBull").parse_listing(listing())
    assert job.description is None
    assert job.content_hash is None            # NULL until the fetch step runs
    assert job.description_url == (
        "https://api.smartrecruiters.com/v1/companies/RedBull/postings/744000153217222"
    )


def test_parse_listing_requires_id_and_name():
    scraper = SmartRecruitersScraper("RedBull")
    with pytest.raises(ParseError, match="'id'"):
        scraper.parse_listing({"name": "Engineer"})
    with pytest.raises(ParseError, match="'name'"):
        scraper.parse_listing({"id": "1"})


def test_parse_listing_normalizes_the_released_date():
    job = SmartRecruitersScraper("RedBull").parse_listing(listing())
    assert job.posted_at == "2026-10-02T15:22:56Z"


def test_parse_listing_builds_the_public_url():
    job = SmartRecruitersScraper("RedBull").parse_listing(listing())
    assert job.url == "https://jobs.smartrecruiters.com/RedBull/744000153217222"


def test_parse_listing_uses_the_full_location():
    job = SmartRecruitersScraper("RedBull").parse_listing(listing())
    assert job.location == "Springfield, IL, United States"


def test_parse_listing_composes_a_location_when_full_is_absent():
    item = listing(location={"city": "Austin", "region": "TX", "country": "us",
                             "remote": True})
    job = SmartRecruitersScraper("RedBull").parse_listing(item)
    assert job.location == "Austin, TX, us / Remote"


def test_parse_listing_falls_back_to_the_api_company_name():
    job = SmartRecruitersScraper("RedBull").parse_listing(listing())
    assert job.company == "Red Bull"


def test_configured_name_wins_over_the_api_name():
    job = SmartRecruitersScraper("RedBull", name="Red Bull GmbH").parse_listing(listing())
    assert job.company == "Red Bull GmbH"


def test_unparseable_date_is_none():
    job = SmartRecruitersScraper("RedBull").parse_listing(
        listing(releasedDate="not a date")
    )
    assert job.posted_at is None


# ---------------------------------------------------------------------------
# parse_companies
# ---------------------------------------------------------------------------

def test_parses_an_identifier_only():
    (spec,) = sr.parse_companies("RedBull").specs
    assert (spec.identifier, spec.country, spec.name) == ("RedBull", None, None)


def test_parses_a_country():
    (spec,) = sr.parse_companies("RedBull us").specs
    assert (spec.country, spec.name) == ("us", None)


def test_country_is_lowercased():
    (spec,) = sr.parse_companies("RedBull US").specs
    assert spec.country == "us"


def test_star_means_every_country():
    (spec,) = sr.parse_companies("RedBull * Red Bull").specs
    assert spec.country is None
    assert spec.name == "Red Bull"


def test_parses_a_display_name():
    (spec,) = sr.parse_companies("RedBull Red Bull").specs
    assert (spec.country, spec.name) == (None, "Red Bull")


def test_parses_country_and_multi_word_name():
    (spec,) = sr.parse_companies("acme us Acme Corporation Ltd").specs
    assert (spec.identifier, spec.country, spec.name) == (
        "acme", "us", "Acme Corporation Ltd"
    )


def test_blank_and_comment_lines_are_ignored():
    assert sr.parse_companies("\n   \n# comment\n").specs == []


def test_trailing_comments_are_stripped():
    (spec,) = sr.parse_companies("RedBull us Red Bull  # the energy drink").specs
    assert spec.name == "Red Bull"


def test_duplicate_identifiers_collapse_to_one():
    text = "RedBull us Red Bull\nRedBull * Red Bull GmbH"
    (spec,) = sr.parse_companies(text).specs
    assert spec.country == "us"                    # first line wins


def test_order_is_preserved():
    text = "c\na\nb"
    assert [s.identifier for s in sr.parse_companies(text).specs] == ["c", "a", "b"]


def test_source_label_is_built():
    (spec,) = sr.parse_companies("RedBull us").specs
    assert spec.source == "smartrecruiters:RedBull"


# ---------------------------------------------------------------------------
# resolve / load / build
# ---------------------------------------------------------------------------

def test_resolve_prefers_an_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTRECRUITERS_COMPANIES_FILE", str(tmp_path / "env.txt"))
    assert sr.resolve_companies_file(tmp_path / "given.txt") == tmp_path / "given.txt"


def test_resolve_uses_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTRECRUITERS_COMPANIES_FILE", str(tmp_path / "env.txt"))
    assert sr.resolve_companies_file() == tmp_path / "env.txt"


def test_resolve_falls_back_to_the_repo_default(monkeypatch):
    monkeypatch.delenv("SMARTRECRUITERS_COMPANIES_FILE", raising=False)
    assert sr.resolve_companies_file() == sr.DEFAULT_COMPANIES_FILE
    assert sr.DEFAULT_COMPANIES_FILE.name == "smartrecruiters_companies.txt"


def test_default_file_lives_at_the_repo_root():
    assert sr.DEFAULT_COMPANIES_FILE.parent.name == "jobecosystem"
    assert sr.DEFAULT_COMPANIES_FILE.parent.name != "src"


def test_load_reads_a_file(tmp_path):
    path = tmp_path / "companies.txt"
    path.write_text("RedBull us Red Bull\n", encoding="utf-8")
    (spec,) = sr.load_companies(path).specs
    assert spec.identifier == "RedBull"


def test_load_returns_empty_for_a_missing_file(tmp_path):
    outcome = sr.load_companies(tmp_path / "nope.txt")
    assert outcome.specs == []
    assert outcome.errors == []


def test_the_shipped_file_loads_cleanly():
    outcome = sr.load_companies()
    assert outcome.errors == []
    assert "RedBull" in {s.identifier for s in outcome.specs}


def test_build_creates_one_scraper_per_company(tmp_path):
    path = tmp_path / "companies.txt"
    path.write_text("RedBull us Red Bull\nacme Acme\n", encoding="utf-8")
    scrapers = sr.build_scrapers(path)
    assert [s.source for s in scrapers] == [
        "smartrecruiters:RedBull", "smartrecruiters:acme"
    ]
    assert scrapers[0].country == "us"
    assert scrapers[0].company == "Red Bull"


def test_build_returns_empty_for_a_missing_file(tmp_path):
    assert sr.build_scrapers(tmp_path / "nope.txt") == []


def test_build_passes_fetch_json_through(tmp_path):
    path = tmp_path / "companies.txt"
    path.write_text("RedBull\n", encoding="utf-8")
    stub = paged_stub({0: page(0, [], total=0)})
    (scraper,) = sr.build_scrapers(path, fetch_json=stub)
    assert scraper.fetch() == []


# ---------------------------------------------------------------------------
# integration with the runner
# ---------------------------------------------------------------------------

def test_scrapers_from_the_file_run_through_the_runner(conn, tmp_path):
    path = tmp_path / "companies.txt"
    path.write_text("RedBull us Red Bull\n", encoding="utf-8")
    stub = paged_stub({0: page(0, [listing()], total=1)})

    summary = runner.run_scrapers(conn, sr.build_scrapers(path, fetch_json=stub))
    assert [o.source for o in summary] == ["smartrecruiters:RedBull"]
    assert summary.ok

    row = conn.execute(
        "SELECT source, title, description, description_url, content_hash"
        " FROM jobs"
    ).fetchone()
    assert row["source"] == "smartrecruiters:RedBull"
    assert row["title"] == "Office Administrator (Part Time)"
    assert row["description"] is None
    assert row["content_hash"] is None
    assert row["description_url"].endswith("/postings/744000153217222")


# ---------------------------------------------------------------------------
# description parsing
# ---------------------------------------------------------------------------

def test_parse_description_joins_the_sections():
    payload = {"jobAd": {"sections": {
        "companyDescription": {"title": "About", "text": "<p>Acme</p>"},
        "jobDescription": {"title": "Role", "text": "<p>Build things</p>"},
        "qualifications": {"title": "You", "text": "<ul><li>Python</li></ul>"},
    }}}
    text = sr.parse_description(payload)
    assert "Build things" in text
    assert "Python" in text
    # Section titles are kept as headings for keyword matching.
    assert "About" in text
    assert "Role" in text


def test_parse_description_ignores_blank_sections():
    payload = {"jobAd": {"sections": {
        "jobDescription": {"title": "Role", "text": "<p>Body</p>"},
        "additionalInformation": {"title": "More", "text": "   "},
    }}}
    assert sr.parse_description(payload) == "Role\nBody"


def test_parse_description_requires_sections():
    with pytest.raises(sr.DescriptionError, match="jobAd.sections"):
        sr.parse_description({"jobAd": {}})


def test_parse_description_rejects_an_empty_detail():
    payload = {"jobAd": {"sections": {"jobDescription": {"text": "  "}}}}
    with pytest.raises(sr.DescriptionError, match="no text"):
        sr.parse_description(payload)
