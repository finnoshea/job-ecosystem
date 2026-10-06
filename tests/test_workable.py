"""Tests for jobecosystem.ingest.sources.workable.

The scraper reads Workable's cross-company marketplace feed. Parser tests run
against stubbed JSON shaped like a live page; pagination is driven by an
injected fetcher that returns successive pages.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from jobecosystem.ingest.base import FetchError, ParseError
from jobecosystem.ingest.sources import workable
from jobecosystem.ingest.sources.workable import WorkableScraper


def record(**overrides) -> dict:
    """A marketplace record with the fields the parser reads."""
    item = {
        "id": "job-1",
        "title": "Survey Technician",
        "company": {"id": "c1", "title": "Blew & Associates"},
        "url": "https://jobs.workable.com/view/xyz/survey",
        "location": {
            "city": "Milwaukee",
            "subregion": "Wisconsin",
            "countryName": "United States",
        },
        "locations": ["TELECOMMUTE", "Milwaukee, Wisconsin, United States"],
        "workplace": "remote",
        "employmentType": "Full-time",
        "created": "2026-10-05T16:06:55.362Z",
        "description": "<p>Blew is hiring.</p>",
        "requirementsSection": "<ul><li>Surveying</li></ul>",
        "benefitsSection": "<p>Great benefits.</p>",
    }
    item.update(overrides)
    return item


def paged(payloads: list[dict]):
    """A fetcher that returns each payload in turn and records the URLs."""
    calls: list[str] = []

    def fetch(url):
        calls.append(url)
        return payloads[len(calls) - 1]

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_source_is_the_aggregate_feed():
    assert WorkableScraper().source == "workable"


def test_page_url_without_filters():
    assert WorkableScraper().page_url() == workable.LISTING_URL


def test_page_url_carries_filters_and_token():
    scraper = WorkableScraper(query="engineer", location="Remote", workplace="remote")
    query = parse_qs(urlparse(scraper.page_url("tok")).query)
    assert query["query"] == ["engineer"]
    assert query["location"] == ["Remote"]
    assert query["workplace"] == ["remote"]
    assert query["pageToken"] == ["tok"]


def test_zero_pages_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        WorkableScraper(pages=0)


def test_default_page_budget_is_the_editable_constant():
    assert WorkableScraper().pages == workable.DEFAULT_PAGES


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_listing_maps_the_fields():
    job = WorkableScraper().parse_listing(record())
    assert job.source == "workable"
    assert job.external_id == "job-1"
    assert job.title == "Survey Technician"
    assert job.company == "Blew & Associates"
    assert job.url == "https://jobs.workable.com/view/xyz/survey"
    assert job.posted_at == "2026-10-05T16:06:55Z"
    assert job.description_url is None
    assert job.content_hash is not None


def test_telecommute_becomes_remote():
    job = WorkableScraper().parse_listing(record())
    assert job.location == "Remote, Milwaukee, Wisconsin, United States"


def test_location_falls_back_to_the_object():
    job = WorkableScraper().parse_listing(
        record(locations=None,
               location={"city": "Berlin", "countryName": "Germany"})
    )
    assert job.location == "Berlin, Germany"


def test_description_covers_every_section():
    text = WorkableScraper().parse_listing(record()).description
    assert "Blew is hiring." in text
    assert "Surveying" in text
    assert "Great benefits." in text


def test_a_missing_title_is_a_parse_error():
    with pytest.raises(ParseError, match="title"):
        WorkableScraper().parse_listing(record(title=""))


def test_a_missing_company_is_labelled_unknown():
    job = WorkableScraper().parse_listing(record(company=None))
    assert job.company == "Unknown"


# ---------------------------------------------------------------------------
# fetch and pagination
# ---------------------------------------------------------------------------

def test_fetch_follows_the_page_token():
    fetch = paged([
        {"jobs": [record(id="a")], "nextPageToken": "t1"},
        {"jobs": [record(id="b")]},
    ])
    jobs = WorkableScraper(pages=5, fetch_json=fetch).fetch()
    assert [j.external_id for j in jobs] == ["a", "b"]
    assert len(fetch.calls) == 2
    assert "pageToken=t1" in fetch.calls[1]


def test_fetch_stops_at_the_page_cap():
    fetch = paged([
        {"jobs": [record(id="a")], "nextPageToken": "t1"},
        {"jobs": [record(id="b")]},
    ])
    jobs = WorkableScraper(pages=1, fetch_json=fetch).fetch()
    assert [j.external_id for j in jobs] == ["a"]
    assert len(fetch.calls) == 1


def test_fetch_stops_when_there_is_no_token():
    fetch = paged([{"jobs": [record(id="a")]}])
    jobs = WorkableScraper(pages=5, fetch_json=fetch).fetch()
    assert [j.external_id for j in jobs] == ["a"]
    assert len(fetch.calls) == 1


def test_a_later_page_failure_keeps_what_was_collected():
    calls: list[str] = []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            return {"jobs": [record(id="a")], "nextPageToken": "t1"}
        raise FetchError("boom")

    result = WorkableScraper(pages=5, fetch_json=fetch).scrape()
    assert [j.external_id for j in result.jobs] == ["a"]
    assert any("pagination stopped" in message for _, message in result.errors)


def test_a_first_page_failure_is_fatal():
    def boom(url):
        raise FetchError("refused")

    with pytest.raises(FetchError, match="refused"):
        WorkableScraper(fetch_json=boom).fetch()


def test_a_response_without_a_jobs_list_is_a_fetch_error():
    with pytest.raises(FetchError, match="no 'jobs' list"):
        WorkableScraper(fetch_json=lambda url: {"total": 1}).fetch()


def test_a_malformed_record_is_reported_not_fatal():
    fetch = paged([{"jobs": [record(), {"id": "no-title"}, record(id="ok")]}])
    result = WorkableScraper(fetch_json=fetch).scrape()
    assert [j.external_id for j in result.jobs] == ["job-1", "ok"]
    assert any("no-title" in message for _, message in result.errors)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

def test_build_scrapers_returns_one_configured_scraper():
    (scraper,) = workable.build_scrapers(pages=3, query="engineer")
    assert scraper.source == "workable"
    assert scraper.pages == 3
    assert scraper.query == "engineer"
