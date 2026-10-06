"""Tests for jobecosystem.ingest.sources.lever.

Parser tests run against stubbed JSON shaped like a live response from
``spotify``: a Lever listing already carries the full description, so nothing
here expects a second request.
"""

from __future__ import annotations

import pytest

from jobecosystem.ingest.base import FetchError, ParseError
from jobecosystem.ingest.sources import lever
from jobecosystem.ingest.sources.lever import LeverScraper, lever_description


def posting(**overrides) -> dict:
    """A posting with the fields the parser reads, plus a list section."""
    item = {
        "id": "abc-123",
        "text": "Senior Software Engineer",
        "categories": {
            "location": "Remote",
            "team": "Engineering",
            "allLocations": ["Remote"],
            "commitment": "Full-time",
            "department": "Engineering",
        },
        "createdAt": 1700000000000,
        "hostedUrl": "https://jobs.lever.co/spotify/abc-123",
        "applyUrl": "https://jobs.lever.co/spotify/abc-123/apply",
        "descriptionPlain": "We are hiring.",
        "description": "<p>We are hiring.</p>",
        "lists": [
            {"text": "Responsibilities", "content": "<ul><li>Build things</li></ul>"}
        ],
        "additionalPlain": "Apply now.",
    }
    item.update(overrides)
    return item


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_listing_url():
    assert LeverScraper("spotify").listing_url == (
        "https://api.lever.co/v0/postings/spotify?mode=json"
    )


def test_source_defaults_to_the_site():
    assert LeverScraper("spotify").source == "lever:spotify"


def test_company_defaults_to_the_site():
    assert LeverScraper("spotify").company == "spotify"


def test_company_can_be_relabelled():
    assert LeverScraper("spotify", name="Spotify").company == "Spotify"


def test_empty_site_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        LeverScraper("")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_listing_maps_the_fields():
    job = LeverScraper("spotify", name="Spotify").parse_listing(posting())
    assert job.source == "lever:spotify"
    assert job.external_id == "abc-123"
    assert job.title == "Senior Software Engineer"
    assert job.company == "Spotify"
    assert job.location == "Remote"
    assert job.url == "https://jobs.lever.co/spotify/abc-123"
    # The description is inline, so there is nothing left to fetch.
    assert job.description_url is None
    assert job.content_hash is not None          # set because a description exists


def test_created_at_epoch_milliseconds_becomes_iso():
    job = LeverScraper("spotify").parse_listing(posting())
    assert job.posted_at == "2023-11-14T22:13:20Z"


def test_location_falls_back_to_all_locations():
    job = LeverScraper("spotify").parse_listing(
        posting(categories={"allLocations": ["Austin", "Remote"]})
    )
    assert job.location == "Austin, Remote"


def test_description_combines_intro_lists_and_additional():
    text = lever_description(posting())
    assert "We are hiring." in text
    assert "Responsibilities" in text
    assert "Build things" in text
    assert "Apply now." in text


def test_missing_id_is_a_parse_error():
    with pytest.raises(ParseError, match="id"):
        LeverScraper("spotify").parse_listing(posting(id=None))


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def test_fetch_parses_every_posting():
    payload = [posting(), posting(id="def-456")]
    scraper = LeverScraper("spotify", fetch_json=lambda url: payload)
    jobs = scraper.fetch()
    assert [j.external_id for j in jobs] == ["abc-123", "def-456"]
    assert {j.source for j in jobs} == {"lever:spotify"}


def test_a_malformed_posting_is_reported_not_fatal():
    payload = [posting(), {"id": "no-title"}, posting(id="ok")]
    result = LeverScraper("spotify", fetch_json=lambda url: payload).scrape()
    assert [j.external_id for j in result.jobs] == ["abc-123", "ok"]
    assert any("no-title" in message for _, message in result.errors)


def test_a_dead_board_is_skipped_not_fatal():
    def gone(url):
        raise lever.BoardGone("GET https://api.lever.co/v0/postings/dead: 404")

    result = LeverScraper("dead", fetch_json=gone).scrape()
    assert result.jobs == []
    assert any("404" in message for _, message in result.errors)


def test_a_non_404_fetch_error_propagates():
    def missing(url):
        raise FetchError("HTTP 503")

    with pytest.raises(FetchError, match="503"):
        LeverScraper("spotify", fetch_json=missing).fetch()


def test_an_unnormalized_exception_becomes_a_fetch_error():
    def boom(url):
        raise RuntimeError("connection reset")

    with pytest.raises(FetchError, match="could not read"):
        LeverScraper("spotify", fetch_json=boom).fetch()


def test_a_non_list_response_is_a_fetch_error():
    with pytest.raises(FetchError, match="expected a list"):
        LeverScraper("spotify", fetch_json=lambda url: {"jobs": []}).fetch()


# ---------------------------------------------------------------------------
# company list
# ---------------------------------------------------------------------------

def test_parse_companies_reads_slug_and_name():
    outcome = lever.parse_companies("acme Acme Corp\n# comment\n\nbeta\n")
    assert [(spec.slug, spec.name) for spec in outcome.specs] == [
        ("acme", "Acme Corp"), ("beta", None)
    ]


def test_a_missing_companies_file_yields_nothing(tmp_path):
    outcome = lever.load_companies(tmp_path / "nope.txt")
    assert len(outcome) == 0
    assert outcome.errors == []


def test_build_scrapers_from_a_file(tmp_path):
    path = tmp_path / "lever_companies.txt"
    path.write_text("acme Acme Corp\nbeta\n", encoding="utf-8")
    scrapers = lever.build_scrapers(path)
    assert [s.source for s in scrapers] == ["lever:acme", "lever:beta"]
    assert scrapers[0].company == "Acme Corp"
