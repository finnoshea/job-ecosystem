"""Tests for jobecosystem.ingest.sources.ashby.

Parser tests run against a saved board payload (``tests/fixtures/ashby_board.json``),
so they need no network. A couple of tests hit the live board and are marked
``network``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobecosystem.ingest import runner
from jobecosystem.ingest.base import FetchError, ParseError
from jobecosystem.ingest.sources.ashby import BOARD_URL, AshbyScraper

FIXTURE = Path(__file__).parent / "fixtures" / "ashby_board.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def scraper(payload) -> AshbyScraper:
    return AshbyScraper(
        "ramp", company="Ramp", fetch_json=lambda url: payload
    )


def minimal_listing(**overrides) -> dict:
    """A listing with only the fields the parser requires."""
    listing = {"id": "abc-123", "title": "Software Engineer"}
    listing.update(overrides)
    return listing


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_board_url():
    assert AshbyScraper("ramp").board_url == BOARD_URL.format(board="ramp")
    assert AshbyScraper("ramp").board_url == \
        "https://api.ashbyhq.com/posting-api/job-board/ramp"


def test_source_defaults_to_board_name():
    assert AshbyScraper("ramp").source == "ashby:ramp"


def test_source_can_be_overridden_for_multi_board_companies():
    scraper = AshbyScraper("ramp-eng", source="ashby:ramp")
    assert scraper.source == "ashby:ramp"
    assert scraper.board_url.endswith("/ramp-eng")


def test_company_defaults_to_board_slug():
    assert AshbyScraper("ramp").company == "ramp"


def test_company_can_be_given_a_display_name():
    assert AshbyScraper("ramp", company="Ramp Financial").company == "Ramp Financial"


def test_empty_board_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        AshbyScraper("")


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def test_fetch_parses_every_listed_job(scraper, payload):
    jobs = scraper.fetch()
    listed = [j for j in payload["jobs"] if j.get("isListed") is not False]
    assert len(jobs) == len(listed)
    assert len(jobs) == 4


def test_fetch_labels_source_and_company(scraper):
    jobs = scraper.fetch()
    assert {j.source for j in jobs} == {"ashby:ramp"}
    assert {j.company for j in jobs} == {"Ramp"}


def test_unlisted_jobs_are_skipped(payload):
    modified = dict(payload)
    modified["jobs"] = [dict(payload["jobs"][0], isListed=False)]
    scraper = AshbyScraper("ramp", fetch_json=lambda url: modified)
    assert scraper.fetch() == []


def test_fetch_returns_empty_list_for_an_empty_board():
    scraper = AshbyScraper("ramp", fetch_json=lambda url: {"apiVersion": "1", "jobs": []})
    assert scraper.fetch() == []


def test_response_without_jobs_list_raises_fetch_error():
    scraper = AshbyScraper("ramp", fetch_json=lambda url: {"apiVersion": "1"})
    with pytest.raises(FetchError, match="no 'jobs' list"):
        scraper.fetch()


def test_http_failure_is_normalized_to_fetch_error():
    def boom(url):
        raise RuntimeError("connection reset")

    with pytest.raises(FetchError, match="could not read"):
        AshbyScraper("ramp", fetch_json=boom).fetch()


def test_fetch_error_is_not_rewrapped():
    def missing(url):
        raise FetchError("board not found (404)")

    with pytest.raises(FetchError, match="board not found"):
        AshbyScraper("ramp", fetch_json=missing).fetch()


def test_one_bad_listing_does_not_discard_the_rest(payload):
    modified = dict(payload)
    modified["jobs"] = [{"no_id": True}, payload["jobs"][0], "not-a-dict"]
    scraper = AshbyScraper("ramp", fetch_json=lambda url: modified)
    jobs = scraper.fetch()
    assert len(jobs) == 1
    assert jobs[0].external_id == payload["jobs"][0]["id"]


def test_bad_listings_are_reported_not_dropped(payload):
    # The gap this closes: failures used to vanish, so a board that suddenly
    # changed shape looked like a board with fewer openings.
    modified = dict(payload)
    modified["jobs"] = [payload["jobs"][0], {"id": "bad-1"}, "junk"]
    scraper = AshbyScraper("ramp", fetch_json=lambda url: modified)
    result = scraper.scrape()
    assert result.ok is False
    assert len(result.jobs) == 1
    assert len(result.errors) == 2


def test_parse_error_message_names_the_offending_job(payload):
    modified = dict(payload)
    modified["jobs"] = [{"id": "bad-1"}]  # no title
    scraper = AshbyScraper("ramp", fetch_json=lambda url: modified)
    (_tenant, message), = scraper.scrape().errors
    assert "bad-1" in message
    assert "missing 'title'" in message


def test_non_dict_listing_is_reported_by_position(payload):
    modified = dict(payload)
    modified["jobs"] = ["junk", payload["jobs"][0]]
    scraper = AshbyScraper("ramp", fetch_json=lambda url: modified)
    (_tenant, message), = scraper.scrape().errors
    assert "listing 0" in message
    assert "expected an object" in message


def test_unlisted_jobs_are_not_reported_as_errors(payload):
    # Skipping an unlisted posting is intended, not a failure.
    modified = dict(payload)
    modified["jobs"] = [dict(payload["jobs"][0], isListed=False), payload["jobs"][0]]
    scraper = AshbyScraper("ramp", fetch_json=lambda url: modified)
    result = scraper.scrape()
    assert result.ok
    assert len(result.jobs) == 1


def test_errors_do_not_accumulate_across_runs(payload):
    modified = dict(payload)
    modified["jobs"] = [{"id": "bad-1"}]
    scraper = AshbyScraper("ramp", fetch_json=lambda url: modified)
    for _ in range(3):
        assert len(scraper.scrape().errors) == 1


def test_a_clean_board_reports_no_errors(scraper):
    assert scraper.scrape().errors == []


def test_fetch_errors_still_raise_rather_than_being_reported():
    # A dead board is fatal for the source; a bad listing is not.
    scraper = AshbyScraper("ramp", fetch_json=lambda url: {"apiVersion": "1"})
    with pytest.raises(FetchError):
        scraper.scrape()


# ---------------------------------------------------------------------------
# parse_listing: required fields
# ---------------------------------------------------------------------------

def test_parse_listing_uses_ashby_id_as_external_id(scraper):
    job = scraper.parse_listing(minimal_listing(id="xyz-789"))
    assert job.external_id == "xyz-789"


def test_missing_id_raises_parse_error(scraper):
    with pytest.raises(ParseError, match="missing 'id'"):
        scraper.parse_listing({"title": "Engineer"})


def test_missing_title_raises_parse_error(scraper):
    with pytest.raises(ParseError, match="missing 'title'"):
        scraper.parse_listing({"id": "abc"})


def test_blank_id_raises_parse_error(scraper):
    with pytest.raises(ParseError, match="missing 'id'"):
        scraper.parse_listing({"id": "   ", "title": "Engineer"})


def test_non_string_id_raises_parse_error(scraper):
    with pytest.raises(ParseError, match="missing 'id'"):
        scraper.parse_listing({"id": 123, "title": "Engineer"})


def test_title_is_stripped(scraper):
    # Ashby titles sometimes carry a leading space.
    job = scraper.parse_listing(minimal_listing(title="  Security Engineer  "))
    assert job.title == "Security Engineer"


def test_real_payload_title_is_stripped(scraper, payload):
    job = scraper.parse_listing(payload["jobs"][0])
    assert job.title == job.title.strip()
    assert not job.title.startswith(" ")


# ---------------------------------------------------------------------------
# parse_listing: optional fields
# ---------------------------------------------------------------------------

def test_url_prefers_job_url_over_apply_url(scraper):
    job = scraper.parse_listing(
        minimal_listing(jobUrl="https://x.test/job", applyUrl="https://x.test/apply")
    )
    assert job.url == "https://x.test/job"


def test_url_falls_back_to_apply_url(scraper):
    job = scraper.parse_listing(minimal_listing(applyUrl="https://x.test/apply"))
    assert job.url == "https://x.test/apply"


def test_url_is_none_when_absent(scraper):
    assert scraper.parse_listing(minimal_listing()).url is None


def test_description_prefers_plain_over_html(scraper):
    job = scraper.parse_listing(
        minimal_listing(descriptionPlain="Plain text", descriptionHtml="<p>HTML</p>")
    )
    assert job.description == "Plain text"


def test_description_is_none_without_plain_field(scraper):
    # HTML alone is not used: stripping it is not worth the risk of mangling.
    job = scraper.parse_listing(minimal_listing(descriptionHtml="<p>HTML</p>"))
    assert job.description is None


def test_description_is_stripped(scraper):
    job = scraper.parse_listing(minimal_listing(descriptionPlain="  Body  "))
    assert job.description == "Body"


def test_salary_fields_are_left_unset(scraper):
    # Ashby returns no structured compensation; see the module docstring.
    job = scraper.parse_listing(minimal_listing(descriptionPlain="Pay: $100,000"))
    assert job.salary_min is None
    assert job.salary_max is None


def test_raw_json_round_trips_the_listing(scraper):
    listing = minimal_listing(department="Engineering")
    job = scraper.parse_listing(listing)
    assert json.loads(job.raw_json) == listing


def test_bookkeeping_fields_are_left_to_upsert(scraper):
    job = scraper.parse_listing(minimal_listing())
    assert job.first_seen_at is None
    assert job.last_seen_at is None
    assert job.repost_count == 0
    assert job.status == "new"


def test_content_hash_is_filled_in_when_a_description_is_present(scraper):
    listing = minimal_listing(descriptionPlain="Body text")
    assert len(scraper.parse_listing(listing).content_hash) == 64


def test_content_hash_is_none_without_a_description(scraper):
    # The real board always sends a description, but a listing-only response
    # must not produce a hash that would collide on titles.
    assert scraper.parse_listing(minimal_listing()).content_hash is None


# ---------------------------------------------------------------------------
# parse_listing: location
# ---------------------------------------------------------------------------

def test_location_is_the_primary_when_only_that_is_given(scraper):
    job = scraper.parse_listing(minimal_listing(location="Austin, TX"))
    assert job.location == "Austin, TX"


def test_location_is_none_when_nothing_is_given(scraper):
    assert scraper.parse_listing(minimal_listing()).location is None


def test_remote_flag_is_annotated(scraper):
    job = scraper.parse_listing(minimal_listing(location="NYC", isRemote=True))
    assert job.location == "NYC / Remote"


def test_remote_is_not_duplicated_when_already_in_the_name(scraper):
    job = scraper.parse_listing(minimal_listing(location="Remote (US)", isRemote=True))
    assert job.location == "Remote (US)"


def test_workplace_type_is_included_when_not_onsite(scraper):
    job = scraper.parse_listing(minimal_listing(location="NYC", workplaceType="Hybrid"))
    assert job.location == "NYC / Hybrid"


def test_onsite_workplace_type_is_omitted_as_noise(scraper):
    job = scraper.parse_listing(minimal_listing(location="NYC", workplaceType="OnSite"))
    assert job.location == "NYC"


def test_secondary_locations_are_appended(scraper):
    job = scraper.parse_listing(minimal_listing(
        location="NYC",
        secondaryLocations=[{"location": "Remote (Canada)"}, {"location": "Remote (US)"}],
    ))
    assert job.location == "NYC / Remote (Canada) / Remote (US)"


def test_secondary_locations_are_not_duplicated(scraper):
    job = scraper.parse_listing(minimal_listing(
        location="NYC", secondaryLocations=[{"location": "NYC"}]
    ))
    assert job.location == "NYC"


def test_malformed_secondary_locations_are_ignored(scraper):
    job = scraper.parse_listing(minimal_listing(
        location="NYC",
        secondaryLocations=["junk", {"no_location": 1}, {"location": "  "},
                            {"location": "Boston"}],
    ))
    assert job.location == "NYC / Boston"


def test_real_payload_location_combines_sources(scraper, payload):
    job = scraper.parse_listing(payload["jobs"][0])
    assert "New York, NY (HQ)" in job.location
    assert "Remote" in job.location


# ---------------------------------------------------------------------------
# parse_listing: timestamps
# ---------------------------------------------------------------------------

def test_published_at_is_normalized_to_our_format(scraper):
    job = scraper.parse_listing(minimal_listing(publishedAt="2026-04-07T17:12:35.753+00:00"))
    assert job.posted_at == "2026-04-07T17:12:35Z"


def test_published_at_accepts_z_suffix(scraper):
    job = scraper.parse_listing(minimal_listing(publishedAt="2026-04-07T17:12:35Z"))
    assert job.posted_at == "2026-04-07T17:12:35Z"


def test_published_at_converts_a_non_utc_offset(scraper):
    job = scraper.parse_listing(minimal_listing(publishedAt="2026-04-07T17:12:35+05:00"))
    assert job.posted_at == "2026-04-07T12:12:35Z"


def test_naive_timestamp_is_treated_as_utc(scraper):
    job = scraper.parse_listing(minimal_listing(publishedAt="2026-04-07T17:12:35"))
    assert job.posted_at == "2026-04-07T17:12:35Z"


def test_missing_published_at_is_none(scraper):
    assert scraper.parse_listing(minimal_listing()).posted_at is None


def test_unparseable_published_at_is_none_not_an_error(scraper):
    job = scraper.parse_listing(minimal_listing(publishedAt="last Tuesday"))
    assert job.posted_at is None


def test_all_real_timestamps_normalize(scraper, payload):
    import re

    pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
    for listing in payload["jobs"]:
        job = scraper.parse_listing(listing)
        if job.posted_at is not None:
            assert pattern.fullmatch(job.posted_at), job.posted_at


# ---------------------------------------------------------------------------
# parse_payload
# ---------------------------------------------------------------------------

def test_parse_payload_matches_fetch(scraper, payload):
    assert [j.external_id for j in scraper.parse_payload(payload)] == \
           [j.external_id for j in scraper.fetch()]


def test_parse_payload_rejects_a_bad_envelope(scraper):
    with pytest.raises(FetchError):
        scraper.parse_payload({"nope": []})


# ---------------------------------------------------------------------------
# end-to-end through the runner
# ---------------------------------------------------------------------------

def test_scraper_runs_through_the_runner(conn, scraper):
    summary = runner.run_scrapers(conn, [scraper])
    outcome = summary.outcomes[0]
    assert outcome.ok
    assert outcome.fetched == 4
    assert outcome.inserted == 4
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 1


def test_rerunning_the_same_board_inserts_nothing_new(conn, scraper):
    runner.run_scrapers(conn, [scraper])
    summary = runner.run_scrapers(conn, [scraper])
    assert summary.outcomes[0].inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 4


def test_scraped_jobs_land_in_the_views(conn, scraper):
    runner.run_scrapers(conn, [scraper])
    assert conn.execute("SELECT COUNT(*) FROM jobs_unseen").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM jobs_today").fetchone()[0] == 4


def test_scraped_jobs_carry_usable_content(conn, scraper):
    runner.run_scrapers(conn, [scraper])
    rows = conn.execute(
        "SELECT title, description, url FROM jobs WHERE description IS NOT NULL"
    ).fetchall()
    assert rows
    for row in rows:
        assert row["title"]
        assert row["url"].startswith("https://")


def test_per_listing_errors_reach_the_database(conn, payload):
    # End-to-end: reported errors become rows in scrape_errors and a non-zero
    # error count on the run, while the good listings still land.
    modified = dict(payload)
    modified["jobs"] = [payload["jobs"][0], {"id": "bad-1"}, "junk"]
    scraper = AshbyScraper("ramp", company="Ramp", fetch_json=lambda url: modified)

    summary = runner.run_scrapers(conn, [scraper])
    outcome = summary.outcomes[0]
    assert outcome.ok is True          # the board succeeded; listings did not
    assert outcome.fetched == 1
    assert len(outcome.errors) == 2

    row = conn.execute("SELECT fetched, errors FROM scrape_runs").fetchone()
    assert (row["fetched"], row["errors"]) == (1, 2)
    stored = [r[0] for r in conn.execute("SELECT error FROM scrape_errors ORDER BY id")]
    assert any("bad-1" in message for message in stored)
    assert any("expected an object" in message for message in stored)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# live board (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.slow
def test_live_board_is_reachable(conn):
    """Hits the real Ashby API; run with ``-m network``."""
    pytest.importorskip("httpx")
    scraper = AshbyScraper("ramp", company="Ramp")
    jobs = scraper.fetch()
    assert len(jobs) > 10
    assert all(j.external_id and j.title for j in jobs)
