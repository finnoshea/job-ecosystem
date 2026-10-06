"""Tests for jobecosystem.ingest.sources.workday.

Parser tests run against a saved listing payload
(``tests/fixtures/workday_listing.json``), so they need no network. One test
hits the live API and is marked ``network``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobecosystem.ingest.base import FetchError, ParseError
from jobecosystem.ingest import runner
from jobecosystem.ingest.sources.workday import (
    LISTING_URL,
    PAGE_SIZE,
    TenantSpec,
    WorkdayScraper,
)

FIXTURE = Path(__file__).parent / "fixtures" / "workday_listing.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def make_scraper(*, pages=None, post_json=None, **kwargs) -> WorkdayScraper:
    """Scraper wired to a stub that serves the fixture, one page then empty."""
    if post_json is None:
        calls: list[dict] = []

        def post_json(url, body):
            calls.append(body)
            return pages[len(calls) - 1] if calls and len(calls) <= len(pages) else {
                "total": 0,
                "jobPostings": [],
            }

    scraper = WorkdayScraper(
        "asml.wd3", "asml", "ASMLPrivate1", company="ASML", post_json=post_json, **kwargs
    )
    scraper.calls = getattr(post_json, "calls", [])
    return scraper


def listing(**overrides) -> dict:
    """A minimal Workday posting."""
    posting = {
        "title": "Software Engineer",
        "externalPath": "/job/Veldhoven-Netherlands/Software-Engineer_J-123",
        "locationsText": "Veldhoven, Netherlands",
        "postedOn": "Posted Today",
        "bulletFields": ["J-123"],
    }
    posting.update(overrides)
    return posting


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def test_listing_url_is_built_from_the_three_parts():
    scraper = WorkdayScraper("nvidia.wd5", "nvidia", "NVIDIAExternalCareerSite")
    assert scraper.listing_url == LISTING_URL.format(
        host="nvidia.wd5", tenant="nvidia", site="NVIDIAExternalCareerSite"
    )


def test_source_label_includes_tenant_and_site():
    # One tenant can expose several sites, so the site must be in the label.
    scraper = WorkdayScraper("salesforce.wd12", "salesforce", "Slack")
    assert scraper.source == "workday:salesforce:Slack"


def test_company_defaults_to_the_tenant():
    assert WorkdayScraper("asml.wd3", "asml", "ASMLPrivate1").company == "asml"


def test_company_can_be_a_display_name():
    scraper = WorkdayScraper("asml.wd3", "asml", "ASMLPrivate1", company="ASML")
    assert scraper.company == "ASML"


@pytest.mark.parametrize("field", ["host", "tenant", "site"])
def test_empty_addressing_parts_are_rejected(field):
    parts = {"host": "h.wd1", "tenant": "t", "site": "s"}
    parts[field] = ""
    with pytest.raises(ValueError, match=field):
        WorkdayScraper(**parts)


def test_from_spec_round_trips():
    spec = TenantSpec(host="nvidia.wd5", tenant="nvidia", site="Site", company="Nvidia")
    scraper = WorkdayScraper.from_spec(spec)
    assert (scraper.host, scraper.tenant, scraper.site) == ("nvidia.wd5", "nvidia", "Site")
    assert scraper.company == "Nvidia"


def test_page_size_is_the_protocol_constant():
    # Workday rejects limit > 20 with HTTP 400, so this is not a tunable.
    assert PAGE_SIZE == 20


# ---------------------------------------------------------------------------
# parse_listing
# ---------------------------------------------------------------------------

def test_parse_listing_maps_the_listing_fields():
    job = make_scraper().parse_listing(listing())
    assert job.source == "workday:asml:ASMLPrivate1"
    assert job.company == "ASML"
    assert job.title == "Software Engineer"
    assert job.location == "Veldhoven, Netherlands"
    assert job.external_id == "J-123"


def test_external_id_prefers_the_requisition_number():
    # bulletFields[0] survives a rename, unlike the slug in externalPath.
    job = make_scraper().parse_listing(
        listing(externalPath="/job/X/Old-Slug_J-999", bulletFields=["JR999"])
    )
    assert job.external_id == "JR999"


def test_external_id_falls_back_to_the_path():
    job = make_scraper().parse_listing(
        listing(externalPath="/job/X/Slug_J-1", bulletFields=None)
    )
    assert job.external_id == "/job/X/Slug_J-1"


def test_external_id_skips_blank_bullet_fields():
    job = make_scraper().parse_listing(
        listing(externalPath="/job/X/S_J-1", bulletFields=["  ", ""])
    )
    assert job.external_id == "/job/X/S_J-1"


def test_external_id_skips_non_string_bullet_fields():
    job = make_scraper().parse_listing(
        listing(externalPath="/job/X/S_J-1", bulletFields=[None, 7])
    )
    assert job.external_id == "/job/X/S_J-1"


def test_parse_listing_ignores_posted_on_prose():
    # "Posted 5 Days Ago" would become a different date on every run.
    job = make_scraper().parse_listing(listing(postedOn="Posted 5 Days Ago"))
    assert job.posted_at is None


def test_parse_listing_leaves_description_unset():
    job = make_scraper().parse_listing(listing())
    assert job.description is None
    assert job.content_hash is None


def test_parse_listing_sets_a_description_url():
    job = make_scraper().parse_listing(listing())
    assert job.description_url == (
        "https://asml.wd3.myworkdayjobs.com/wday/cxs/asml/ASMLPrivate1"
        "/job/Veldhoven-Netherlands/Software-Engineer_J-123"
    )


def test_public_url_is_the_human_facing_one():
    job = make_scraper().parse_listing(listing())
    assert job.url == (
        "https://asml.wd3.myworkdayjobs.com/ASMLPrivate1"
        "/job/Veldhoven-Netherlands/Software-Engineer_J-123"
    )
    assert "/wday/cxs/" not in job.url


def test_missing_title_raises_parse_error():
    with pytest.raises(ParseError, match="missing 'title'"):
        make_scraper().parse_listing({"externalPath": "/job/X/Y_1"})


def test_missing_external_path_raises_parse_error():
    with pytest.raises(ParseError, match="missing 'externalPath'"):
        make_scraper().parse_listing({"title": "Engineer"})


def test_blank_title_raises_parse_error():
    with pytest.raises(ParseError, match="missing 'title'"):
        make_scraper().parse_listing(listing(title="   "))


def test_title_is_stripped():
    job = make_scraper().parse_listing(listing(title="  Engineer  "))
    assert job.title == "Engineer"


def test_missing_location_is_none():
    job = make_scraper().parse_listing(listing(locationsText=None))
    assert job.location is None


def test_raw_json_round_trips():
    posting = listing()
    job = make_scraper().parse_listing(posting)
    assert json.loads(job.raw_json) == posting


def test_real_payload_parses(payload):
    scraper = make_scraper()
    jobs = [scraper.parse_listing(p) for p in payload["jobPostings"]]
    assert len(jobs) == 4
    for job in jobs:
        assert job.external_id
        assert job.title
        assert job.description_url.startswith("https://")
        assert job.content_hash is None


# ---------------------------------------------------------------------------
# pagination
# ---------------------------------------------------------------------------

def test_fetch_collects_a_single_page():
    page = {"total": 2, "jobPostings": [listing(bulletFields=["A"]),
                                        listing(bulletFields=["B"])]}
    scraper = make_scraper(pages=[page])
    jobs = scraper.fetch()
    assert [j.external_id for j in jobs] == ["A", "B"]


def test_fetch_paginates_until_total_is_reached():
    pages = [
        {"total": 3, "jobPostings": [listing(bulletFields=[f"A{i}"]) for i in range(2)]},
        {"total": 3, "jobPostings": [listing(bulletFields=["A2"])]},
    ]
    scraper = make_scraper(pages=pages)
    jobs = scraper.fetch()
    assert [j.external_id for j in jobs] == ["A0", "A1", "A2"]


def test_fetch_advances_offset_by_page_size():
    offsets = []

    def post_json(url, body):
        offsets.append(body["offset"])
        if body["offset"] == 0:
            return {"total": 40, "jobPostings": [listing(bulletFields=["A"])]}
        return {"total": 40, "jobPostings": []}

    WorkdayScraper("h.wd1", "t", "s", post_json=post_json).fetch()
    assert offsets == [0, PAGE_SIZE]


def test_fetch_sends_the_expected_body():
    bodies = []

    def post_json(url, body):
        bodies.append(body)
        return {"total": 0, "jobPostings": []}

    WorkdayScraper("h.wd1", "t", "s", post_json=post_json).fetch()
    assert bodies[0] == {
        "appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""
    }


def test_fetch_stops_when_a_page_is_empty():
    pages = [
        {"total": 99, "jobPostings": [listing(bulletFields=["A"])]},
        {"total": 99, "jobPostings": []},
    ]
    scraper = make_scraper(pages=pages)
    assert len(scraper.fetch()) == 1


def test_fetch_stops_when_pagination_stalls():
    # A board ignoring offset would otherwise loop until max_pages.
    same = {"total": 99, "jobPostings": [listing(bulletFields=["A"])]}

    def post_json(url, body):
        return same

    scraper = WorkdayScraper("h.wd1", "t", "s", post_json=post_json)
    assert len(scraper.fetch()) == 1


def test_fetch_deduplicates_repeated_ids():
    same = {"total": 5, "jobPostings": [listing(bulletFields=["A"])]}

    def post_json(url, body):
        return same

    scraper = WorkdayScraper("h.wd1", "t", "s", post_json=post_json)
    assert len(scraper.fetch()) == 1


def test_fetch_respects_max_pages():
    calls = []

    def post_json(url, body):
        calls.append(body)
        return {"total": 10_000, "jobPostings": [listing(bulletFields=[str(len(calls))])]}

    scraper = WorkdayScraper("h.wd1", "t", "s", post_json=post_json, max_pages=3)
    scraper.fetch()
    assert len(calls) == 3


def test_fetch_reports_a_shortfall_against_total():
    # The gap between `total` and what was collected is worth surfacing.
    pages = [{"total": 50, "jobPostings": [listing(bulletFields=["A"])]},
             {"total": 50, "jobPostings": []}]
    result = make_scraper(pages=pages).scrape()
    assert result.ok is False
    assert any("1 of 50" in message for _, message in result.errors)


def test_fetch_is_quiet_when_total_matches():
    pages = [{"total": 1, "jobPostings": [listing(bulletFields=["A"])]}]
    assert make_scraper(pages=pages).scrape().ok is True


def test_fetch_tolerates_a_missing_total():
    pages = [{"jobPostings": [listing(bulletFields=["A"])]},
             {"jobPostings": []}]
    result = make_scraper(pages=pages).scrape()
    assert len(result.jobs) == 1
    assert result.ok is True


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------

def test_first_page_failure_raises():
    def post_json(url, body):
        raise FetchError("connection refused")

    with pytest.raises(FetchError):
        WorkdayScraper("h.wd1", "t", "s", post_json=post_json).fetch()


def test_later_page_failure_keeps_what_was_collected():
    calls = []

    def post_json(url, body):
        calls.append(body)
        if body["offset"] == 0:
            return {"total": 40, "jobPostings": [listing(bulletFields=["A"])]}
        raise FetchError("connection reset")

    scraper = WorkdayScraper("h.wd1", "t", "s", post_json=post_json)
    result = scraper.scrape()
    assert len(result.jobs) == 1
    assert result.ok is False
    assert any("offset 20" in message for _, message in result.errors)


def test_unexpected_exception_becomes_a_fetch_error():
    def post_json(url, body):
        raise RuntimeError("boom")

    with pytest.raises(FetchError, match="could not read"):
        WorkdayScraper("h.wd1", "t", "s", post_json=post_json).fetch()


def test_response_without_job_postings_raises():
    def post_json(url, body):
        return {"total": 5}

    with pytest.raises(FetchError, match="no 'jobPostings'"):
        WorkdayScraper("h.wd1", "t", "s", post_json=post_json).fetch()


def test_non_object_response_raises():
    def post_json(url, body):
        return ["not", "an", "object"]

    with pytest.raises(FetchError, match="expected an object"):
        WorkdayScraper("h.wd1", "t", "s", post_json=post_json).fetch()


def test_non_list_job_postings_raises():
    def post_json(url, body):
        return {"jobPostings": "nope"}

    with pytest.raises(FetchError, match="expected a list"):
        WorkdayScraper("h.wd1", "t", "s", post_json=post_json).fetch()


def test_bad_items_are_reported_not_dropped_silently():
    page = {
        "total": 3,
        "jobPostings": [listing(bulletFields=["A"]), {"no_title": True}, "junk"],
    }
    result = make_scraper(pages=[page]).scrape()
    assert len(result.jobs) == 1
    messages = [message for _, message in result.errors]
    assert any("item 1" in m and "missing 'title'" in m for m in messages)
    assert any("item 2" in m and "expected an object" in m for m in messages)
    # The shortfall against total is reported too, since 1 of 3 were collected.
    assert any("1 of 3" in m for m in messages)


# ---------------------------------------------------------------------------
# through the runner
# ---------------------------------------------------------------------------

def test_runner_stores_jobs_without_descriptions(conn):
    page = {"total": 2, "jobPostings": [listing(bulletFields=["A"]),
                                        listing(bulletFields=["B"])]}
    summary = runner.run_scrapers(conn, [make_scraper(pages=[page])])
    outcome = summary.outcomes[0]
    assert outcome.ok
    assert outcome.inserted == 2

    row = conn.execute("SELECT * FROM jobs WHERE external_id = 'A'").fetchone()
    assert row["description"] is None
    assert row["content_hash"] is None
    assert row["description_fetched_at"] is None
    assert row["description_url"] is not None
    assert row["rating"] is None
    assert row["status"] == "new"


def test_stored_jobs_are_queued_for_description_fetch(conn):
    page = {"total": 2, "jobPostings": [listing(bulletFields=["A"]),
                                       listing(bulletFields=["B"])]}
    runner.run_scrapers(conn, [make_scraper(pages=[page])])
    assert conn.execute(
        "SELECT COUNT(*) FROM jobs_needing_descriptions"
    ).fetchone()[0] == 2


def test_no_false_reposts_before_descriptions_arrive(conn):
    # Two same-titled rows with NULL hashes must not look like a repost.
    page = {"total": 2, "jobPostings": [
        listing(bulletFields=["A"], title="Software Engineer"),
        listing(bulletFields=["B"], title="Software Engineer"),
    ]}
    runner.run_scrapers(conn, [make_scraper(pages=[page])])
    assert conn.execute("SELECT COUNT(*) FROM jobs_reposted").fetchone()[0] == 0


def test_rerunning_workday_inserts_nothing_new(conn):
    page = {"total": 1, "jobPostings": [listing(bulletFields=["A"])]}
    runner.run_scrapers(conn, [make_scraper(pages=[page])])
    summary = runner.run_scrapers(conn, [make_scraper(pages=[page])])
    assert summary.outcomes[0].inserted == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_two_sites_of_one_tenant_are_separate_sources(conn):
    page = {"total": 1, "jobPostings": [listing(bulletFields=["A"])]}
    main = WorkdayScraper("sf.wd12", "sf", "Main", post_json=lambda u, b: page)
    other = WorkdayScraper("sf.wd12", "sf", "Other", post_json=lambda u, b: page)
    runner.run_scrapers(conn, [main, other])
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert [r[0] for r in conn.execute(
        "SELECT DISTINCT source FROM jobs ORDER BY source"
    )] == ["workday:sf:Main", "workday:sf:Other"]


# ---------------------------------------------------------------------------
# live site (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.network
@pytest.mark.slow
def test_live_workday_site_is_reachable(conn):
    """Hits the real Workday API; run with ``-m network``."""
    pytest.importorskip("httpx")
    scraper = WorkdayScraper("asml.wd3", "asml", "ASMLPrivate1", company="ASML")
    scraper.max_pages = 2
    result = scraper.scrape()
    assert len(result.jobs) == PAGE_SIZE * 2
    assert all(j.external_id and j.title for j in result.jobs)
    assert all(j.description_url for j in result.jobs)
