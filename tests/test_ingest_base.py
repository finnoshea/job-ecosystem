"""Tests for jobecosystem.ingest.base: the scraper interface contract."""

from __future__ import annotations

import time

import pytest

from jobecosystem.core.models import Job
from jobecosystem.ingest import base


class StubScraper(base.Scraper):
    """Minimal concrete scraper, with fetch behavior supplied per test."""

    source = "stub"

    def __init__(self, jobs=None, error=None, delay=0.0):
        self._jobs = jobs or []
        self._error = error
        self._delay = delay

    def fetch(self):
        if self._delay:
            time.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return list(self._jobs)


def make_job(external_id="1"):
    return Job(source="stub", external_id=external_id, company="Acme", title="Eng")


# ---------------------------------------------------------------------------
# abstractness and the source contract
# ---------------------------------------------------------------------------

def test_scraper_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        base.Scraper()  # type: ignore[abstract]


def test_subclass_must_implement_fetch():
    class Incomplete(base.Scraper):
        source = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_missing_source_attribute_is_rejected():
    class NoSource(base.Scraper):
        def fetch(self):
            return []

    with pytest.raises(NotImplementedError, match="non-empty `source`"):
        NoSource().source_name


def test_source_name_defaults_to_source_attribute():
    assert StubScraper().source_name == "stub"


def test_source_can_be_overridden_via_property():
    class Renamed(base.Scraper):
        source = "raw"

        @property
        def source_name(self):
            return "mapped"

        def fetch(self):
            return []

    assert Renamed().source_name == "mapped"


def test_repr_includes_class_and_source():
    assert repr(StubScraper()) == "<StubScraper source='stub'>"


# ---------------------------------------------------------------------------
# parse_listing
# ---------------------------------------------------------------------------

class ParsingScraper(base.Scraper):
    """Concrete scraper whose parse_listing maps a minimal listing shape."""

    source = "parsing"

    def fetch(self):
        return [self.parse_listing(raw) for raw in self.raw_listings]

    def parse_listing(self, listing: dict) -> Job:
        try:
            return Job(
                source=self.source_name,
                external_id=str(listing["id"]),
                company=listing["company"],
                title=listing["title"],
                location=listing.get("location"),
            )
        except KeyError as error:
            raise base.ParseError(f"listing missing {error.args[0]!r}") from error

    def __init__(self, raw_listings=None):
        self.raw_listings = raw_listings or []


def test_parse_listing_maps_one_listing_to_a_job():
    job = ParsingScraper().parse_listing(
        {"id": 41832, "company": "Acme", "title": "Eng"}
    )
    assert job.source == "parsing"
    assert job.external_id == "41832"
    assert job.company == "Acme"
    assert job.title == "Eng"
    assert job.content_hash is None      # no description in this listing


def test_parse_listing_carries_optional_fields():
    job = ParsingScraper().parse_listing(
        {"id": 1, "company": "Acme", "title": "Eng", "location": "Remote"}
    )
    assert job.location == "Remote"
    assert job.posted_at is None


def test_parse_listing_raises_parse_error_on_missing_field():
    with pytest.raises(base.ParseError, match="listing missing 'title'"):
        ParsingScraper().parse_listing({"id": 1, "company": "Acme"})


def test_parse_listing_does_not_touch_bookkeeping_fields():
    # upsert owns these; a scraper setting them would fight the repost logic.
    job = ParsingScraper().parse_listing(
        {"id": 1, "company": "Acme", "title": "Eng"}
    )
    assert job.first_seen_at is None
    assert job.last_seen_at is None
    assert job.repost_count == 0
    assert job.status == "new"


def test_parse_listing_is_usable_without_network_access():
    # The point of separating it from fetch(): mapping can be tested against a
    # saved payload.
    job = ParsingScraper().parse_listing(
        {"id": 7, "company": "Acme", "title": "Eng"}
    )
    assert job.external_id == "7"


def test_base_parse_listing_is_not_abstract_but_must_be_overridden():
    # Subclasses that implement fetch() differently need not use it, so it is
    # deliberately not an abstractmethod.
    class FetchOnly(base.Scraper):
        source = "fetchonly"

        def fetch(self):
            return []

    scraper = FetchOnly()
    assert scraper.fetch() == []
    with pytest.raises(NotImplementedError, match="must implement parse_listing"):
        scraper.parse_listing({})


def test_fetch_uses_parse_listing_for_each_listing():
    scraper = ParsingScraper(
        [
            {"id": 1, "company": "Acme", "title": "Eng"},
            {"id": 2, "company": "Beta", "title": "Dev"},
        ]
    )
    jobs = scraper.scrape().jobs
    assert [j.external_id for j in jobs] == ["1", "2"]
    assert [j.company for j in jobs] == ["Acme", "Beta"]


def test_one_bad_listing_can_be_collected_without_losing_the_rest():
    # The runner's per-item error path: a single unparseable listing should not
    # discard the others.
    scraper = ParsingScraper()
    raw = [
        {"id": 1, "company": "Acme", "title": "Eng"},
        {"id": 2, "company": "Beta"},  # missing title
        {"id": 3, "company": "Gamma", "title": "Dev"},
    ]
    jobs, errors = [], []
    for listing in raw:
        try:
            jobs.append(scraper.parse_listing(listing))
        except base.ParseError as error:
            errors.append((None, str(error)))

    assert [j.external_id for j in jobs] == ["1", "3"]
    assert len(errors) == 1
    assert errors[0][0] is None


# ---------------------------------------------------------------------------
# error reporting
# ---------------------------------------------------------------------------

class ReportingScraper(base.Scraper):
    """Scraper that reports a recoverable error and still returns a job."""

    source = "reporting"

    def __init__(self, errors=(), fail=False):
        self.errors_to_report = list(errors)
        self.fail = fail

    def fetch(self):
        for message in self.errors_to_report:
            self.report_error(message)
        if self.fail:
            raise base.FetchError("board down")
        return [Job(source="reporting", external_id="1", company="Acme", title="Eng")]


def test_reported_errors_reach_the_scrape_result():
    result = ReportingScraper(errors=["listing 4: missing title"]).scrape()
    assert result.errors == [(None, "listing 4: missing title")]
    assert result.ok is False
    assert len(result.jobs) == 1      # a reported error does not discard jobs


def test_reported_errors_carry_a_tenant_when_given():
    # tenant is for boards with per-tenant endpoints, like Workday.
    scraper = ReportingScraper()
    scraper.report_error("timeout", tenant="acme")
    assert scraper._drain_errors() == [("acme", "timeout")]


def test_no_reported_errors_means_ok():
    assert ReportingScraper().scrape().ok is True


def test_errors_are_cleared_between_runs():
    # A reused scraper instance must not accumulate stale errors.
    scraper = ReportingScraper(errors=["boom"])
    assert len(scraper.scrape().errors) == 1
    assert len(scraper.scrape().errors) == 1


def test_report_error_works_without_super_init():
    # A subclass defining its own __init__ and skipping super().__init__()
    # must still be able to report errors.
    scraper = ReportingScraper(errors=["boom"])
    result = scraper.scrape()
    assert result.errors == [(None, "boom")]


def test_errors_are_cleared_at_the_start_of_each_run():
    # An error reported by a run that then failed for another reason must not be
    # replayed by the next attempt.
    scraper = ReportingScraper(errors=["earlier failure"], fail=True)
    with pytest.raises(base.FetchError):
        scraper.scrape()

    scraper.fail = False
    scraper.errors_to_report = []
    assert scraper.scrape().errors == []


def test_drain_errors_is_empty_when_nothing_was_reported():
    assert ReportingScraper()._drain_errors() == []


# ---------------------------------------------------------------------------
# ScrapeResult
# ---------------------------------------------------------------------------

def test_empty_result_is_ok():
    # A board with no openings is not a failure.
    result = base.ScrapeResult(source="stub")
    assert result.ok is True
    assert len(result) == 0


def test_result_with_errors_is_not_ok():
    result = base.ScrapeResult(source="stub", errors=[("tenant", "boom")])
    assert result.ok is False


def test_result_len_counts_jobs_not_errors():
    result = base.ScrapeResult(
        source="stub", jobs=[make_job("1"), make_job("2")], errors=[(None, "x")]
    )
    assert len(result) == 2


def test_errors_default_to_a_fresh_list_per_instance():
    # Field default_factory, not a shared mutable default.
    first = base.ScrapeResult(source="a")
    second = base.ScrapeResult(source="b")
    first.errors.append((None, "boom"))
    assert second.errors == []


def test_error_tenant_is_optional():
    # Sources without a tenant concept record None.
    result = base.ScrapeResult(source="ashby", errors=[(None, "timeout")])
    assert result.errors[0][0] is None


# ---------------------------------------------------------------------------
# scrape()
# ---------------------------------------------------------------------------

def test_scrape_wraps_fetch_output():
    jobs = [make_job("1"), make_job("2")]
    result = StubScraper(jobs=jobs).scrape()
    assert result.source == "stub"
    assert [j.external_id for j in result.jobs] == ["1", "2"]
    assert result.ok


def test_scrape_measures_duration():
    result = StubScraper(delay=0.02).scrape()
    assert result.duration_ms >= 20


def test_scrape_of_fetch_returning_empty_is_ok():
    result = StubScraper().scrape()
    assert result.jobs == []
    assert result.ok


def test_scrape_copies_the_job_list():
    # The result should not alias a list the scraper may still hold.
    original = [make_job("1")]
    result = StubScraper(jobs=original).scrape()
    result.jobs.append(make_job("2"))
    assert len(original) == 1


@pytest.mark.parametrize("error", [base.FetchError("down"), base.ParseError("bad")])
def test_scrape_propagates_scraper_errors(error):
    # A broken board must stay distinguishable from an empty board, so errors
    # are not swallowed into an empty result here. The runner records them.
    with pytest.raises(base.ScraperError):
        StubScraper(error=error).scrape()


def test_scrape_does_not_swallow_unexpected_errors():
    # A bug in a scraper should surface, not look like a quiet empty run.
    with pytest.raises(ZeroDivisionError):
        StubScraper(error=ZeroDivisionError()).scrape()


# ---------------------------------------------------------------------------
# error hierarchy
# ---------------------------------------------------------------------------

def test_error_hierarchy():
    assert issubclass(base.FetchError, base.ScraperError)
    assert issubclass(base.ParseError, base.ScraperError)
    assert issubclass(base.ScraperError, Exception)


def test_catching_scraper_error_catches_both_subtypes():
    for error_type in (base.FetchError, base.ParseError):
        with pytest.raises(base.ScraperError):
            raise error_type("boom")
