"""Workable job scraper: the cross-company marketplace feed.

Workable's per-company widget endpoint
(``apply.workable.com/api/v1/widget/accounts/{account}``) is unreliable -- for
many real accounts it returns an empty ``jobs`` list, and the SPI v3 endpoint
requires a token -- so this scraper uses Workable's own job marketplace instead::

    GET https://jobs.workable.com/api/v1/jobs[?query=&location=&workplace=]
        [&pageToken=...]

That endpoint is public and unauthenticated, lists jobs for **every** Workable
company (about 170k at the time of writing), and paginates by returning a
``nextPageToken`` that is passed back as ``pageToken``. Each record carries the
full ``description`` (plus ``requirementsSection`` and ``benefitsSection``), so
-- like Lever and Ashby -- no second request per job is needed and there is no
``parse_description`` parser.

Deliberate choices:

* **No company list.** One source covers every company, which is the point: the
  marketplace is where the small-company tail lives. There is no
  ``workable_companies.txt`` to maintain.
* **A capped rolling window, not a full crawl.** There is no ``since``/date
  filter, and a full crawl would be ~8,500 pages, so each run reads at most
  ``pages`` pages (newest first) and relies on ``ingest.upsert`` to dedupe what
  it has already seen. ``DEFAULT_PAGES`` is the knob; ``--workable-pages``
  overrides it per run.
* **Optional server-side filters.** ``query``, ``location`` and ``workplace``
  narrow the feed before it is returned, which is cheaper than fetching
  everything and discarding most of it.

The feed is an internal API of Workable's marketplace, more likely to change
than the per-company endpoints, so the ``jobs`` list is validated before use.

The HTTP dependency is injected (``fetch_json``) so the parser is testable
against a saved payload with no network access.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

from ...core.models import Job
from ..base import FetchError, ParseError, Scraper
from .description import html_to_text

#: Workable's public marketplace feed, across all companies.
LISTING_URL = "https://jobs.workable.com/api/v1/jobs"

#: Records the feed returns per page. Not configurable -- it is the server's
#: choice -- but named so the default page budget is readable as a row count.
PAGE_SIZE = 20

#: Pages to read per run, 20 jobs each. A cap, not a target: the feed is
#: ordered newest-first, so a rolling window catches new postings without
#: crawling the whole marketplace. Editable at runtime with ``--workable-pages``.
DEFAULT_PAGES = 25

#: Signature of the injectable HTTP call: URL -> decoded JSON.
FetchJson = Callable[[str], Any]


class WorkableScraper(Scraper):
    """Scrape Workable's cross-company marketplace feed.

    ``query``, ``location`` and ``workplace`` are passed through as server-side
    filters; ``pages`` caps how many pages of 20 records are read.
    """

    source = "workable"

    def __init__(
        self,
        *,
        query: str | None = None,
        location: str | None = None,
        workplace: str | None = None,
        pages: int = DEFAULT_PAGES,
        fetch_json: FetchJson | None = None,
        timeout: float = 30.0,
    ) -> None:
        if pages < 1:
            raise ValueError("pages must be at least 1")
        self.query = query
        self.location = location
        self.workplace = workplace
        self.pages = pages
        self.timeout = timeout
        self._fetch_json = fetch_json

    def page_url(self, page_token: str | None = None) -> str:
        """The feed URL for one page, with any configured filters."""
        params: dict[str, str] = {}
        if self.query:
            params["query"] = self.query
        if self.location:
            params["location"] = self.location
        if self.workplace:
            params["workplace"] = self.workplace
        if page_token:
            params["pageToken"] = page_token
        return f"{LISTING_URL}?{urlencode(params)}" if params else LISTING_URL

    def fetch(self) -> list[Job]:
        """Read up to ``pages`` pages of the feed.

        Raises :class:`FetchError` if the first page cannot be read; a failure
        on a *later* page ends pagination with what was collected so far and
        reports it, since a partial feed beats none. Per-record problems go
        through :meth:`report_error`.
        """
        jobs: list[Job] = []
        seen: set[str] = set()
        token: str | None = None

        for page in range(self.pages):
            url = self.page_url(token)
            try:
                payload = self._get_page(url)
            except FetchError:
                if page == 0:
                    raise
                self.report_error(
                    f"pagination stopped after {len(jobs)} jobs"
                )
                break

            records = payload.get("jobs")
            if not isinstance(records, list):
                raise FetchError(
                    f"workable: response has no 'jobs' list ({url})"
                )
            if not records:
                break

            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    self.report_error(
                        f"page {page + 1} item {index}: expected an object,"
                        f" got {type(record).__name__}"
                    )
                    continue
                try:
                    job = self.parse_listing(record)
                except ParseError as error:
                    identifier = record.get("id")
                    label = (
                        f"job {identifier}" if identifier is not None
                        else f"page {page + 1} item {index}"
                    )
                    self.report_error(f"{label}: {error}")
                    continue
                if job.external_id in seen:
                    continue
                seen.add(job.external_id)
                jobs.append(job)

            token = payload.get("nextPageToken")
            if not token:
                break
        return jobs

    def parse_listing(self, record: dict) -> Job:
        """Map one marketplace record onto a :class:`Job`.

        Requires ``id`` and ``title``. The description is set here -- the record
        carries the whole thing -- so no ``description_url`` is needed and the
        row never enters the description-fetch queue.
        """
        external_id = _required_str(record, "id")
        title = _required_str(record, "title")

        return Job(
            source=self.source,
            external_id=external_id,
            company=_company(record),
            title=title,
            location=_location(record),
            description=_description(record),
            url=_clean(record.get("url")),
            posted_at=_timestamp(record.get("created")),
            raw_json=_raw(record),
        )

    def _get_page(self, url: str) -> dict:
        """Request one page of the feed, normalized to an object."""
        fetcher = self._fetch_json or _http_get_json
        try:
            payload = fetcher(url)
        except FetchError:
            raise
        except Exception as error:  # noqa: BLE001 - normalized for the runner
            raise FetchError(f"workable: could not read {url}: {error}") from error

        if not isinstance(payload, dict):
            raise FetchError(
                f"workable: response was {type(payload).__name__},"
                f" expected an object ({url})"
            )
        if "jobs" not in payload:
            raise FetchError(f"workable: response has no 'jobs' list ({url})")
        return payload


def build_scrapers(
    *,
    fetch_json: FetchJson | None = None,
    pages: int = DEFAULT_PAGES,
    query: str | None = None,
    location: str | None = None,
    workplace: str | None = None,
) -> list[Scraper]:
    """Build the single marketplace scraper.

    There is no companies file: one scraper covers every Workable company.
    ``fetch_json`` is passed through for tests, which supply a stub instead of
    hitting the network.
    """
    return [WorkableScraper(
        query=query,
        location=location,
        workplace=workplace,
        pages=pages,
        fetch_json=fetch_json,
    )]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _http_get_json(url: str, *, timeout: float = 30.0) -> Any:
    """Default fetcher: a plain GET returning decoded JSON.

    Imported lazily so the parser can be used, and tested, without httpx.
    """
    import httpx

    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as error:
        raise FetchError(f"GET {url} failed: {error}") from error

    content_type = response.headers.get("content-type") or "unknown"
    if response.status_code >= 400:
        raise FetchError(
            f"GET {url}: HTTP {response.status_code} ({content_type})"
        )
    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise FetchError(
            f"GET {url}: HTTP {response.status_code} ({content_type}):"
            f" response was not JSON: {error}"
        ) from error


def _required_str(record: dict, key: str) -> str:
    """Read a required non-empty string, or raise :class:`ParseError`."""
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParseError(f"missing {key!r}")
    return value.strip()


def _clean(value: Any) -> str | None:
    """Trimmed string, or ``None`` when absent or blank."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _company(record: dict) -> str:
    """Company display name from the nested ``company`` object."""
    company = record.get("company")
    if isinstance(company, dict):
        name = _clean(company.get("title"))
        if name:
            return name
    return "Unknown"


def _location(record: dict) -> str | None:
    """Location string, preferring the ``locations`` list.

    ``TELECOMMUTE`` is the feed's remote sentinel; it is shown as ``Remote``
    because the raw token reads like noise in a job list.
    """
    locations = record.get("locations")
    if isinstance(locations, list):
        parts = [
            "Remote" if value.strip().upper() == "TELECOMMUTE" else value.strip()
            for value in locations
            if isinstance(value, str) and value.strip()
        ]
        if parts:
            return ", ".join(parts)

    location = record.get("location")
    if isinstance(location, dict):
        joined = ", ".join(
            value.strip()
            for value in (
                location.get("city"),
                location.get("subregion"),
                location.get("countryName"),
            )
            if isinstance(value, str) and value.strip()
        )
        if joined:
            return joined

    return _clean(record.get("workplace"))


def _description(record: dict) -> str | None:
    """Full posting text from the record's HTML sections.

    The feed splits it into ``description``, ``requirementsSection`` and
    ``benefitsSection``; all three are concatenated in that order.
    """
    parts: list[str] = []
    for key in ("description", "requirementsSection", "benefitsSection"):
        raw = record.get(key)
        if isinstance(raw, str) and raw.strip():
            text = html_to_text(raw)
            if text:
                parts.append(text)
    joined = "\n\n".join(parts)
    return joined or None


def _timestamp(value: Any) -> str | None:
    """Normalize Workable's timestamp to ``YYYY-MM-DDTHH:MM:SSZ``.

    ``created`` arrives as ``2026-10-05T16:06:55.362Z``; fractional seconds and
    any offset are dropped so comparisons stay consistent with the other
    sources. Returns ``None`` when absent or unparseable.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _raw(record: dict) -> str | None:
    """The vendor payload, for debugging a parser that broke."""
    try:
        return json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
