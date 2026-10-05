"""Lever job-board scraper.

Lever exposes a public, unauthenticated JSON API per company::

    GET https://api.lever.co/v0/postings/{site}?mode=json

The response is a bare JSON array of postings. Unlike Greenhouse and
SmartRecruiters, a Lever posting already carries its **full description** --
``descriptionPlain`` (the intro), ``lists`` (named sections whose ``content`` is
HTML), and ``additionalPlain`` -- so there is no second request per job and no
``parse_description`` parser. That mirrors Ashby, not the two sources that fetch
descriptions later.

Deliberate choices:

* **One site per scraper.** The site slug is part of the URL, so the ``source``
  label is ``lever:{slug}`` and the list lives in ``lever_companies.txt``.
* **A 404 is a dead board, not a failure.** That list is grown by
  ``jobecosystem-discover`` from Hacker News links, most of which are years old;
  a closed or renamed board is expected. Such a site reports an error and
  returns no jobs, so one stale slug does not fail the whole source. (Greenhouse
  treats 404 as fatal because its list is curated by hand.)
* **The company name comes from the config, not the API.** A Lever listing has
  no company field, so the display name is the ``lever_companies.txt`` entry or
  the slug.

The HTTP dependency is injected (``fetch_json``) so the parser is testable
against a saved payload with no network access.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...core.models import Job
from ..base import FetchError, ParseError, Scraper
from . import companies
from .description import html_to_text

#: Lever's public posting list, per site.
LISTING_URL = "https://api.lever.co/v0/postings/{site}?mode=json"

#: Signature of the injectable HTTP call: URL -> decoded JSON.
FetchJson = Callable[[str], Any]

#: src/jobecosystem/ingest/sources/lever.py
#:   -> sources -> ingest -> jobecosystem -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_COMPANIES_FILE = _REPO_ROOT / "lever_companies.txt"


class BoardGone(FetchError):
    """The site has no Lever board (HTTP 404).

    Expected for slugs discovered from old listings, so it is handled as "no
    jobs here" rather than a failed source.
    """


@dataclass(frozen=True, slots=True)
class CompanySpec:
    """One configured site: the slug, plus an optional display name."""

    slug: str
    name: str | None = None

    @property
    def source(self) -> str:
        """The ``jobs.source`` label: ``lever:<slug>``."""
        return f"lever:{self.slug}"


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """What was understood from a companies file, and what was not."""

    specs: list[CompanySpec]
    errors: list[tuple[int, str]]

    def __bool__(self) -> bool:
        return bool(self.specs)

    def __len__(self) -> int:
        return len(self.specs)


class LeverScraper(Scraper):
    """Scrape one Lever site.

    ``site`` is the company slug (the segment in ``jobs.lever.co/{site}``). It
    doubles as the ``source`` label and as the display name unless ``name`` is
    given.
    """

    def __init__(
        self,
        site: str,
        *,
        name: str | None = None,
        source: str | None = None,
        fetch_json: FetchJson | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not site:
            raise ValueError("site must be a non-empty string")
        self.site = site
        self.source = source or f"lever:{site}"
        # Display name for SQL table ``jobs``'s ``company`` column; Lever has no
        # company field to prefer.
        self.company = name or site
        self._name = name
        self.timeout = timeout
        self._fetch_json = fetch_json

    @property
    def listing_url(self) -> str:
        """The endpoint this scraper reads."""
        return LISTING_URL.format(site=self.site)

    def fetch(self) -> list[Job]:
        """Fetch every posting on the site.

        A missing board (404) is reported and yields no jobs; any other read
        failure raises :class:`FetchError`. Per-listing problems go through
        :meth:`report_error`, so one malformed posting does not discard the rest.
        """
        try:
            payload = self._get_json()
        except BoardGone as error:
            self.report_error(str(error))
            return []

        if not isinstance(payload, list):
            raise FetchError(
                f"lever site {self.site!r}: response was"
                f" {type(payload).__name__}, expected a list"
            )

        jobs: list[Job] = []
        seen: set[str] = set()
        for index, posting in enumerate(payload):
            if not isinstance(posting, dict):
                self.report_error(
                    f"listing {index}: expected an object,"
                    f" got {type(posting).__name__}"
                )
                continue
            try:
                job = self.parse_listing(posting)
            except ParseError as error:
                identifier = posting.get("id")
                label = f"job {identifier}" if identifier is not None else f"listing {index}"
                self.report_error(f"{label}: {error}")
                continue
            if job.external_id in seen:
                continue
            seen.add(job.external_id)
            jobs.append(job)
        return jobs

    def parse_listing(self, posting: dict) -> Job:
        """Map one Lever posting onto a :class:`Job`.

        Requires ``id`` and ``text``. The description is set here -- a Lever
        listing carries the whole thing -- so no ``description_url`` is needed
        and the row never enters the description-fetch queue.
        """
        external_id = _required_str(posting, "id")
        title = _required_str(posting, "text")

        return Job(
            source=self.source,
            external_id=external_id,
            company=self._name or self.site,
            title=title,
            location=_location(posting),
            description=lever_description(posting),
            url=_clean(posting.get("hostedUrl")) or _clean(posting.get("applyUrl")),
            posted_at=_timestamp_ms(posting.get("createdAt")),
            raw_json=_raw(posting),
        )

    def _get_json(self) -> Any:
        fetcher = self._fetch_json or _http_get_json
        try:
            return fetcher(self.listing_url)
        except FetchError:
            raise
        except Exception as error:  # noqa: BLE001 - normalized for the runner
            raise FetchError(
                f"lever site {self.site!r}: could not read"
                f" {self.listing_url}: {error}"
            ) from error


def lever_description(posting: dict) -> str | None:
    """The full posting text, from the parts Lever splits it into.

    ``descriptionPlain`` is the intro, ``lists`` holds named sections whose
    ``content`` is HTML, and ``additionalPlain`` is a closing note. They are
    joined so the stored text is the whole posting, not just the intro.
    """
    parts: list[str] = []

    intro = _clean(posting.get("descriptionPlain"))
    if intro:
        parts.append(intro)

    for section in posting.get("lists") or []:
        if not isinstance(section, dict):
            continue
        heading = _clean(section.get("text"))
        body = html_to_text(section.get("content") or "")
        if heading and body:
            parts.append(f"{heading}\n{body}")
        elif heading:
            parts.append(heading)
        elif body:
            parts.append(body)

    additional = _clean(posting.get("additionalPlain"))
    if additional:
        parts.append(additional)

    joined = "\n\n".join(part for part in parts if part)
    return joined or None


# ---------------------------------------------------------------------------
# company list
# ---------------------------------------------------------------------------

def resolve_companies_file(path: str | Path | None = None) -> Path:
    """Resolve the companies file: argument > ``LEVER_COMPANIES_FILE`` >
    ``lever_companies.txt`` at the repo root."""
    return companies.resolve_file(
        path, env_var="LEVER_COMPANIES_FILE", default=DEFAULT_COMPANIES_FILE
    )


def _specific(outcome: companies.ParseOutcome) -> ParseOutcome:
    """Map the shared tokenized entries onto Lever specs."""
    return ParseOutcome(
        specs=[
            CompanySpec(slug=entry.slug, name=entry.name)
            for entry in outcome.entries
        ],
        errors=outcome.errors,
    )


def parse_companies(text: str) -> ParseOutcome:
    """Parse companies-file text into specs, collecting anything unusable.

    Each line is ``slug [Display Name]``. Comments, blank lines, and duplicate
    slugs are skipped rather than raising -- one typo must not stop the daily
    scrape. Duplicates keep the first occurrence.
    """
    return _specific(companies.parse_lines(text))


def load_companies(path: str | Path | None = None) -> ParseOutcome:
    """Load and parse the companies file.

    Returns an empty outcome when the file is missing, rather than raising: a
    missing optional config file means "no sites configured".
    """
    return _specific(companies.load_lines(
        path, env_var="LEVER_COMPANIES_FILE", default=DEFAULT_COMPANIES_FILE
    ))


def build_scrapers(
    path: str | Path | None = None,
    *,
    fetch_json: FetchJson | None = None,
) -> list[Scraper]:
    """Build one :class:`LeverScraper` per configured site.

    ``fetch_json`` is passed through for tests, which supply a stub instead of
    hitting the network.
    """
    return [
        LeverScraper(spec.slug, name=spec.name, fetch_json=fetch_json)
        for spec in load_companies(path).specs
    ]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _http_get_json(url: str, *, timeout: float = 30.0) -> Any:
    """Default fetcher: a plain GET returning decoded JSON.

    A 404 becomes :class:`BoardGone` so the caller can skip a dead board;
    everything else is a normal :class:`FetchError`. Imported lazily so the
    parser can be used, and tested, without httpx.
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

    if response.status_code == 404:
        raise BoardGone(f"GET {url}: board not found (404)")
    if response.status_code >= 400:
        raise FetchError(f"GET {url}: HTTP {response.status_code}")

    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise FetchError(f"GET {url}: response was not JSON: {error}") from error


def _required_str(posting: dict, key: str) -> str:
    """Read a required non-empty string, or raise :class:`ParseError`."""
    value = posting.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParseError(f"missing {key!r}")
    return value.strip()


def _clean(value: Any) -> str | None:
    """Trimmed string, or ``None`` when absent or blank."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _location(posting: dict) -> str | None:
    """Location string from ``categories.location``, falling back to all sites."""
    categories = posting.get("categories")
    if not isinstance(categories, dict):
        return None

    location = _clean(categories.get("location"))
    if location:
        return location

    all_locations = categories.get("allLocations")
    if isinstance(all_locations, list):
        joined = ", ".join(
            value.strip()
            for value in all_locations
            if isinstance(value, str) and value.strip()
        )
        if joined:
            return joined
    return None


def _timestamp_ms(value: Any) -> str | None:
    """Normalize Lever's ``createdAt`` (epoch milliseconds) to ``...Z``.

    Accepts an ISO string too, since a saved payload or future change might use
    one. Returns ``None`` when absent or unparseable.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        try:
            parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def _raw(posting: dict) -> str | None:
    """The vendor payload, for debugging a parser that broke."""
    try:
        return json.dumps(posting, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
