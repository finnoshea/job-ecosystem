"""SmartRecruiters job scraper.

SmartRecruiters exposes a public, unauthenticated JSON API per company::

    GET https://api.smartrecruiters.com/v1/companies/{company}/postings
        ?limit=100&offset=0[&country=us]

The response is ``{"offset": ..., "limit": ..., "totalFound": ..., "content":
[...]}``. Each item carries metadata only -- ``id``, ``name``, ``refNumber``,
``releasedDate``, ``location``, ``company``, ``department`` -- and **no
description text**. The full text lives behind a second request per job::

    GET https://api.smartrecruiters.com/v1/companies/{company}/postings/{id}
        -> jobAd.sections.{companyDescription, jobDescription,
                           qualifications, additionalInformation}

Deliberate choices, mirroring :mod:`jobecosystem.ingest.sources.workday`:

* **No description on the listing pass.** Fetching one per job would turn a
  handful of list requests into thousands. Rows are stored description-less
  (``content_hash`` NULL) with ``description_url`` populated, and the paced
  fetch is a separate operation -- see :func:`fetch_description`, which is a
  stub for now.
* **One company per scraper.** The company identifier is part of the URL, so a
  company with two SmartRecruiters sites needs two entries (and the ``source``
  label is the identifier, keeping rows distinguishable).
* **Country is a server-side filter.** ``country=us`` narrows the list before
  it is returned (Red Bull: 1230 globally, 470 in the US), which is cheaper
  than fetching everything and discarding most of it.

The list of companies is read from ``smartrecruiters_companies.txt`` at the
repo root; see that file for how to find an identifier.

The HTTP dependency is injected (``fetch_json``) so the parser is testable
against a saved payload with no network access.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from ...core.models import Job
from ..base import FetchError, ParseError, Scraper

#: SmartRecruiters' public posting list, per company.
LISTING_URL = "https://api.smartrecruiters.com/v1/companies/{company}/postings"

#: The per-job detail endpoint, where the description actually lives.
DETAIL_URL = "https://api.smartrecruiters.com/v1/companies/{company}/postings/{posting}"

#: Human-facing posting URL. The API's ``postingUrl`` (a slugged variant of
#: this) only appears on the detail response, so the listing pass builds the
#: id-only form, which SmartRecruiters serves directly.
PUBLIC_URL = "https://jobs.smartrecruiters.com/{company}/{posting}"

#: SmartRecruiters accepts larger limits, but 100 keeps a page small enough to
#: parse quickly and matches the documented maximum.
PAGE_SIZE = 100

#: Refuse to page forever if a company reports an implausible total or ignores
#: ``offset``. 100 pages is 10,000 openings, far past any real board.
MAX_PAGES = 100

#: Signature of the injectable HTTP call: URL -> decoded JSON.
FetchJson = Callable[[str], Any]

#: src/jobecosystem/ingest/sources/smartrecruiters.py
#:   -> sources -> ingest -> jobecosystem -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_COMPANIES_FILE = _REPO_ROOT / "smartrecruiters_companies.txt"


@dataclass(frozen=True, slots=True)
class CompanySpec:
    """One configured company: its identifier plus optional filters/name."""

    identifier: str
    country: str | None = None
    name: str | None = None

    @property
    def source(self) -> str:
        """The ``jobs.source`` label: ``smartrecruiters:<identifier>``."""
        return f"smartrecruiters:{self.identifier}"


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """What was understood from a companies file, and what was not."""

    specs: list[CompanySpec]
    errors: list[tuple[int, str]]

    def __bool__(self) -> bool:
        return bool(self.specs)

    def __len__(self) -> int:
        return len(self.specs)


class SmartRecruitersScraper(Scraper):
    """Scrape one SmartRecruiters company.

    ``company`` is the SmartRecruiters identifier from the careers URL (the
    segment after ``jobs.smartrecruiters.com/``). It doubles as the ``source``
    label and as the display name unless ``name`` is given.
    """

    def __init__(
        self,
        company: str,
        *,
        country: str | None = None,
        name: str | None = None,
        source: str | None = None,
        fetch_json: FetchJson | None = None,
        timeout: float = 30.0,
        page_size: int = PAGE_SIZE,
        max_pages: int = MAX_PAGES,
    ) -> None:
        if not company:
            raise ValueError("company must be a non-empty string")
        self.identifier = company
        self.country = _normalize_country(country)
        self.source = source or f"smartrecruiters:{company}"
        # Display name for SQL table ``jobs``'s ``company`` column. Preferred
        # over the API's own name so a config can relabel a company; the API's
        # name is the fallback in parse_listing.
        self.company = name or company
        self._name = name
        self.timeout = timeout
        self.page_size = page_size
        self.max_pages = max_pages
        self._fetch_json = fetch_json

    @property
    def listing_url(self) -> str:
        """The endpoint this scraper reads, without query parameters."""
        return LISTING_URL.format(company=self.identifier)

    def page_url(self, offset: int) -> str:
        """The listing URL for one page of results."""
        params: dict[str, object] = {"limit": self.page_size, "offset": offset}
        if self.country:
            params["country"] = self.country
        return f"{self.listing_url}?{urlencode(params)}"

    def detail_url(self, posting_id: str) -> str:
        """Where this posting's full description would be fetched from."""
        return DETAIL_URL.format(company=self.identifier, posting=posting_id)

    def public_url(self, posting_id: str) -> str:
        """The human-facing URL for a posting."""
        return PUBLIC_URL.format(company=self.identifier, posting=posting_id)

    def fetch(self) -> list[Job]:
        """Fetch every public posting for this company, paging by ``offset``.

        Raises :class:`FetchError` if the first page cannot be read; a failure
        on a *later* page ends pagination with what was collected so far and
        reports it, since a partial board beats none. Per-item problems go
        through :meth:`report_error`.
        """
        jobs: list[Job] = []
        seen: set[str] = set()
        offset = 0
        total: int | None = None

        for page in range(self.max_pages):
            try:
                payload = self._get_page(offset)
            except FetchError:
                if page == 0:
                    raise
                self.report_error(
                    f"pagination stopped at offset {offset} after {len(jobs)} jobs"
                )
                break

            postings = payload.get("content")
            if not isinstance(postings, list):
                raise FetchError(
                    f"smartrecruiters company {self.identifier!r}:"
                    " response has no 'content' list"
                )
            if total is None:
                total = _as_int(payload.get("totalFound"))

            if not postings:
                break

            new = 0
            for index, posting in enumerate(postings):
                if not isinstance(posting, dict):
                    self.report_error(
                        f"offset {offset} item {index}: expected an object,"
                        f" got {type(posting).__name__}"
                    )
                    continue
                if not self._is_public(posting):
                    continue
                try:
                    job = self.parse_listing(posting)
                except ParseError as error:
                    self.report_error(f"offset {offset} item {index}: {error}")
                    continue
                # A company that ignores offset would otherwise loop forever.
                if job.external_id in seen:
                    continue
                seen.add(job.external_id)
                jobs.append(job)
                new += 1

            if total is not None and len(jobs) >= total:
                break
            if new == 0:
                # No progress: pagination has stalled, so stop rather than spin.
                break

            offset += self.page_size

        if total is not None and len(jobs) < total:
            self.report_error(f"collected {len(jobs)} of {total} postings")
        return jobs

    def parse_listing(self, posting: dict) -> Job:
        """Map one SmartRecruiters posting onto a :class:`Job`.

        Requires only ``id`` and ``name``. No description is set -- the listing
        does not carry one -- but ``description_url`` is, so the separate
        fetch step can find it later.
        """
        external_id = _required_str(posting, "id")
        title = _required_str(posting, "name")

        return Job(
            source=self.source,
            external_id=external_id,
            company=self._name or _nested_str(posting, "company", "name") or self.identifier,
            title=title,
            location=_location(posting),
            description=None,           # listings carry none; see module docstring
            url=self.public_url(external_id),
            description_url=self.detail_url(external_id),
            posted_at=_timestamp(posting.get("releasedDate")),
            raw_json=_raw(posting),
        )

    @staticmethod
    def _is_public(posting: dict) -> bool:
        """Honour ``visibility``: only PUBLIC postings are stored.

        Mirrors Ashby's treatment of ``isListed``. A posting without the field
        is kept, since the public endpoint normally omits non-public ones.
        """
        visibility = posting.get("visibility")
        if isinstance(visibility, str) and visibility.strip():
            return visibility.strip().upper() == "PUBLIC"
        return True

    def _get_page(self, offset: int) -> dict:
        """Request one page of listings."""
        url = self.page_url(offset)
        fetcher = self._fetch_json or _http_get_json
        try:
            payload = fetcher(url)
        except FetchError:
            raise
        except Exception as error:  # noqa: BLE001 - normalized for the runner
            raise FetchError(
                f"smartrecruiters company {self.identifier!r}: could not read"
                f" {url}: {error}"
            ) from error

        if not isinstance(payload, dict):
            raise FetchError(
                f"smartrecruiters company {self.identifier!r}: response was"
                f" {type(payload).__name__}, expected an object"
            )
        if "content" not in payload:
            raise FetchError(
                f"smartrecruiters company {self.identifier!r}:"
                " response has no 'content' list"
            )
        return payload


def fetch_description(*_args, **_kwargs):
    """Reserved: the separate, paced SmartRecruiters description fetch.

    Not implemented yet. The listing pass stores rows with ``description IS
    NULL`` and a populated ``description_url``; fetching the text from that URL
    is its own step, shaped like
    :mod:`jobecosystem.ingest.sources.workday_description` (one request per job,
    spaced, re-runnable, writing back to SQL table ``jobs``). Until it exists,
    descriptions are simply absent and :func:`parse_listing` is the whole
    scraper.
    """
    raise NotImplementedError(
        "SmartRecruiters description fetching is not implemented yet;"
        " listings are stored without descriptions"
    )


# ---------------------------------------------------------------------------
# company list
# ---------------------------------------------------------------------------

def resolve_companies_file(path: str | Path | None = None) -> Path:
    """Resolve the companies file: argument > ``SMARTRECRUITERS_COMPANIES_FILE``
    > ``smartrecruiters_companies.txt`` at the repo root."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("SMARTRECRUITERS_COMPANIES_FILE")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_COMPANIES_FILE


def parse_companies(text: str) -> ParseOutcome:
    """Parse companies-file text into specs, collecting anything unusable.

    Each line is ``identifier [country] [Display Name]``. ``country`` is
    recognised only when it is ``*`` or a two-letter code, so a bare
    ``identifier  Display Name`` line is not misread as a country; anything
    else after the identifier becomes the display name.

    Comments (``#``), blank lines, and duplicate identifiers are skipped rather
    than raising -- one typo must not stop the daily scrape. Duplicates keep the
    first occurrence, since the identifier already determines the source.
    """
    specs: list[CompanySpec] = []
    errors: list[tuple[int, str]] = []
    seen: set[str] = set()

    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        try:
            parts = shlex.split(line)
        except ValueError as error:  # unbalanced quotes
            errors.append((number, f"could not parse: {error}"))
            continue

        if not parts:
            continue

        identifier = parts[0]
        country: str | None = None
        rest = parts[1:]
        if rest and _looks_like_country(rest[0]):
            country = _normalize_country(rest[0])
            rest = rest[1:]
        name = " ".join(rest) or None

        if identifier in seen:
            continue
        seen.add(identifier)
        specs.append(CompanySpec(identifier=identifier, country=country, name=name))

    return ParseOutcome(specs=specs, errors=errors)


def load_companies(path: str | Path | None = None) -> ParseOutcome:
    """Load and parse the companies file.

    Returns an empty outcome when the file is missing, rather than raising: a
    missing optional config file means "no companies configured", which the
    caller reports clearly.
    """
    companies_file = resolve_companies_file(path)
    if not companies_file.exists():
        return ParseOutcome(specs=[], errors=[])
    return parse_companies(companies_file.read_text(encoding="utf-8"))


def build_scrapers(
    path: str | Path | None = None,
    *,
    fetch_json: FetchJson | None = None,
) -> list[Scraper]:
    """Build one :class:`SmartRecruitersScraper` per configured company.

    ``fetch_json`` is passed through for tests, which supply a stub instead of
    hitting the network.
    """
    return [
        SmartRecruitersScraper(
            spec.identifier,
            country=spec.country,
            name=spec.name,
            fetch_json=fetch_json,
        )
        for spec in load_companies(path).specs
    ]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _looks_like_country(token: str) -> bool:
    """True when a token should be read as a country filter, not a name."""
    return token == "*" or (len(token) == 2 and token.isalpha())


def _normalize_country(country: str | None) -> str | None:
    """Lowercase a country code; ``*`` and blanks mean "no filter"."""
    if country is None:
        return None
    token = country.strip()
    if not token or token == "*":
        return None
    return token.lower()


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

    if response.status_code == 404:
        raise FetchError(f"GET {url}: company not found (404)")
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


def _nested_str(posting: dict, outer: str, inner: str) -> str | None:
    """Trimmed string at ``posting[outer][inner]``, or ``None``."""
    nested = posting.get(outer)
    if not isinstance(nested, dict):
        return None
    value = nested.get(inner)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _location(posting: dict) -> str | None:
    """Location string, preferring the API's precomposed ``fullLocation``.

    Remote/hybrid flags are appended because the schema has a single
    ``location`` column and they are useful when scanning.
    """
    raw = posting.get("location")
    if not isinstance(raw, dict):
        return None

    full = raw.get("fullLocation")
    parts: list[str] = []
    if isinstance(full, str) and full.strip():
        parts.append(full.strip())
    else:
        composed = [raw.get("city"), raw.get("region"), raw.get("country")]
        joined = ", ".join(p.strip() for p in composed if isinstance(p, str) and p.strip())
        if joined:
            parts.append(joined)

    lowered = " ".join(parts).lower()
    if raw.get("remote") is True and "remote" not in lowered:
        parts.append("Remote")
    if raw.get("hybrid") is True and "hybrid" not in lowered:
        parts.append("Hybrid")

    return " / ".join(parts) if parts else None


def _timestamp(value: Any) -> str | None:
    """Normalize SmartRecruiters' timestamp to ``YYYY-MM-DDTHH:MM:SSZ``.

    ``releasedDate`` arrives as ``2026-10-02T15:22:56.691Z``; the fractional
    seconds and any offset are dropped so comparisons stay consistent with the
    other sources. Returns ``None`` when absent or unparseable.
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


def _as_int(value: Any) -> int | None:
    """Best-effort integer, for ``totalFound``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _raw(posting: dict) -> str | None:
    """The vendor payload, for debugging a parser that broke."""
    try:
        return json.dumps(posting, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
