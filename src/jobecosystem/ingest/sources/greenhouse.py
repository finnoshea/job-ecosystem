"""Greenhouse job-board scraper.

Greenhouse exposes a public, unauthenticated JSON API per board::

    GET https://boards-api.greenhouse.io/v1/boards/{board}/jobs

The response is ``{"jobs": [...], "meta": {"total": N}}``. The listing carries
metadata only -- ``id``, ``title``, ``location``, ``requisition_id``,
``first_published``, ``absolute_url`` -- and **no description text**. The full
text lives behind a second request per job::

    GET https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{id}
        -> adds ``content`` (HTML-escaped job description)

Deliberate choices, mirroring :mod:`jobecosystem.ingest.sources.smartrecruiters`:

* **No description on the listing pass.** The board endpoint *can* be asked to
  inline descriptions with ``?content=true``, but doing so for every job in one
  response is exactly the burst the paced fetch exists to avoid, and it makes
  the listing page huge. Rows are stored description-less (``content_hash``
  NULL) with ``description_url`` populated; the text is fetched later by the
  shared, paced :mod:`jobecosystem.ingest.sources.description`, which calls
  :func:`parse_description` here.
* **One board per scraper.** The board slug is part of the URL, so a company
  with two boards needs two entries (and the ``source`` label is the slug,
  keeping rows distinguishable).
* **No pagination.** ``/v1/boards/{board}/jobs`` returns the whole board and
  reports the count in ``meta.total``; there is no offset to advance. A board
  that returns fewer jobs than ``meta.total`` is reported rather than silently
  truncated.

The list of boards is read from ``greenhouse_companies.txt`` at the repo root;
see that file for how to find a board slug.

The HTTP dependency is injected (``fetch_json``) so the parser is testable
against a saved payload with no network access.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...core.models import Job
from ..base import FetchError, ParseError, Scraper
from . import companies
from .description import DescriptionError, html_to_text

#: Greenhouse's public posting list, per board.
LISTING_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"

#: The per-job detail endpoint, where the description actually lives.
DETAIL_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job}"

#: Signature of the injectable HTTP call: URL -> decoded JSON.
FetchJson = Callable[[str], Any]

#: src/jobecosystem/ingest/sources/greenhouse.py
#:   -> sources -> ingest -> jobecosystem -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_COMPANIES_FILE = _REPO_ROOT / "greenhouse_companies.txt"


@dataclass(frozen=True, slots=True)
class BoardSpec:
    """One configured board: the slug, plus an optional display name."""

    board: str
    name: str | None = None

    @property
    def source(self) -> str:
        """The ``jobs.source`` label: ``greenhouse:<board>``."""
        return f"greenhouse:{self.board}"


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """What was understood from a companies file, and what was not."""

    specs: list[BoardSpec]
    errors: list[tuple[int, str]]

    def __bool__(self) -> bool:
        return bool(self.specs)

    def __len__(self) -> int:
        return len(self.specs)


class GreenhouseScraper(Scraper):
    """Scrape one Greenhouse board.

    ``board`` is the company's Greenhouse board slug (the segment in
    ``boards.greenhouse.io/{board}``). It doubles as the ``source`` label and as
    the display name unless ``name`` is given.
    """

    def __init__(
        self,
        board: str,
        *,
        name: str | None = None,
        source: str | None = None,
        fetch_json: FetchJson | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not board:
            raise ValueError("board must be a non-empty string")
        self.board = board
        self.source = source or f"greenhouse:{board}"
        # Display name for SQL table ``jobs``'s ``company`` column. The board
        # listing returns one; ``self.company`` is the config fallback, and
        # parse_listing prefers the API's name when no config name was given.
        self.company = name or board
        self._name = name
        self.timeout = timeout
        self._fetch_json = fetch_json

    @property
    def listing_url(self) -> str:
        """The endpoint this scraper reads."""
        return LISTING_URL.format(board=self.board)

    def detail_url(self, job_id: str) -> str:
        """Where this posting's full description would be fetched from."""
        return DETAIL_URL.format(board=self.board, job=job_id)

    def fetch(self) -> list[Job]:
        """Fetch every posting on the board.

        Raises :class:`FetchError` if the board cannot be read. Per-listing
        problems are reported through :meth:`report_error` instead, so one
        malformed job does not discard the rest.
        """
        payload = self._get_json()
        postings = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(postings, list):
            raise FetchError(
                f"greenhouse board {self.board!r}: response has no 'jobs' list"
            )

        jobs: list[Job] = []
        seen: set[str] = set()
        for index, posting in enumerate(postings):
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

        total = _as_int((payload.get("meta") or {}).get("total")
                        if isinstance(payload.get("meta"), dict) else None)
        if total is not None and len(jobs) < total:
            self.report_error(f"collected {len(jobs)} of {total} postings")
        return jobs

    def parse_listing(self, posting: dict) -> Job:
        """Map one Greenhouse posting onto a :class:`Job`.

        Requires only ``id`` and ``title``. No description is set -- the listing
        does not carry one -- but ``description_url`` is, so the separate fetch
        step can find it later.
        """
        posting_id = _required(posting, "id")
        title = _required_str(posting, "title")

        return Job(
            source=self.source,
            external_id=posting_id,
            company=self._name or _clean(posting.get("company_name")) or self.board,
            title=title,
            location=_location(posting),
            description=None,           # listings carry none; see module docstring
            url=_clean(posting.get("absolute_url")),
            description_url=self.detail_url(posting_id),
            posted_at=_timestamp(posting.get("first_published")),
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
                f"greenhouse board {self.board!r}: could not read"
                f" {self.listing_url}: {error}"
            ) from error


def parse_description(payload: Any) -> str:
    """Extract plain text from a Greenhouse job-detail response.

    The detail endpoint adds a ``content`` field, but it is HTML-escaped HTML
    (``&lt;p&gt;`` rather than ``<p>``), so it is unescaped once before the
    shared converter strips the tags. Raises :class:`DescriptionError` when
    there is no usable text, so a silent schema change surfaces as a failure
    rather than an empty row.
    """
    if not isinstance(payload, dict):
        raise DescriptionError(
            f"detail response was {type(payload).__name__}, expected an object"
        )

    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise DescriptionError("'content' is missing or empty")

    text = html_to_text(html.unescape(content))
    if not text:
        # Present but nothing but markup: treat as a failure, not a description.
        raise DescriptionError("description contained no text after HTML stripping")
    return text


# ---------------------------------------------------------------------------
# board list
# ---------------------------------------------------------------------------

def resolve_companies_file(path: str | Path | None = None) -> Path:
    """Resolve the companies file: argument > ``GREENHOUSE_BOARDS_FILE`` >
    ``greenhouse_companies.txt`` at the repo root."""
    return companies.resolve_file(
        path, env_var="GREENHOUSE_BOARDS_FILE", default=DEFAULT_COMPANIES_FILE
    )


def _specific(outcome: companies.ParseOutcome) -> ParseOutcome:
    """Map the shared tokenized entries onto Greenhouse board specs."""
    return ParseOutcome(
        specs=[
            BoardSpec(board=entry.slug, name=entry.name)
            for entry in outcome.entries
        ],
        errors=outcome.errors,
    )


def parse_boards(text: str) -> ParseOutcome:
    """Parse companies-file text into specs, collecting anything unusable.

    Each line is ``board [Display Name]``. Comments (``#``), blank lines, and
    duplicate boards are skipped rather than raising -- one typo must not stop
    the daily scrape. Duplicates keep the first occurrence so an earlier line
    with a display name is not overwritten by a later bare duplicate.

    The tokenizing itself is shared with the other per-board sources; see
    :mod:`jobecosystem.ingest.sources.companies`.
    """
    return _specific(companies.parse_lines(text))


def load_companies(path: str | Path | None = None) -> ParseOutcome:
    """Load and parse the companies file.

    Returns an empty outcome when the file is missing, rather than raising: a
    missing optional config file means "no boards configured", which the caller
    reports clearly.
    """
    return _specific(companies.load_lines(
        path, env_var="GREENHOUSE_BOARDS_FILE", default=DEFAULT_COMPANIES_FILE
    ))


def build_scrapers(
    path: str | Path | None = None,
    *,
    fetch_json: FetchJson | None = None,
) -> list[Scraper]:
    """Build one :class:`GreenhouseScraper` per configured board.

    ``fetch_json`` is passed through for tests, which supply a stub instead of
    hitting the network.
    """
    return [
        GreenhouseScraper(spec.board, name=spec.name, fetch_json=fetch_json)
        for spec in load_companies(path).specs
    ]


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

    if response.status_code == 404:
        raise FetchError(f"GET {url}: board not found (404)")
    if response.status_code >= 400:
        raise FetchError(f"GET {url}: HTTP {response.status_code}")

    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise FetchError(f"GET {url}: response was not JSON: {error}") from error


def _required(posting: dict, key: str) -> str:
    """Read a required identifier, accepting a number or a string.

    Greenhouse ids are integers, but a saved payload or a future API change
    could make them strings; both normalize to the same external id.
    """
    value = posting.get(key)
    if isinstance(value, bool) or value is None:
        raise ParseError(f"missing {key!r}")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ParseError(f"missing {key!r}")


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
    """Location string from the nested ``location.name`` Greenhouse returns."""
    location = posting.get("location")
    if isinstance(location, dict):
        return _clean(location.get("name"))
    return _clean(location)


def _timestamp(value: Any) -> str | None:
    """Normalize a Greenhouse timestamp to ``YYYY-MM-DDTHH:MM:SSZ``.

    ``first_published`` arrives as ``2026-08-11T19:32:34-04:00``; the offset is
    converted to UTC and fractional seconds dropped so comparisons stay
    consistent with the other sources. Returns ``None`` when absent or
    unparseable.
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
    """Best-effort integer, for ``meta.total``."""
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
