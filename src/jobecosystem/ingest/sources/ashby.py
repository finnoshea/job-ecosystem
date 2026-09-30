"""Ashby job-board scraper.

Ashby exposes a public, unauthenticated JSON board per company::

    GET https://api.ashbyhq.com/posting-api/job-board/{board}

The response is ``{"apiVersion": ..., "jobs": [...]}`` where each job is a flat
object with ``id``, ``title``, ``location``, ``descriptionPlain``,
``publishedAt``, and friends. No pagination, no auth, no rate limiting worth
worrying about at daily cadence.

Known gaps, deliberate rather than overlooked:

* **No structured salary.** Ashby returns no compensation field; rates appear
  only as prose inside the description (e.g. "The expected base salary ranges
  are:"). ``salary_min``/``salary_max`` are therefore left unset. Parsing numbers
  out of descriptions would risk inventing data, so it is not attempted.
* **One board per instance.** A company with multiple Ashby boards needs one
  scraper per board; the ``source`` label is the board name, so the rows stay
  distinguishable.
* **``isListed``** is honoured: unlisted postings are skipped rather than
  stored, since they are not publicly visible.

The HTTP dependency is injected (``fetch_json``) so the parser is testable
against a saved payload with no network access.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

from ...core.models import Job
from ..base import FetchError, ParseError, Scraper

#: Ashby's public posting API, per board.
BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{board}"

#: Signature of the injectable HTTP call: takes a URL, returns decoded JSON.
FetchJson = Callable[[str], Any]

#: Ashby returns ISO-8601 with an offset ("2026-04-07T17:12:35.753+00:00") or
#: "Z". Our columns store "YYYY-MM-DDTHH:MM:SSZ", so the offset and fractional
#: seconds are normalized rather than stored as-is -- mixed formats would break
#: the lexicographic comparisons the recency views rely on.


class AshbyScraper(Scraper):
    """Scrape one Ashby board.

    ``board`` is the company's Ashby board slug (the segment after
    ``jobs.ashbyhq.com/``). ``board`` doubles as the ``source`` label unless
    ``source`` is passed explicitly, which is what you want when one company has
    several boards.
    """

    def __init__(
        self,
        board: str,
        *,
        source: str | None = None,
        company: str | None = None,
        fetch_json: FetchJson | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not board:
            raise ValueError("board must be a non-empty string")
        self.board = board
        self.source = source or f"ashby:{board}"
        # Display name for SQL table ``jobs``'s ``company`` column. Boards omit
        # it, so the slug is the fallback.
        self.company = company or board
        self.timeout = timeout
        self._fetch_json = fetch_json

    @property
    def board_url(self) -> str:
        """The full URL this scraper reads."""
        return BOARD_URL.format(board=self.board)

    def fetch(self) -> list[Job]:
        """Fetch every listed posting on the board.

        Raises :class:`FetchError` if the board itself cannot be read. Per-
        listing problems are reported through :meth:`report_error` instead, so
        one malformed job does not discard the rest of the board; the runner
        writes them to SQL table ``scrape_errors``.
        """
        payload = self._get_json()
        raw_jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(raw_jobs, list):
            raise FetchError(
                f"ashby board {self.board!r}: response has no 'jobs' list"
            )

        jobs: list[Job] = []
        for index, listing in enumerate(raw_jobs):
            if not isinstance(listing, dict):
                self.report_error(
                    f"listing {index}: expected an object, got {type(listing).__name__}"
                )
                continue
            if listing.get("isListed") is False:
                # Present in the API but not publicly visible.
                continue
            try:
                jobs.append(self.parse_listing(listing))
            except ParseError as error:
                # Identified by Ashby's own id when present, since that is what
                # makes the failure actionable. The board is already implied by
                # the run row, so the message omits it.
                identifier = listing.get("id")
                label = f"job {identifier}" if isinstance(identifier, str) else f"listing {index}"
                self.report_error(f"{label}: {error}")
        return jobs

    def parse_listing(self, listing: dict) -> Job:
        """Map one Ashby job object onto a :class:`Job`.

        Raises :class:`ParseError` when ``id`` or ``title`` is missing, since a
        row without either is not identifiable.
        """
        external_id = _required_str(listing, "id")
        title = _required_str(listing, "title")

        return Job(
            source=self.source,
            external_id=external_id,
            company=self.company,
            title=title.strip(),
            location=_location(listing),
            description=_description(listing),
            url=_first_str(listing, "jobUrl", "applyUrl"),
            posted_at=_timestamp(listing.get("publishedAt")),
            # salary_min/salary_max intentionally unset: Ashby has no structured
            # compensation field. See the module docstring.
            raw_json=_raw(listing),
        )

    def parse_payload(self, payload: Any) -> list[Job]:
        """Parse a full board response, for replaying a saved payload in tests."""
        raw_jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(raw_jobs, list):
            raise FetchError(
                f"ashby board {self.board!r}: response has no 'jobs' list"
            )
        return [
            self.parse_listing(listing)
            for listing in raw_jobs
            if isinstance(listing, dict) and listing.get("isListed") is not False
        ]

    def _get_json(self) -> Any:
        fetcher = self._fetch_json or _http_get_json
        try:
            return fetcher(self.board_url)
        except FetchError:
            raise
        except Exception as error:  # noqa: BLE001 - normalized for the runner
            raise FetchError(
                f"ashby board {self.board!r}: could not read {self.board_url}: {error}"
            ) from error


def _http_get_json(url: str, *, timeout: float = 30.0) -> Any:
    """Default fetcher: a plain GET returning decoded JSON.

    Imported lazily so the parser can be used, and tested, without httpx.
    """
    import httpx

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
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


def _required_str(listing: dict, key: str) -> str:
    """Read a required non-empty string field, or raise :class:`ParseError`.

    Board-agnostic on purpose: the caller adds the job identifier to the message.
    """
    value = listing.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParseError(f"missing {key!r}")
    return value.strip()


def _first_str(listing: dict, *keys: str) -> str | None:
    """First present, non-empty string among ``keys``."""
    for key in keys:
        value = listing.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _location(listing: dict) -> str | None:
    """Primary location, annotated with remote/hybrid and secondary sites.

    Ashby splits this across ``location``, ``isRemote``, ``workplaceType``, and
    ``secondaryLocations``. They are combined into one column because the schema
    has a single ``location`` field and the extras are useful when scanning.
    """
    primary = listing.get("location")
    parts: list[str] = []
    if isinstance(primary, str) and primary.strip():
        parts.append(primary.strip())

    workplace = listing.get("workplaceType")
    if listing.get("isRemote") is True and "remote" not in " ".join(parts).lower():
        parts.append("Remote")
    elif isinstance(workplace, str) and workplace.strip() and workplace.strip().lower() != "onsite":
        parts.append(workplace.strip())

    for site in _secondary_locations(listing):
        if site not in parts:
            parts.append(site)

    return " / ".join(parts) if parts else None


def _secondary_locations(listing: dict) -> Iterator[str]:
    """Secondary site names, ignoring malformed entries."""
    secondary = listing.get("secondaryLocations")
    if not isinstance(secondary, list):
        return
    for entry in secondary:
        name = entry.get("location") if isinstance(entry, dict) else None
        if isinstance(name, str) and name.strip():
            yield name.strip()


def _description(listing: dict) -> str | None:
    """Plain-text description, which is what gets embedded and keyword-matched.

    Prefers ``descriptionPlain`` over ``descriptionHtml`` because stripping HTML
    reliably is more work than it is worth, and the plain field is provided.
    """
    plain = listing.get("descriptionPlain")
    if isinstance(plain, str) and plain.strip():
        return plain.strip()
    return None


def _timestamp(value: Any) -> str | None:
    """Normalize an Ashby timestamp to our ``YYYY-MM-DDTHH:MM:SSZ`` format.

    Returns ``None`` when the value is absent or unparseable: ``posted_at`` is
    nullable precisely because boards are unreliable about it.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip().replace("Z", "+00:00")
    try:
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _raw(listing: dict) -> str | None:
    """The vendor payload, for debugging a parser that broke."""
    try:
        return json.dumps(listing, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
