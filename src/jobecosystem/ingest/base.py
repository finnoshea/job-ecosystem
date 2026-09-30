"""The scraper interface every source implements.

A source's job is narrow: fetch listings from one board and parse each one into
a :class:`~jobecosystem.core.models.Job` via :meth:`Scraper.parse_listing`.
It does not touch the database, decide repost semantics, or know about the
runner. That keeps vendor breakage contained -- a dead Workday tenant cannot
corrupt shared state or fail the Ashby scraper, because the only thing that
crosses the boundary is a list of jobs or an exception.

Scrapers are constructed with no arguments and are stateless between runs, so
the runner can build one per run without lifecycle management.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.models import Job


class ScraperError(Exception):
    """Base class for scraper failures that should be logged, not crash a run."""


class FetchError(ScraperError):
    """The board could not be reached, or returned an unusable response."""


class ParseError(ScraperError):
    """The board responded, but a listing could not be parsed into a Job."""


@dataclass(slots=True)
class ScrapeResult:
    """What one scraper returned from one execution.

    ``jobs`` are parsed and un-deduped; the database's unique constraint
    resolves duplicates. ``errors`` holds recoverable per-item failures -- a
    single unparseable listing out of two hundred should not discard the rest.
    Each entry is a ``(tenant, message)`` pair, where ``tenant`` is ``None``
    for sources without tenants::

        errors=[
            ("acme", "listing 41832: missing 'title' field"),
            ("globex", "HTTP 503 from board API"),
            (None, "response was not valid JSON"),
        ]

    ``duration_ms`` is measured, not reported, so it is filled by
    :meth:`Scraper.scrape` rather than by subclasses.
    """

    source: str
    jobs: list[Job] = field(default_factory=list)
    errors: list[tuple[str | None, str]] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        """True when the fetch produced jobs and no errors were recorded."""
        return not self.errors

    def __len__(self) -> int:
        return len(self.jobs)


class Scraper(ABC):
    """Base class for a single job board.

    Subclasses implement :meth:`parse_listing`, mapping one board listing into
    a :class:`Job`, and may override :meth:`source_name` when the class name is
    not the right label. Raising :class:`ScraperError` from ``fetch`` is
    expected and handled by the runner; other exceptions indicate a bug.
    """

    #: Stable identifier stored in ``jobs.source`` and ``scrape_runs.source``.
    #: Renaming this orphans existing rows, so treat it as a storage contract.
    source: str = ""

    def __init__(self) -> None:
        # Per-run error collector, drained by scrape(). Named with a leading
        # underscore and typed as a list so a subclass that forgets to call
        # super().__init__() still fails loudly rather than silently losing
        # errors.
        self._errors: list[tuple[str | None, str]] = []

    def report_error(self, message: str, *, tenant: str | None = None) -> None:
        """Record a recoverable failure for the current run.

        Subclasses call this instead of logging: :meth:`scrape` drains the
        collected errors into :attr:`ScrapeResult.errors`, which the runner
        writes to SQL table ``scrape_errors``. That keeps a single malformed
        listing visible without discarding the rest of the board.

        Creates the collector on demand, so a subclass that defines its own
        ``__init__`` without calling ``super().__init__()`` still works.
        """
        if not hasattr(self, "_errors"):
            self._errors = []
        self._errors.append((tenant, message))

    def _drain_errors(self) -> list[tuple[str | None, str]]:
        """Return and clear the errors collected during this run."""
        collected = getattr(self, "_errors", None)
        if not collected:
            return []
        drained = list(collected)
        collected.clear()
        return drained

    @property
    def source_name(self) -> str:
        """Label for this scraper, defaulting to the ``source`` attribute."""
        if not self.source:
            raise NotImplementedError(
                f"{type(self).__name__} must define a non-empty `source` attribute"
            )
        return self.source

    @abstractmethod
    def fetch(self) -> list[Job]:
        """Fetch and parse every listing this board currently exposes.

        Implementations should:

        * return :class:`Job` objects with ``source``, ``external_id``,
          ``company``, and ``title`` set, letting ``content_hash`` fill itself,
          via :meth:`parse_listing` for each raw listing;
        * leave ``first_seen_at``, ``last_seen_at``, ``repost_count``, and
          ``status`` alone -- ``ingest.upsert`` owns those;
        * raise :class:`FetchError` when the board is unreachable and
          :class:`ParseError` when a response cannot be understood, rather than
          returning a partial list silently.
        """

    def parse_listing(self, listing: dict) -> Job:
        """Map one vendor listing onto a :class:`Job`.

        Separated from :meth:`fetch` so the mapping is testable against a saved
        payload without network access, which is where most scraper breakage
        actually lives. ``listing`` is the raw per-job object as the board
        returned it -- not a whole response envelope.

        Raise :class:`ParseError` when a required field is missing or the wrong
        shape; that keeps one bad listing from discarding the rest of the run.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement parse_listing() or override fetch()"
        )

    def scrape(self) -> ScrapeResult:
        """Run :meth:`fetch` and wrap the outcome with timing and a source label.

        The runner calls this, not ``fetch``, so every run is measured the same
        way. Exceptions propagate: the runner records the failed run, and
        converting errors into empty results here would make a broken board
        indistinguishable from a board with no openings.

        Errors reported via :meth:`report_error` during ``fetch`` are drained
        into :attr:`ScrapeResult.errors`. They are cleared first so a scraper
        instance reused across runs does not accumulate stale ones.
        """
        self._errors = getattr(self, "_errors", [])
        self._errors.clear()

        started = time.monotonic()
        jobs = self.fetch()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return ScrapeResult(
            source=self.source_name,
            jobs=list(jobs),
            errors=self._drain_errors(),
            duration_ms=elapsed_ms,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} source={self.source_name!r}>"
