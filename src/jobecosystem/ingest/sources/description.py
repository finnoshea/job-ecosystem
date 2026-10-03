"""Fetching job descriptions, one job at a time, for every source.

A board listing rarely carries a job's full text -- it lives behind a second
request per job, with a vendor-specific payload shape. At thousands of openings,
fetching every description during a scrape would mean thousands of requests in a
burst, so it is deliberately a separate, paced, re-runnable operation.

This module is the shared owner of that operation and is callable from anywhere::

    fetch_description(conn, job_id)          # on demand, e.g. from the TUI
    fetch_descriptions(conn, limit=200)      # a paced batch, e.g. from cron

Both write the description and its bookkeeping to SQL table ``jobs``, so every
caller gets consistent state. Nothing here is imported by a scraper, so the TUI
can use it without pulling in the scraping code.

Per-source parsing
------------------

The shared half -- fetch, hash, store, pace, resume -- is here; the *shape* is
one parser per source, chosen by the prefix of the ``jobs.source`` value:

    ``workday:*``          ``jobPostingInfo.jobDescription``   (HTML)
    ``smartrecruiters:*``  ``jobAd.sections.*.text``           (HTML)
    ``greenhouse:*``       ``content``                         (escaped HTML)

Ashby needs no parser at all: its board listing already carries
``descriptionPlain``, so an Ashby row has no ``description_url`` to fetch.

Each parser lives next to its scraper and is imported lazily by
:func:`parse_description`, so adding a source means adding one function, not
editing the fetcher. A source with no parser is reported, not guessed at.

Only once
---------

A job with a description is never fetched again; ``force=True`` overrides that.
The write is conditional on the row still being unfetched, so a TUI request
racing a paced batch cannot double-write.

Pacing
------

Requests are spaced by a jittered delay (``delay`` seconds, plus up to
``jitter``) because request *rate*, not volume, is what gets a client blocked.
A batch that would trip a rate limit is better split across runs, which is what
``limit`` and ``jobs_needing_descriptions`` are for: the work queue is
persistent, so an interrupted run resumes rather than restarting.
"""

from __future__ import annotations

import html
import json
import random
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...core import db as core_db
from ...core.models import content_hash as compute_content_hash
from ..base import FetchError

#: Signature of the injectable HTTP call: URL -> decoded JSON. POSTs are not
#: needed here, so a plain GET is enough.
FetchJson = Callable[[str], Any]

#: Seconds between requests by default. Nightly cadence makes patience free.
DEFAULT_DELAY = 0.75
DEFAULT_JITTER = 0.25

#: Reused across sources; these are HTML's tag syntax, not vendor-specific.
_TAG = re.compile(r"<[^>]+>")
_BLOCK_BREAK = re.compile(r"</(p|div|li|h[1-6]|tr|section|article)>", re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RUN = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")

#: Source prefix -> module holding that source's ``parse_description``.
_PARSER_MODULES = {
    "workday": "workday_description",
    "smartrecruiters": "smartrecruiters",
    "greenhouse": "greenhouse",
}


class DescriptionError(FetchError):
    """A description could not be retrieved or understood."""


@dataclass(slots=True)
class DescriptionResult:
    """Outcome of one description fetch.

    ``written`` is False when the row already had a description and ``force``
    was not set -- the ``skipped`` case, which is normal rather than an error.
    """

    job_id: int
    fetched: bool = False
    written: bool = False
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when nothing went wrong, whether or not work was done."""
        return self.error is None

    def __len__(self) -> int:
        return 1 if self.written else 0


@dataclass(slots=True)
class BatchSummary:
    """Outcome of a paced batch."""

    attempted: int = 0
    written: int = 0
    skipped: int = 0
    failed: list[tuple[int, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no fetch failed."""
        return not self.failed

    def __len__(self) -> int:
        return self.attempted


@dataclass(slots=True)
class FetchedDescription:
    """A description retrieved from the wire, not yet stored.

    The split between fetching and storing exists so a caller on a worker thread
    can do the HTTP call without touching the database: SQLite objects belong to
    the thread that created them, so only the fetch runs off-thread and the write
    is handed back. Carries everything the write needs, so storing does not have
    to re-read the row.
    """

    job_id: int
    text: str
    content_hash: str


@dataclass(slots=True)
class FetchOutcome:
    """Result of the read-only half: either a description, or why there isn't one.

    ``skipped`` and ``error`` mirror :class:`DescriptionResult` so a caller that
    split the two steps can still produce the same shape at the end.
    """

    job_id: int
    fetched: FetchedDescription | None = None
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when nothing went wrong, whether or not work was done."""
        return self.error is None


# ---------------------------------------------------------------------------
# per-source dispatch
# ---------------------------------------------------------------------------

def source_family(source: str) -> str:
    """The prefix of a ``jobs.source`` value, e.g. ``workday`` for
    ``workday:asml:Site``."""
    return source.split(":", 1)[0]


def parse_description(payload: Any, source: str) -> str:
    """Extract plain text from a detail response, using the source's parser.

    Raises :class:`DescriptionError` when the payload cannot be understood, so a
    silent schema change at the vendor surfaces as a failure rather than an
    empty row.
    """
    return _parser_for(source)(payload)


def _parser_for(source: str) -> Callable[[Any], str]:
    """The per-source parser for a ``jobs.source`` value.

    Imported lazily so this module does not import every scraper at load time,
    which would also make the TUI pull in scraping code it does not need.
    """
    family = source_family(source)
    module_name = _PARSER_MODULES.get(family)
    if module_name is None:
        raise DescriptionError(
            f"no description parser for source {source!r}"
            f" (family {family!r})"
        )
    module = __import__(f"{__package__}.{module_name}", fromlist=["parse_description"])
    parser = getattr(module, "parse_description", None)
    if not callable(parser):
        raise DescriptionError(
            f"source module {module.__name__!r} has no parse_description()"
        )
    return parser


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def fetch_description(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    force: bool = False,
    fetch_json: FetchJson | None = None,
) -> DescriptionResult:
    """Fetch and store the description for one job.

    Writes ``description``, ``description_fetched_at``, and ``content_hash`` on
    SQL table ``jobs``. Returns without fetching when the row already has a
    description, unless ``force`` is set.

    Composes :func:`fetch_description_from_url` and :func:`store_description`,
    both of which are usable on their own -- which is what the TUI does, so its
    network call can run on a worker thread while the write stays on the thread
    that owns the connection.

    Raises nothing for expected failures: an unreachable endpoint, a missing
    URL, or an unrecognised payload all come back as
    :attr:`DescriptionResult.error`, since callers are a TUI and a batch job and
    neither wants an exception for "that one job is gone".
    """
    outcome = fetch_description_from_url(
        conn, job_id, force=force, fetch_json=fetch_json
    )
    if outcome.fetched is None:
        return DescriptionResult(
            job_id=job_id, skipped=outcome.skipped, error=outcome.error
        )

    written = store_description(conn, outcome.fetched, force=force)
    return DescriptionResult(
        job_id=job_id, fetched=True, written=written, skipped=not written
    )


def record_attempt(conn: sqlite3.Connection, job_id: int) -> None:
    """Count one description attempt, committed before the request is made.

    Incremented ahead of the network call so an interrupted or crashed process
    still consumes the try, and the queue's retry cap cannot loop forever on a
    dead posting. The TUI calls this too, on the thread that owns the
    connection, since it splits the fetch from the store.
    """
    with conn:
        conn.execute(
            "UPDATE jobs SET description_attempts = description_attempts + 1"
            " WHERE id = ?",
            (job_id,),
        )


def fetch_description_from_url(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    force: bool = False,
    fetch_json: FetchJson | None = None,
) -> FetchOutcome:
    """Look up what is needed, count the attempt, fetch, and compute the hash.

    Records the attempt counter (see :func:`record_attempt`) but does not store
    the description itself -- that is :func:`store_description`, so a caller can
    keep the fetch on a worker thread. Returns an outcome rather than raising,
    so a caller never has to catch transport errors. The row's ``source``
    selects the parser.
    """
    row = conn.execute(
        "SELECT id, source, company, title, description_url, content_hash,"
        " description FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return FetchOutcome(job_id=job_id, error=f"no job with id {job_id}")

    if not force and row["content_hash"] is not None:
        return FetchOutcome(job_id=job_id, skipped=True)

    # Counted before the request, so a failed or interrupted fetch still uses up
    # one of the row's tries. A row with no detail URL counts too; otherwise it
    # would sit in the queue forever, never able to make a request.
    record_attempt(conn, job_id)

    url = row["description_url"]
    if not url:
        return FetchOutcome(
            job_id=job_id,
            error=(
                f"job {job_id} has no description_url; it was stored from a"
                " listing that carries no description and no detail link"
            ),
        )

    return fetch_description_from_known_url(
        job_id, row["title"], row["company"], url,
        source=row["source"], fetch_json=fetch_json,
    )


def fetch_description_from_known_url(
    job_id: int,
    title: str,
    company: str,
    url: str,
    *,
    source: str,
    fetch_json: FetchJson | None = None,
) -> FetchOutcome:
    """Fetch and parse a description without touching the database at all.

    This is the piece a worker thread should call: no connection, so no thread
    affinity to violate. The hash is computed here because it needs the title and
    company, which the caller already has. ``source`` selects the parser.
    """
    fetcher = fetch_json or _http_get_json
    try:
        payload = fetcher(url)
        text = parse_description(payload, source)
    except FetchError as error:
        return FetchOutcome(job_id=job_id, error=str(error))
    except Exception as error:  # noqa: BLE001 - surfaced as a result, not raised
        return FetchOutcome(job_id=job_id, error=f"{type(error).__name__}: {error}")

    return FetchOutcome(
        job_id=job_id,
        fetched=FetchedDescription(
            job_id=job_id,
            text=text,
            content_hash=compute_content_hash(title, company, text),
        ),
    )


def store_description(
    conn: sqlite3.Connection,
    fetched: FetchedDescription,
    *,
    force: bool = False,
) -> bool:
    """Write a fetched description. Returns False if another writer won the race.

    The write half, and the only part that must run on the thread that owns the
    connection.
    """
    return _store(
        conn, fetched.job_id, fetched.text, fetched.content_hash, force=force
    )


def fetch_descriptions(
    conn: sqlite3.Connection,
    *,
    limit: int = 200,
    delay: float = DEFAULT_DELAY,
    jitter: float = DEFAULT_JITTER,
    force: bool = False,
    fetch_json: FetchJson | None = None,
    sleep: Callable[[float], None] = time.sleep,
    job_ids: Sequence[int] | None = None,
) -> BatchSummary:
    """Fetch descriptions for up to ``limit`` jobs, paced.

    Uses SQL view ``jobs_needing_descriptions`` unless ``job_ids`` is given, so
    an interrupted batch simply resumes: completed rows leave the queue by
    gaining a ``content_hash``, and rows that have hit the retry cap leave it by
    being exhausted. ``force`` ignores that cap. Each row is parsed by its own
    source's parser. Sleeping is injected so tests do not wait.
    """
    if job_ids is not None:
        ids = list(job_ids)[:limit]
    elif force:
        # The view bakes in the retry cap, so --force cannot read it. Same
        # conditions minus the attempts filter.
        ids = [
            row[0] for row in conn.execute(
                "SELECT id FROM jobs"
                " WHERE content_hash IS NULL AND status <> 'hidden'"
                " ORDER BY status = 'new' DESC, first_seen_at DESC LIMIT ?",
                (limit,),
            )
        ]
    else:
        ids = [
            row[0] for row in conn.execute(
                "SELECT id FROM jobs_needing_descriptions LIMIT ?", (limit,)
            )
        ]

    summary = BatchSummary()
    for index, job_id in enumerate(ids):
        if index:
            sleep(delay + random.uniform(0, jitter))
        summary.attempted += 1
        result = fetch_description(
            conn, job_id, force=force, fetch_json=fetch_json
        )
        if result.error is not None:
            summary.failed.append((job_id, result.error))
        elif result.written:
            summary.written += 1
        else:
            summary.skipped += 1
    return summary


def _store(
    conn: sqlite3.Connection,
    job_id: int,
    text: str,
    digest: str,
    *,
    force: bool,
) -> bool:
    """Write the description, returning False if another writer won the race.

    The ``WHERE`` clause is what makes "only once" hold under concurrent
    callers: a TUI fetch and a batch both target the same row, and the
    conditional update means whichever runs second matches no rows.
    """
    guard = "" if force else " AND content_hash IS NULL"
    with conn:
        cursor = conn.execute(
            "UPDATE jobs SET description = ?, content_hash = ?,"
            " description_fetched_at = ?"
            f" WHERE id = ?{guard}",
            (text, digest, core_db.utcnow(), job_id),
        )
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# shared parsing helpers
# ---------------------------------------------------------------------------

def html_to_text(markup: str) -> str:
    """Convert a description's HTML to readable plain text.

    Block-level closing tags become newlines and ``<br>`` becomes a newline, so
    paragraph structure survives; everything else is dropped. Entities are
    unescaped and whitespace is collapsed, since vendor HTML is machine
    generated and heavy with non-breaking spaces and stray indentation.
    """
    text = _SCRIPT.sub(" ", markup)
    text = _BR.sub("\n", text)
    text = _BLOCK_BREAK.sub("\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")

    lines = [_WHITESPACE_RUN.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _http_get_json(url: str, *, timeout: float = 30.0) -> Any:
    """Default fetcher: a GET returning decoded JSON, errors normalized.

    Failures name the status code and Content-Type, so a non-JSON body can be
    told apart after the fact: a 200 with ``text/html`` is a proxy/WAF page,
    while a 429 is a real rate limit. Without those, both look like an
    indistinguishable "response was not JSON".
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
        raise DescriptionError(f"GET {url} failed: {error}") from error

    content_type = response.headers.get("content-type") or "unknown"
    if response.status_code >= 400:
        raise DescriptionError(
            f"GET {url}: HTTP {response.status_code} ({content_type})"
        )
    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise DescriptionError(
            f"GET {url}: HTTP {response.status_code} ({content_type}):"
            f" response was not JSON: {error}"
        ) from error
