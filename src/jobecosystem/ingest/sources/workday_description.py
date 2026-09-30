"""Fetching Workday job descriptions, one job at a time.

Workday's listing endpoint returns only a title, a location, and an id -- the
full text lives behind a second request per job. At 20 jobs per page and
thousands of openings, fetching every description during a scrape would mean
thousands of requests in a burst, so it is deliberately a separate, paced,
re-runnable operation.

This module is the single owner of that request, and is callable from anywhere::

    fetch_description(conn, job_id)          # on demand, e.g. from the TUI
    fetch_descriptions(conn, limit=200)      # a paced batch, e.g. from cron

Both write the description and its bookkeeping to SQL table ``jobs``, so every
caller gets consistent state. Nothing here is imported by the scraper, so the
TUI can use it without pulling in the scraping code.

Only once
---------

A job with a description is never fetched again; ``force=True`` overrides that.
The write is conditional on the row still being unfetched, so a TUI request
racing a cron batch cannot double-write.

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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...core import db as core_db
from ...core.models import content_hash as compute_content_hash
from ..base import FetchError

#: Workday's job-detail endpoint. ``external_path`` is the value of the
#: listing's ``externalPath`` field, leading slash included.
DETAIL_URL = "https://{host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{external_path}"

#: Signature of the injectable HTTP call: URL -> decoded JSON. POSTs are not
#: needed here, so a plain GET is enough.
FetchJson = Callable[[str], Any]

#: Seconds between requests by default. Nightly cadence makes patience free.
DEFAULT_DELAY = 0.75
DEFAULT_JITTER = 0.25

#: Reused across calls; the pattern is HTML's tag syntax, not vendor-specific.
_TAG = re.compile(r"<[^>]+>")
_BLOCK_BREAK = re.compile(r"</(p|div|li|h[1-6]|tr|section|article)>", re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RUN = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


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

    Raises nothing for expected failures -- an unreachable endpoint, a missing
    URL, or an unrecognised payload all come back as
    :attr:`DescriptionResult.error`, since callers are a TUI and a cron job and
    neither wants an exception for "that one job is gone".
    """
    row = conn.execute(
        "SELECT id, source, company, title, description_url, content_hash, description"
        " FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return DescriptionResult(job_id=job_id, error=f"no job with id {job_id}")

    if not force and row["content_hash"] is not None:
        return DescriptionResult(job_id=job_id, skipped=True)

    url = row["description_url"]
    if not url:
        return DescriptionResult(
            job_id=job_id,
            error=(
                f"job {job_id} has no description_url; it was stored from a"
                " listing that carries no description and no detail link"
            ),
        )

    fetcher = fetch_json or _http_get_json
    try:
        payload = fetcher(url)
        text = parse_description(payload)
    except FetchError as error:
        return DescriptionResult(job_id=job_id, error=str(error))
    except Exception as error:  # noqa: BLE001 - surfaced as a result, not raised
        return DescriptionResult(
            job_id=job_id, error=f"{type(error).__name__}: {error}"
        )

    digest = compute_content_hash(row["title"], row["company"], text)
    if _store(conn, job_id, text, digest, force=force):
        return DescriptionResult(job_id=job_id, fetched=True, written=True)

    # Another writer got there first (the conditional UPDATE matched no rows).
    return DescriptionResult(job_id=job_id, fetched=True, skipped=True)


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
    gaining a ``content_hash``. Sleeping is injected so tests do not wait.
    """
    ids = list(job_ids) if job_ids is not None else [
        row[0] for row in conn.execute(
            "SELECT id FROM jobs_needing_descriptions LIMIT ?", (limit,)
        )
    ]
    if job_ids is not None:
        ids = ids[:limit]

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
    callers: a TUI fetch and a cron batch both target the same row, and the
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
# parsing
# ---------------------------------------------------------------------------

def parse_description(payload: Any) -> str:
    """Extract plain text from a Workday job-detail response.

    The description is HTML; this converts it to text rather than storing tags,
    because the column feeds embeddings and keyword matching. Raises
    :class:`DescriptionError` when the response has no usable text, so a silent
    schema change at the vendor surfaces as a failure instead of empty rows.
    """
    if not isinstance(payload, dict):
        raise DescriptionError(
            f"detail response was {type(payload).__name__}, expected an object"
        )

    info = payload.get("jobPostingInfo")
    if not isinstance(info, dict):
        raise DescriptionError("detail response has no 'jobPostingInfo' object")

    raw = info.get("jobDescription")
    if not isinstance(raw, str) or not raw.strip():
        raise DescriptionError("'jobPostingInfo.jobDescription' is missing or empty")

    text = html_to_text(raw)
    if not text:
        # Present but nothing but markup: treat as a failure, not a description.
        raise DescriptionError("description contained no text after HTML stripping")
    return text


def html_to_text(markup: str) -> str:
    """Convert Workday's description HTML to readable plain text.

    Block-level closing tags become newlines and ``<br>`` becomes a newline, so
    paragraph structure survives; everything else is dropped. Entities are
    unescaped and whitespace is collapsed, since the vendor's HTML is machine
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


def detail_url(host: str, tenant: str, site: str, external_path: str) -> str:
    """Build a job-detail URL, normalizing the leading slash of ``external_path``."""
    if not external_path.startswith("/"):
        external_path = "/" + external_path
    return DETAIL_URL.format(
        host=host, tenant=tenant, site=site, external_path=external_path
    )


def _http_get_json(url: str, *, timeout: float = 30.0) -> Any:
    """Default fetcher: a GET returning decoded JSON, errors normalized."""
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

    if response.status_code >= 400:
        raise DescriptionError(f"GET {url}: HTTP {response.status_code}")
    try:
        return response.json()
    except json.JSONDecodeError as error:
        raise DescriptionError(f"GET {url}: response was not JSON: {error}") from error
