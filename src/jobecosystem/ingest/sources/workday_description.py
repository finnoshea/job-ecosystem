"""Workday-specific description handling: detail URLs and payload parsing.

Workday's listing endpoint returns only a title, a location, and an id -- the
full text lives behind a second request per job. The fetching, hashing, storing
and pacing are shared across sources and live in
:mod:`jobecosystem.ingest.sources.description`, which dispatches to
:func:`parse_description` here based on the job's ``source`` prefix.

This module keeps the Workday pieces -- how to build a detail URL from the
``host``/``tenant``/``site`` addressing, and how to read Workday's
``jobPostingInfo.jobDescription`` payload -- and re-exports the shared API so
existing callers (the scraper, the TUI, the tests) keep importing one module.
"""

from __future__ import annotations

from typing import Any

from .description import (
    BatchSummary,
    DescriptionError,
    DescriptionResult,
    FetchedDescription,
    FetchJson,
    FetchOutcome,
    fetch_description,
    fetch_description_from_url,
    fetch_descriptions,
    html_to_text,
    store_description,
)

#: The shared API re-exported above, plus the Workday pieces defined here.
__all__ = [
    "BatchSummary",
    "DETAIL_URL",
    "DescriptionError",
    "DescriptionResult",
    "FetchJson",
    "FetchOutcome",
    "FetchedDescription",
    "detail_url",
    "fetch_description",
    "fetch_description_from_known_url",
    "fetch_description_from_url",
    "fetch_descriptions",
    "html_to_text",
    "parse_description",
    "store_description",
]

#: Workday's job-detail endpoint. ``external_path`` is the value of the
#: listing's ``externalPath`` field, leading slash included.
DETAIL_URL = "https://{host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{external_path}"


def detail_url(host: str, tenant: str, site: str, external_path: str) -> str:
    """Build a job-detail URL, normalizing the leading slash of ``external_path``."""
    if not external_path.startswith("/"):
        external_path = "/" + external_path
    return DETAIL_URL.format(
        host=host, tenant=tenant, site=site, external_path=external_path
    )


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


def fetch_description_from_known_url(
    job_id: int,
    title: str,
    company: str,
    url: str,
    *,
    fetch_json: FetchJson | None = None,
) -> FetchOutcome:
    """Workday wrapper over the shared fetcher, with ``source`` filled in.

    Kept for callers that know they hold a Workday job and should not have to
    spell out the source; the shared version in
    :mod:`jobecosystem.ingest.sources.description` takes ``source`` explicitly.
    """
    from .description import fetch_description_from_known_url as _fetch

    return _fetch(job_id, title, company, url, source="workday", fetch_json=fetch_json)
