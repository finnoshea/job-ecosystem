"""Workday job scraper.

Workday's public job API is per-tenant *and* per-site, so one company may be
several sources and the addressing is three-part::

    POST https://{host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs
    body {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""}

Unlike Ashby there is no company slug that identifies a board. A bare host
returns HTTP 406, and the site name is not derivable from the company name, so
each site is configured explicitly in ``workday_tenants.txt`` (loaded by
``workday_tenants.py``). See that file for how to find the values.

Two consequences shape this scraper:

* **Page size is capped at 20.** ``limit`` above 20 is rejected with HTTP 400,
  not truncated, so large tenants paginate by ``offset``. The listing gives a
  ``total``, which is trusted for the loop bound but re-checked against what was
  actually collected.
* **Listings carry no description.** A page yields only ``title``,
  ``externalPath``, ``locationsText``, ``postedOn``, and sometimes ``timeType``
  and ``bulletFields``. The full text needs one extra request per job, which is
  deliberately not done here -- see
  :mod:`jobecosystem.ingest.sources.workday_description`. Rows therefore have a
  NULL ``content_hash`` until that fetch happens, and this module stores a
  ``description_url`` so the fetch can find its endpoint later without knowing
  anything about Workday addressing.

``postedOn`` is relative prose ("Posted 5 Days Ago") and is therefore *not*
stored: parsing it would make ``posted_at`` a different date on every run. The
absolute ``startDate`` from the detail endpoint is the better source, and is
left to the description fetcher.

The HTTP dependency is injected so the parser is testable without the network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...core.models import Job
from ..base import FetchError, ParseError, Scraper
from .workday_description import detail_url

#: Job listing endpoint, per tenant and site.
LISTING_URL = "https://{host}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"

#: Workday rejects limit > 20 with HTTP 400, so this is a protocol constant
#: rather than a tunable.
PAGE_SIZE = 20

#: Signature of the injectable HTTP call: (url, body) -> decoded JSON.
PostJson = Callable[[str, dict], Any]

#: Refuse to page forever if a tenant reports an implausible total or the API
#: keeps returning rows regardless of offset. 2000 rows is 100 pages.
MAX_PAGES = 200


@dataclass(frozen=True, slots=True)
class TenantSpec:
    """One configured site: the three addressing parts plus a display name."""

    host: str
    tenant: str
    site: str
    company: str | None = None

    @property
    def label(self) -> str:
        """The ``jobs.source`` value. Includes the site, since one tenant can
        expose several sites with different openings."""
        return f"workday:{self.tenant}:{self.site}"

    @property
    def listing_url(self) -> str:
        """The endpoint this spec reads."""
        return LISTING_URL.format(host=self.host, tenant=self.tenant, site=self.site)


class WorkdayScraper(Scraper):
    """Scrape one Workday site.

    ``company`` is the display name for SQL table ``jobs``'s ``company`` column;
    Workday never returns it, so it falls back to the tenant.
    """

    def __init__(
        self,
        host: str,
        tenant: str,
        site: str,
        *,
        company: str | None = None,
        post_json: PostJson | None = None,
        timeout: float = 30.0,
        max_pages: int = MAX_PAGES,
    ) -> None:
        for name, value in (("host", host), ("tenant", tenant), ("site", site)):
            if not value:
                raise ValueError(f"{name} must be a non-empty string")
        self.spec = TenantSpec(host=host, tenant=tenant, site=site, company=company)
        self.source = self.spec.label
        self.company = company or tenant
        self.timeout = timeout
        self.max_pages = max_pages
        self._post_json = post_json

    @classmethod
    def from_spec(
        cls,
        spec: TenantSpec,
        *,
        post_json: PostJson | None = None,
        max_pages: int = MAX_PAGES,
    ) -> WorkdayScraper:
        """Build a scraper from a parsed config entry."""
        return cls(
            spec.host,
            spec.tenant,
            spec.site,
            company=spec.company,
            post_json=post_json,
            max_pages=max_pages,
        )

    @property
    def host(self) -> str:
        """The Workday host, e.g. ``nvidia.wd5``."""
        return self.spec.host

    @property
    def tenant(self) -> str:
        """The tenant segment of the API path."""
        return self.spec.tenant

    @property
    def site(self) -> str:
        """The site segment of the API path."""
        return self.spec.site

    @property
    def listing_url(self) -> str:
        """The endpoint this scraper reads."""
        return self.spec.listing_url

    def fetch(self) -> list[Job]:
        """Fetch every posting on this site, paginating until exhausted.

        Raises :class:`FetchError` if the first page cannot be read; a failure
        on a *later* page ends pagination with what has been collected so far
        and reports it, since a partial board is far better than none. Per-item
        problems go through :meth:`report_error`.
        """
        jobs: list[Job] = []
        seen: set[str] = set()
        offset = 0
        total: int | None = None

        for page in range(self.max_pages):
            try:
                payload = self._post_page(offset)
            except FetchError:
                if page == 0:
                    raise
                self.report_error(
                    f"pagination stopped at offset {offset} after {len(jobs)} jobs"
                )
                break

            postings = self._extract_postings(payload)
            if total is None:
                total = _as_int(payload.get("total"))

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
                try:
                    job = self.parse_listing(posting)
                except ParseError as error:
                    self.report_error(f"offset {offset} item {index}: {error}")
                    continue
                # A board that ignores offset would otherwise loop forever;
                # de-duplicating here also guards against repeats across pages.
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

            offset += PAGE_SIZE

        if total is not None and len(jobs) < total:
            self.report_error(
                f"collected {len(jobs)} of {total} postings"
            )
        return jobs

    def parse_listing(self, posting: dict) -> Job:
        """Map one Workday listing onto a :class:`Job`.

        Requires only ``title`` and ``externalPath``, since those are what
        identify a row and locate its detail endpoint. No description is set;
        ``description_url`` is, so the paced fetcher can find one later.
        """
        title = _required_str(posting, "title")
        external_path = _required_str(posting, "externalPath")

        return Job(
            source=self.source,
            external_id=self._external_id(posting, external_path),
            company=self.company,
            title=title,
            location=_clean(posting.get("locationsText")),
            description=None,            # listings carry none; see module docstring
            url=self.public_url(external_path),
            description_url=detail_url(
                self.host, self.tenant, self.site, external_path
            ),
            posted_at=None,              # postedOn is relative prose; not stored
            raw_json=_raw(posting),
        )

    def public_url(self, external_path: str) -> str:
        """The human-facing URL for a posting."""
        path = external_path if external_path.startswith("/") else "/" + external_path
        return (
            f"https://{self.host}.myworkdayjobs.com/{self.site}{path}"
        )

    def _external_id(self, posting: dict, external_path: str) -> str:
        """Stable id for the posting.

        Prefers ``bulletFields[0]`` -- the requisition number, e.g. ``JR2026359``
        or ``10145210`` -- because it survives a job being renamed, which changes
        the slug inside ``externalPath``. Falls back to the path itself when
        ``bulletFields`` is absent or unusable.
        """
        bullets = posting.get("bulletFields")
        if isinstance(bullets, list):
            for value in bullets:
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return external_path

    def _post_page(self, offset: int) -> dict:
        """Request one page of listings."""
        body = {
            "appliedFacets": {},
            "limit": PAGE_SIZE,
            "offset": offset,
            "searchText": "",
        }
        poster = self._post_json or _http_post_json
        try:
            payload = poster(self.listing_url, body)
        except FetchError:
            raise
        except Exception as error:  # noqa: BLE001 - normalized for the runner
            raise FetchError(
                f"workday site {self.source!r}: could not read {self.listing_url}:"
                f" {error}"
            ) from error

        if not isinstance(payload, dict):
            raise FetchError(
                f"workday site {self.source!r}: response was"
                f" {type(payload).__name__}, expected an object"
            )
        if "jobPostings" not in payload:
            raise FetchError(
                f"workday site {self.source!r}: response has no 'jobPostings' list"
            )
        return payload

    def _extract_postings(self, payload: dict) -> list[Any]:
        postings = payload.get("jobPostings")
        if postings is None:
            return []
        if not isinstance(postings, list):
            raise FetchError(
                f"workday site {self.source!r}: 'jobPostings' was"
                f" {type(postings).__name__}, expected a list"
            )
        return postings


def _http_post_json(url: str, body: dict, *, timeout: float = 30.0) -> Any:
    """Default fetcher: POST JSON and decode the JSON response."""
    import httpx

    try:
        response = httpx.post(
            url,
            json=body,
            timeout=timeout,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as error:
        raise FetchError(f"POST {url} failed: {error}") from error

    if response.status_code == 404:
        raise FetchError(f"POST {url}: site not found (404)")
    if response.status_code == 400:
        raise FetchError(f"POST {url}: HTTP 400 (bad request body or page size)")
    if response.status_code >= 400:
        raise FetchError(f"POST {url}: HTTP {response.status_code}")
    try:
        return response.json()
    except Exception as error:  # noqa: BLE001 - any decode failure is fatal here
        raise FetchError(f"POST {url}: response was not JSON: {error}") from error


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


def _as_int(value: Any) -> int | None:
    """Best-effort integer, for the ``total`` field."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _raw(posting: dict) -> str | None:
    """The vendor payload, for debugging a parser that broke."""
    import json

    try:
        return json.dumps(posting, ensure_ascii=False)
    except (TypeError, ValueError):
        return None
