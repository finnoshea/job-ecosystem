"""Discover new ATS boards to scrape by mining job URLs from elsewhere.

The scrapers are per-board: each needs a company slug in a ``*_companies.txt``
file, and that file is the bottleneck -- a source is only as wide as its list.
This module widens the lists by reading places that already list many jobs and
pulling out the URLs that point at a known ATS.

Today there is one harvester: :func:`harvest_hn`, which reads the current
"Ask HN: Who is hiring?" thread. Those comments are written by the companies
themselves and are dense with ``jobs.lever.co`` / ``apply.workable.com`` links,
which is exactly the small-company tail the per-board lists miss.

A second harvester over job aggregators (RemoteOK, Himalayas, Jobicy,
Arbeitnow) was written and then removed: none of them expose the employer's ATS
URL. Their records link to their own job pages, the ATS link is not in the
description, and the landing pages link to the employer's own careers site (or
block non-browser clients). Scanning ~500 records turned up one ATS slug, so it
was noise, not coverage. Reaching those employers needs a careers-page crawler
that reads the employer's own site, which is a different feature.

The tool itself never talks to an ATS. Its output is company slugs appended to
``lever_companies.txt``, ``workable_companies.txt`` and friends, which the
scrapers read on their next run; nothing here requires a new adapter to exist
first.

Testability follows the scrapers: the HTTP call is injected, and the
recognition and merge steps are pure functions over text.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .base import FetchError

#: src/jobecosystem/ingest/discover.py -> ingest -> jobecosystem -> src -> root
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Signature of the injectable HTTP call: URL -> decoded JSON. ``user_agent`` is
#: accepted so a future source can send one if it needs to.
FetchJson = Callable[..., Any]


# ---------------------------------------------------------------------------
# recognising an ATS URL
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Family:
    """One ATS, its URL shape, and the companies file it feeds.

    ``pattern`` captures the company slug in its first non-empty group. An
    alternation is allowed (Workable's account appears in two host shapes), so
    extraction takes the first group that matched rather than a fixed index.
    """

    key: str
    pattern: re.Pattern[str]
    companies_file: str


#: Every ATS this tool can recognise. Greenhouse, Ashby and SmartRecruiters are
#: included so discovery grows the hand-maintained lists too, not just the new
#: ones. Workday is deliberately absent: its identity is host + tenant + site in
#: a sectioned file, which the flat ``slug`` model here does not represent.
FAMILIES: tuple[Family, ...] = (
    Family(
        "lever",
        re.compile(r"https?://jobs\.lever\.co/([A-Za-z0-9][\w.-]*)"),
        "lever_companies.txt",
    ),
    Family(
        "workable",
        re.compile(
            r"https?://apply\.workable\.com/([A-Za-z0-9][\w-]*)"
            r"|https?://([A-Za-z0-9][\w-]*)\.workable\.com(?:[/?#]|$)"
        ),
        "workable_companies.txt",
    ),
    Family(
        "greenhouse",
        re.compile(r"https?://(?:job-)?boards\.greenhouse\.io/([A-Za-z0-9][\w-]*)"),
        "greenhouse_companies.txt",
    ),
    Family(
        "ashby",
        re.compile(r"https?://jobs\.ashbyhq\.com/([A-Za-z0-9][\w.-]*)"),
        "ashby_boards.txt",
    ),
    Family(
        "smartrecruiters",
        re.compile(r"https?://jobs\.smartrecruiters\.com/([A-Za-z0-9][\w-]*)"),
        "smartrecruiters_companies.txt",
    ),
)

FAMILIES_BY_KEY = {family.key: family for family in FAMILIES}

#: Path segments that look like a slug but are part of the ATS's own site.
_DENY = frozenset({
    "apply", "jobs", "www", "embed", "job_app", "api", "login", "oauth",
    "v1", "company", "static", "cdn",
})


def extract_slugs(text: str | None) -> dict[str, set[str]]:
    """Every known ATS company slug in an arbitrary blob of text.

    The input is decoded first: aggregators percent-encode the employer URL
    inside their own links, and HN hands back HTML entities. Slugs are lowercased
    because every one of these ATSes treats them that way, so ``Acme`` and
    ``acme`` cannot end up as two lines in the same file.
    """
    if not text:
        return {}

    decoded = unquote(html.unescape(text))
    found: dict[str, set[str]] = {}
    for family in FAMILIES:
        for match in family.pattern.finditer(decoded):
            slug = next((group for group in match.groups() if group), None)
            if not slug:
                continue
            slug = slug.strip().lower()
            if len(slug) < 2 or slug in _DENY:
                continue
            found.setdefault(family.key, set()).add(slug)
    return found


# ---------------------------------------------------------------------------
# what a harvest found
# ---------------------------------------------------------------------------

@dataclass
class Harvest:
    """Slugs found by a source, plus the company names that came with them.

    ``slugs`` maps family -> set of slugs; ``names`` maps family -> slug ->
    display name. Names are best-effort and never affect whether a slug is kept.
    """

    slugs: dict[str, set[str]] = field(default_factory=dict)
    names: dict[str, dict[str, str]] = field(default_factory=dict)

    def add(self, text: str | None, company: str | None = None) -> None:
        """Record every ATS URL in ``text``, optionally labelling them."""
        for family, values in extract_slugs(text).items():
            self.slugs.setdefault(family, set()).update(values)
            if company:
                labels = self.names.setdefault(family, {})
                for slug in values:
                    labels.setdefault(slug, company)

    def merge(self, other: "Harvest") -> None:
        """Fold another harvest into this one, keeping the first name seen."""
        for family, values in other.slugs.items():
            self.slugs.setdefault(family, set()).update(values)
        for family, labels in other.names.items():
            mine = self.names.setdefault(family, {})
            for slug, name in labels.items():
                mine.setdefault(slug, name)

    def __bool__(self) -> bool:
        return any(self.slugs.values())


# ---------------------------------------------------------------------------
# harvester: Hacker News "Who is hiring?"
# ---------------------------------------------------------------------------

#: The newest story by the ``whoishiring`` bot whose title starts this way is
#: the current monthly thread. Resolved each run because the id changes monthly.
HN_SEARCH_URL = (
    "https://hn.algolia.com/api/v1/search"
    "?tags=story,author_whoishiring&hitsPerPage=10"
)
HN_ITEM_URL = "https://hn.algolia.com/api/v1/items/{id}"


def latest_thread_id(payload: Any) -> str | None:
    """The id of the newest "Ask HN: Who is hiring?" story, or ``None``."""
    if not isinstance(payload, dict):
        return None
    hits = payload.get("hits")
    if not isinstance(hits, list):
        return None
    stories = [
        hit
        for hit in hits
        if isinstance(hit, dict)
        and str(hit.get("title") or "").startswith("Ask HN: Who is hiring")
        and hit.get("objectID")
    ]
    if not stories:
        return None
    newest = max(stories, key=lambda hit: str(hit.get("created_at") or ""))
    return str(newest["objectID"])


def harvest_hn(fetch_json: FetchJson) -> Harvest:
    """Read the current hiring thread and collect everything it links to."""
    payload = fetch_json(HN_SEARCH_URL)
    thread_id = latest_thread_id(payload)
    if thread_id is None:
        raise FetchError("no 'Ask HN: Who is hiring?' thread found")
    return harvest_hn_thread(fetch_json(HN_ITEM_URL.format(id=thread_id)))


def harvest_hn_thread(thread: Any) -> Harvest:
    """Collect slugs and company names from a thread's top-level comments.

    Only top-level comments are read: replies are discussion, not postings.
    A comment with no text (deleted, or a dead reply stub) is skipped.
    """
    if not isinstance(thread, dict):
        raise FetchError(
            f"thread payload was {type(thread).__name__}, expected an object"
        )
    harvest = Harvest()
    for comment in thread.get("children") or []:
        if not isinstance(comment, dict):
            continue
        text = comment.get("text")
        if not text:
            continue
        harvest.add(text, _company_from_comment(text))
    return harvest


_TAGS = re.compile(r"<[^>]+>")


def _company_from_comment(text: str) -> str | None:
    """Best-effort company name from a hiring comment.

    Most posts open with ``Company | Role | Location | ...``; take the first
    field and drop a trailing ``(YC W20)`` tag. Never raises: a bad guess just
    means the slug is written without a display name.
    """
    plain = html.unescape(_TAGS.sub(" ", text))
    for raw_line in plain.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        name = line.split("|", 1)[0]
        name = re.sub(r"\s*\((?:YC|yc)[^)]*\)\s*$", "", name)
        name = re.sub(r"\s+[-–—]\s*$", "", name).strip(" .,:;")
        if 1 < len(name) <= 80:
            return name
        return None
    return None


# ---------------------------------------------------------------------------
# running a source
# ---------------------------------------------------------------------------

#: Source keys ``harvest`` understands. Only HN, for the reason in the module
#: docstring; the dispatch is kept so a working source can be added in one place.
SOURCES: tuple[str, ...] = ("hn",)


def harvest(key: str, fetch_json: FetchJson) -> Harvest:
    """Run the named source."""
    if key == "hn":
        return harvest_hn(fetch_json)
    raise FetchError(f"unknown source {key!r}")


def http_get_json(url: str, *, user_agent: str | None = None) -> Any:
    """Default fetcher: a GET returning decoded JSON.

    Imported lazily so the parsers can be used, and tested, without httpx.
    """
    import httpx

    headers = {"Accept": "application/json"}
    if user_agent:
        headers["User-Agent"] = user_agent
    try:
        response = httpx.get(
            url, timeout=30.0, follow_redirects=True, headers=headers
        )
    except httpx.HTTPError as error:
        raise FetchError(f"GET {url} failed: {error}") from error
    if response.status_code >= 400:
        raise FetchError(f"GET {url}: HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as error:
        raise FetchError(f"GET {url}: response was not JSON: {error}") from error


# ---------------------------------------------------------------------------
# writing the companies files
# ---------------------------------------------------------------------------

#: Written when a companies file does not exist yet. Existing files (with their
#: curated headers) are never rewritten -- new lines are only appended.
DEFAULT_HEADER = """\
# Companies discovered automatically; review before relying on them.
# One slug per line, optionally followed by a display name.
"""


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    """What a merge did, or would do under ``--dry-run``."""

    path: Path
    added: list[str]
    existing: int


def read_existing_slugs(path: str | Path) -> set[str]:
    """First token of every non-comment line, lowercased.

    The same shape the scrapers read: ``slug [more...]``. Comments and blank
    lines are ignored, so re-merging cannot duplicate an entry.
    """
    path = Path(path)
    if not path.exists():
        return set()
    known: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        known.add(line.split()[0].lower())
    return known


def merge_into_file(
    path: str | Path,
    slugs: set[str] | list[str],
    *,
    names: dict[str, str] | None = None,
    dry_run: bool = False,
    header: str = DEFAULT_HEADER,
) -> MergeOutcome:
    """Append slugs not already present, preserving the file's comments.

    Returns what was added (or would be). One line per slug, sorted, with the
    display name when one is known. Append-only, so a bad discovery run is a
    ``git checkout`` away from undone.
    """
    path = Path(path)
    known = read_existing_slugs(path)
    additions = sorted(
        {slug for slug in slugs if slug and slug.lower() not in known}
    )
    if dry_run or not additions:
        return MergeOutcome(path=path, added=additions, existing=len(known))

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if not path.exists():
        lines.append(header.rstrip() + "\n")
    labels = names or {}
    for slug in additions:
        name = labels.get(slug)
        lines.append(f"{slug}  {name}\n" if name else f"{slug}\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)
    return MergeOutcome(path=path, added=additions, existing=len(known))
