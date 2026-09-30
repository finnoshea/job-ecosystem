"""Loading the list of Workday sites to scrape.

Workday-specific: this parses the ``workday_tenants.txt`` format and builds
:class:`~jobecosystem.ingest.sources.workday.WorkdayScraper` instances.

Workday addressing needs three parts per line, unlike Ashby's single slug:

    workday_tenants.txt, one site per line
        host  tenant  site  [Display Name]
        # comment
        <blank>

The header of that file explains where each part comes from.

The parser is deliberately forgiving -- comments, blanks, and duplicate
host+tenant+site triples are skipped rather than failing the run, since a typo
in one line must not stop the daily scrape. Malformed lines are reported through
the returned errors list rather than raised, so a partially broken config still
scrapes what it can.

Location is resolved like the database path: an explicit argument, then
``WORKDAY_TENANTS_FILE``, then ``workday_tenants.txt`` at the repo root.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from ..base import Scraper
from .workday import TenantSpec, WorkdayScraper

#: src/jobecosystem/ingest/sources/workday_tenants.py
#:   -> sources -> ingest -> jobecosystem -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TENANTS_FILE = _REPO_ROOT / "workday_tenants.txt"


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """The result of parsing a tenants file: what was understood, and what was not."""

    specs: list[TenantSpec]
    errors: list[tuple[int, str]]

    def __bool__(self) -> bool:
        return bool(self.specs)

    def __len__(self) -> int:
        return len(self.specs)


def resolve_tenants_file(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the tenants file: argument > ``WORKDAY_TENANTS_FILE`` > default."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("WORKDAY_TENANTS_FILE")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_TENANTS_FILE


def parse_tenants(text: str) -> ParseOutcome:
    """Parse tenants-file text into specs, collecting anything unusable.

    Accepts three fields (``host tenant site``) or four (plus a display name).
    Duplicate host+tenant+site triples collapse to the first occurrence, so the
    same site listed twice is one source rather than two.

    A site name cannot contain a space: anything after the third field is taken
    as the display name. Workday site names are URL path segments, so they never
    have spaces in practice.
    """
    specs: list[TenantSpec] = []
    errors: list[tuple[int, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        try:
            parts = shlex.split(line)
        except ValueError as error:  # unbalanced quotes
            errors.append((number, f"could not parse: {error}"))
            continue

        if len(parts) < 3:
            errors.append((
                number,
                f"expected 'host tenant site [name]', got {len(parts)} field(s)",
            ))
            continue

        host, tenant, site = parts[0], parts[1], parts[2]
        company = " ".join(parts[3:]) or None

        key = (host, tenant, site)
        if key in seen:
            continue
        seen.add(key)
        specs.append(TenantSpec(host=host, tenant=tenant, site=site, company=company))

    return ParseOutcome(specs=specs, errors=errors)


def load_tenants(
    path: str | os.PathLike[str] | None = None,
) -> ParseOutcome:
    """Load and parse the tenants file.

    Returns an empty outcome when the file does not exist, rather than raising:
    a missing optional config file means "no tenants configured", which the
    caller can report clearly.
    """
    tenants_file = resolve_tenants_file(path)
    if not tenants_file.exists():
        return ParseOutcome(specs=[], errors=[])
    return parse_tenants(tenants_file.read_text(encoding="utf-8"))


def build_scrapers(
    path: str | os.PathLike[str] | None = None,
    *,
    post_json=None,
) -> list[Scraper]:
    """Build one :class:`WorkdayScraper` per configured site.

    ``post_json`` is passed through for tests, which supply a stub instead of
    hitting the network.
    """
    return [
        WorkdayScraper.from_spec(spec, post_json=post_json)
        for spec in load_tenants(path).specs
    ]
