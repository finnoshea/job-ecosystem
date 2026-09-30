"""Loading the list of Ashby boards to scrape.

Ashby-specific: this parses the ``ashby_boards.txt`` format and builds
:class:`~jobecosystem.ingest.sources.ashby.AshbyScraper` instances. Other
sources need their own loader, since their configuration differs (Workday, for
instance, is per-tenant rather than per-board).

The list is a plain text file so it can be edited without touching code:

    ashby_boards.txt, one board per line
        slug
        slug  Display Name
        # comment
        <blank>

The parser is deliberately forgiving -- it skips comments, blanks, and duplicate
slugs rather than failing the whole run. A typo in one line must not stop the
daily scrape.

Location is resolved like the database path: an explicit argument, then
``ASHBY_BOARDS_FILE``, then ``ashby_boards.txt`` at the repo root.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from ..base import Scraper
from .ashby import AshbyScraper

#: src/jobecosystem/ingest/sources/ashby_boards.py
#:   -> sources -> ingest -> jobecosystem -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BOARDS_FILE = _REPO_ROOT / "ashby_boards.txt"


@dataclass(frozen=True, slots=True)
class BoardSpec:
    """One configured board: the slug, plus an optional display name."""

    slug: str
    company: str | None = None

    @property
    def source(self) -> str:
        """The ``jobs.source`` label: ``ashby:<slug>``."""
        return f"ashby:{self.slug}"


def resolve_boards_file(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the board list file: argument > ``ASHBY_BOARDS_FILE`` > default."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("ASHBY_BOARDS_FILE")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_BOARDS_FILE


def parse_boards(text: str, *, origin: str = "<string>") -> list[BoardSpec]:
    """Parse board-list text into specs, skipping anything unusable.

    Comments (``#``) and blank lines are ignored. Duplicate slugs are collapsed,
    keeping the first occurrence so an earlier line with a display name wins.
    Lines with extra tokens beyond ``slug [Display Name]`` are still accepted:
    the remainder becomes the display name, so no quoting is required.
    """
    specs: list[BoardSpec] = []
    seen: set[str] = set()

    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = shlex.split(line)
        if not parts:
            continue

        slug = parts[0]
        if slug in seen:
            # First occurrence wins, so a display name is not overwritten by a
            # later bare duplicate.
            continue
        seen.add(slug)

        company = " ".join(parts[1:]) or None
        specs.append(BoardSpec(slug=slug, company=company))

    return specs


def load_boards(
    path: str | os.PathLike[str] | None = None,
) -> list[BoardSpec]:
    """Load and parse the board list file.

    Returns an empty list when the file does not exist, rather than raising: a
    missing optional config file should mean "no boards configured", which the
    caller can report clearly.
    """
    boards_file = resolve_boards_file(path)
    if not boards_file.exists():
        return []
    text = boards_file.read_text(encoding="utf-8")
    return parse_boards(text, origin=str(boards_file))


def build_scrapers(
    path: str | os.PathLike[str] | None = None,
    *,
    fetch_json=None,
) -> list[Scraper]:
    """Build one :class:`AshbyScraper` per configured board.

    ``fetch_json`` is passed through for tests, which supply a stub instead of
    hitting the network.
    """
    return [
        AshbyScraper(
            spec.slug,
            company=spec.company,
            fetch_json=fetch_json,
        )
        for spec in load_boards(path)
    ]
