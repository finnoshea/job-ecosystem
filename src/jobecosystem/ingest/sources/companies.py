"""Shared parsing for the per-board ``*_companies.txt`` config files.

Ashby, Greenhouse, Lever and SmartRecruiters all read the same shape of file:
one entry per line, whitespace-separated, with ``#`` comments and blank lines
ignored and duplicates collapsed (first occurrence wins). Only the *meaning* of
the tokens after the first differs -- Ashby/Greenhouse/Lever read them as a
display name, SmartRecruiters reads an optional country code first -- so the
tokenizing lives here and each source maps the fields onto its own spec.

Workday is the exception: ``workday_tenants.txt`` is sectioned (host, tenant,
site) and is parsed by :mod:`jobecosystem.ingest.sources.workday_tenants`, which
does not use this module.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Entry:
    """One non-comment line: the leading identifier and the tokens after it."""

    slug: str
    fields: tuple[str, ...]

    @property
    def name(self) -> str | None:
        """The remaining tokens joined, or ``None`` when there are none."""
        return " ".join(self.fields) or None


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """What was understood from a companies file, and what was not."""

    entries: list[Entry]
    errors: list[tuple[int, str]]

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def parse_lines(text: str) -> ParseOutcome:
    """Tokenize a companies file, skipping comments, blanks, and duplicates.

    The first token is the identifier; the rest is left for the caller to
    interpret. A line that cannot be tokenized (unbalanced quotes) is recorded
    as an error rather than raising, so one typo does not stop the scrape.
    Duplicates keep the first occurrence, so an earlier line that names a
    company is not overwritten by a later bare duplicate.
    """
    entries: list[Entry] = []
    errors: list[tuple[int, str]] = []
    seen: set[str] = set()

    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            parts = shlex.split(line)
        except ValueError as error:  # unbalanced quotes
            errors.append((number, f"could not parse: {error}"))
            continue
        if not parts:
            continue

        slug = parts[0]
        if slug in seen:
            continue
        seen.add(slug)
        entries.append(Entry(slug=slug, fields=tuple(parts[1:])))

    return ParseOutcome(entries=entries, errors=errors)


def resolve_file(
    path: str | os.PathLike[str] | None,
    *,
    env_var: str,
    default: Path,
) -> Path:
    """Resolve a companies file: argument > environment > repository default."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get(env_var)
    if configured:
        return Path(configured).expanduser()
    return default


def load_lines(
    path: str | os.PathLike[str] | None,
    *,
    env_var: str,
    default: Path,
) -> ParseOutcome:
    """Read and tokenize a companies file, or an empty outcome if it is absent.

    A missing optional config file means "nothing configured", which the caller
    reports clearly; it is not an error here.
    """
    companies_file = resolve_file(path, env_var=env_var, default=default)
    if not companies_file.exists():
        return ParseOutcome(entries=[], errors=[])
    return parse_lines(companies_file.read_text(encoding="utf-8"))
