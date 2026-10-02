"""Command-line entry point for scraping.

Intended for cron: no prompts, no TTY assumptions, logs nothing to stdout except
a short summary, and exits non-zero when any source failed so the scheduler
notices. All the durable detail lands in SQL table ``scrape_runs`` and
``scrape_errors``, which is what to inspect afterwards -- see SQL view
``run_stats``.

    jobecosystem-scrape                       # every configured source
    jobecosystem-scrape --source ashby        # only Ashby boards
    jobecosystem-scrape --db /tmp/jobs.db     # a different database
    jobecosystem-scrape --quiet               # just the exit code

Exit codes:

    0   every source succeeded
    1   at least one source failed (the run still happened)
    2   could not start: bad arguments, unopenable database, no sources configured
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from typing import IO, Sequence

from ..core import db as core_db
from . import runner
from .base import Scraper
from .sources.ashby_boards import build_scrapers as build_ashby_scrapers
from .sources.smartrecruiters import build_scrapers as build_smartrecruiters_scrapers
from .sources.workday_tenants import build_scrapers as build_workday_scrapers

PROGRAM = "jobecosystem-scrape"

EXIT_OK = 0
EXIT_SOURCE_FAILED = 1
EXIT_SETUP_FAILED = 2


@dataclass(slots=True)
class Selection:
    """Which sources to run, and any problem encountered while resolving them."""

    scrapers: list[Scraper] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.scrapers)


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Scrape configured job boards into the SQLite database.",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help=(
            "database file; defaults to $DB_PATH, then jobs.db at the repo root"
        ),
    )
    parser.add_argument(
        "--source",
        metavar="NAME",
        action="append",
        default=None,
        help=(
            "only run sources whose label starts with NAME (e.g. 'ashby' or"
            " 'workday'). Repeatable. Default: all configured sources."
        ),
    )
    parser.add_argument(
        "--ashby-boards",
        metavar="PATH",
        default=None,
        help="board list file; defaults to $ASHBY_BOARDS_FILE, then ashby_boards.txt",
    )
    parser.add_argument(
        "--workday-tenants",
        metavar="PATH",
        default=None,
        help=(
            "tenant list file; defaults to $WORKDAY_TENANTS_FILE, then"
            " workday_tenants.txt"
        ),
    )
    parser.add_argument(
        "--smartrecruiters-companies",
        metavar="PATH",
        default=None,
        help=(
            "company list file; defaults to $SMARTRECRUITERS_COMPANIES_FILE,"
            " then smartrecruiters_companies.txt"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print nothing on success; errors still go to stderr",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the sources that would run, then exit without scraping",
    )
    return parser


def select_scrapers(args: argparse.Namespace) -> Selection:
    """Build the scrapers for this invocation, honouring ``--source``.

    ``--source`` matches either a family (``ashby``, ``workday``) or a full
    source label (``workday:nvidia:NVIDIAExternalCareerSite``). A full label is
    refined *after* the scrapers are built, because a source label is not known
    until the config file is parsed -- so family selection happens first, then
    any exact labels filter the result.
    """
    selection = Selection()

    family, exact = _split_filters(args.source)
    include_ashby = not family or "ashby" in family
    include_workday = not family or "workday" in family
    include_smartrecruiters = not family or "smartrecruiters" in family

    if include_ashby:
        selection.scrapers.extend(build_ashby_scrapers(args.ashby_boards))
    if include_workday:
        selection.scrapers.extend(build_workday_scrapers(args.workday_tenants))
    if include_smartrecruiters:
        selection.scrapers.extend(
            build_smartrecruiters_scrapers(args.smartrecruiters_companies)
        )

    if exact:
        selection.scrapers = [
            scraper for scraper in selection.scrapers
            if scraper.source in exact or any(
                scraper.source.startswith(label) for label in exact
            )
        ]

    if args.source and not selection.scrapers:
        selection.errors.append(
            f"no sources matched {', '.join(sorted(set(args.source)))};"
            " check the spelling against the configured files"
        )
    return selection


def _split_filters(
    wanted: Sequence[str] | None,
) -> tuple[set[str], set[str]]:
    """Split ``--source`` values into family names and full labels.

    A value with a ``:`` is a full label (``workday:t:S``); anything else is a
    family name (``workday``). Distinguishing them is what lets a filter narrow
    within one family instead of selecting the whole family.
    """
    if not wanted:
        return set(), set()
    families = {name for name in wanted if ":" not in name}
    labels = {name for name in wanted if ":" in name}
    return families, labels


def run(
    args: argparse.Namespace,
    *,
    out: IO[str] | None = None,
    err: IO[str] | None = None,
) -> int:
    """Run the scrape described by ``args``; return an exit code.

    Separate from :func:`main` so tests can drive it without touching
    ``sys.argv`` or a real process exit. ``out``/``err`` default to the live
    streams, resolved at call time rather than bound at definition time, so a
    redirected ``sys.stdout`` is honoured.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    db_path = core_db.resolve_db_path(args.db)

    try:
        conn = core_db.connect(db_path)
    except sqlite3.Error as error:
        print(f"{PROGRAM}: could not open {db_path}: {error}", file=err)
        return EXIT_SETUP_FAILED

    try:
        selection = select_scrapers(args)
        for message in selection.errors:
            print(f"{PROGRAM}: {message}", file=err)

        if args.dry_run:
            if not args.quiet:
                print(f"would scrape {len(selection)} source(s) into {db_path}", file=out)
                for scraper in selection.scrapers:
                    print(f"  {scraper.source}", file=out)
            return EXIT_SETUP_FAILED if selection.errors else EXIT_OK

        if not len(selection):
            print(
                f"{PROGRAM}: no sources configured; check ashby_boards.txt,"
                " workday_tenants.txt, and smartrecruiters_companies.txt",
                file=err,
            )
            return EXIT_SETUP_FAILED

        summary = runner.run_scrapers(
            conn,
            selection.scrapers,
            # Report each source as it finishes; the totals line below follows
            # once the last one is done.
            on_outcome=(
                None if args.quiet
                else lambda outcome: _report_outcome(outcome, out=out)
            ),
        )
        if not args.quiet:
            _report_totals(summary, out=out)
        if not summary.ok:
            print(
                f"{PROGRAM}: {len(summary.failed_sources)} source(s) failed:"
                f" {', '.join(summary.failed_sources)}",
                file=err,
            )
            return EXIT_SOURCE_FAILED
        return EXIT_OK
    finally:
        conn.close()


def _report_outcome(outcome: runner.SourceOutcome, *, out) -> None:
    """Print one source's result as soon as that source finishes.

    Flushed immediately: under cron stdout is usually a pipe, and without the
    flush these lines would sit in the buffer until the process exits -- the
    very delay this exists to remove.
    """
    status = "ok" if outcome.ok else "FAILED"
    detail = f"{outcome.fetched} fetched, {outcome.inserted} new"
    if outcome.errors:
        detail += f", {len(outcome.errors)} error(s)"
    if outcome.error:
        detail += f" -- {outcome.error}"
    print(f"{outcome.source}: {status} ({detail})", file=out, flush=True)


def _report_totals(summary: runner.RunSummary, *, out) -> None:
    """Print the run-wide totals once every source has reported.

    Deliberately terse. The database holds the detail; this exists so a failed
    cron mail says which source broke and how much moved.
    """
    fetched = sum(o.fetched for o in summary)
    inserted = sum(o.inserted for o in summary)
    # Two different kinds of trouble: per-listing errors recorded during a run
    # that still completed, and sources that failed outright. Counting only the
    # former would report "0 errors" next to a FAILED line.
    listing_errors = sum(len(o.errors) for o in summary)
    failed = len(summary.failed_sources)

    print(
        f"{len(summary)} source(s): {fetched} fetched, {inserted} new,"
        f" {listing_errors} listing error(s), {failed} failed source(s)",
        file=out,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
