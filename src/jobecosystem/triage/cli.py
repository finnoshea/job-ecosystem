"""Command-line entry point for triage.

Separate from the scraper entry point on purpose: this one needs a terminal and
a human, while ``jobecosystem-scrape`` runs unattended on a timer and must
report failure through its exit code. They share only the database path.

    jobecosystem-triage                   # the default database
    jobecosystem-triage --db /tmp/jobs.db
    jobecosystem-triage --list-sources    # what is in there, then exit

Exit codes:

    0   the interface ran and was closed normally
    2   could not start: unopenable database, or the TUI extra is missing
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from typing import Sequence

from ..core import db as core_db

PROGRAM = "jobecosystem-triage"

EXIT_OK = 0
EXIT_SETUP_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Browse and triage scraped jobs in a terminal interface.",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="database file; defaults to $DB_PATH, then jobs.db at the repo root",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=300,
        metavar="N",
        help="rows to load per view (default: 300)",
    )
    parser.add_argument(
        "--resume",
        metavar="PATH",
        default=None,
        help="resume JSON to match jobs against with ctrl+r; defaults to"
        " $TAILOR_RESUME_FILE, then resume.base.json at the repo root",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="print the stored sources and job counts, then exit",
    )
    return parser


def run(args: argparse.Namespace, *, out=sys.stdout, err=sys.stderr) -> int:
    """Run the interface described by ``args``; return an exit code.

    Separate from :func:`main` so tests can drive the non-interactive paths
    without starting a terminal application.
    """
    db_path = core_db.resolve_db_path(args.db)
    try:
        conn = core_db.connect(db_path)
    except sqlite3.Error as error:
        print(f"{PROGRAM}: could not open {db_path}: {error}", file=err)
        return EXIT_SETUP_FAILED

    try:
        if args.list_sources:
            from . import queries as q

            counts = q.counts_by_source(conn)
            if not counts:
                print(f"{db_path}: no jobs stored yet", file=out)
                return EXIT_OK
            width = max(len(source) for source in counts)
            for source, count in counts.items():
                print(f"{source:<{width}}  {count:>6}", file=out)
            print(f"{'total':<{width}}  {sum(counts.values()):>6}", file=out)
            return EXIT_OK

        if args.page_size <= 0:
            print(f"{PROGRAM}: --page-size must be positive", file=err)
            return EXIT_SETUP_FAILED

        try:
            from .tui.app import run as run_tui
        except ImportError as error:
            print(
                f"{PROGRAM}: the terminal interface needs the 'tui' extra;"
                f" install with `pip install -e .[tui]` ({error})",
                file=err,
            )
            return EXIT_SETUP_FAILED

        return run_tui(
            db_path, page_size=args.page_size, resume_path=args.resume
        )
    finally:
        conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
