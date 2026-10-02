"""Command-line entry point for the paced description fetch.

Intended for cron, like ``jobecosystem-scrape``: no prompts, no TTY
assumptions, one terse summary, and an exit code the scheduler can act on. It is
a separate command from the scrape on purpose -- descriptions are fetched one
paced request per job, so they belong on their own schedule, not bolted onto the
listing scrape.

    jobecosystem-describe --limit 300          # up to 300 descriptions this run
    jobecosystem-describe --db /tmp/jobs.db    # a different database
    jobecosystem-describe --quiet              # just the exit code

The queue is SQL view ``jobs_needing_descriptions``, so a run resumes where the
last one stopped: a completed row leaves the queue by gaining a ``content_hash``.

Exit codes:

    0   every attempted description was fetched (including "nothing to do")
    1   at least one description failed (the rest still happened)
    2   could not start: bad arguments, unopenable database
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from typing import IO, Sequence

from ..core import db as core_db
from .sources import description

PROGRAM = "jobecosystem-describe"

EXIT_OK = 0
EXIT_DESCRIPTION_FAILED = 1
EXIT_SETUP_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Fetch pending job descriptions into the SQLite database.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        metavar="N",
        type=int,
        default=200,
        help=(
            "maximum descriptions to fetch this run (default: 200); bounds how"
            " long a scheduled run takes"
        ),
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
        "--delay",
        metavar="SECONDS",
        type=float,
        default=description.DEFAULT_DELAY,
        help=(
            "pause between requests (default: %(default)s); request rate, not"
            " volume, is what gets a client blocked"
        ),
    )
    parser.add_argument(
        "--jitter",
        metavar="SECONDS",
        type=float,
        default=description.DEFAULT_JITTER,
        help="random extra added to each pause (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="refetch jobs that already have a description",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print nothing on success; failures still go to stderr",
    )
    return parser


def run(
    args: argparse.Namespace,
    *,
    out: IO[str] | None = None,
    err: IO[str] | None = None,
) -> int:
    """Run the fetch described by ``args``; return an exit code.

    Separate from :func:`main` so tests can drive it without touching
    ``sys.argv`` or a real process exit. ``out``/``err`` default to the live
    streams, resolved at call time rather than bound at definition time.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err

    if args.limit < 0:
        # A negative LIMIT is not "no limit" here: SQLite would read -1 as
        # unbounded, turning a bounded cron run into an open-ended one.
        print(f"{PROGRAM}: --limit must be 0 or greater", file=err)
        return EXIT_SETUP_FAILED

    db_path = core_db.resolve_db_path(args.db)
    try:
        conn = core_db.connect(db_path)
    except sqlite3.Error as error:
        print(f"{PROGRAM}: could not open {db_path}: {error}", file=err)
        return EXIT_SETUP_FAILED

    try:
        summary = description.fetch_descriptions(
            conn,
            limit=args.limit,
            delay=args.delay,
            jitter=args.jitter,
            force=args.force,
        )
        if not args.quiet:
            _report(summary, out=out)
        if summary.failed:
            for job_id, message in summary.failed:
                print(f"{PROGRAM}: job {job_id}: {message}", file=err)
            print(
                f"{PROGRAM}: {len(summary.failed)} description(s) failed",
                file=err,
            )
            return EXIT_DESCRIPTION_FAILED
        return EXIT_OK
    finally:
        conn.close()


def _report(summary: description.BatchSummary, *, out) -> None:
    """Print one summary line.

    Deliberately terse and flushed: under cron stdout is a pipe, so without the
    flush this line would sit in the buffer until the process exits.
    """
    print(
        f"described {summary.written} of {summary.attempted}"
        f" ({summary.written} written, {summary.skipped} skipped,"
        f" {len(summary.failed)} failed)",
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
