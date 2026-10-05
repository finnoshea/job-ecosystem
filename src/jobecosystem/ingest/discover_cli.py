"""Command-line entry point for ATS-board discovery.

Grows the ``*_companies.txt`` files from job URLs found on Hacker News. No ATS
is contacted, and no database is touched: the command reads text, extracts
slugs, and appends them to the lists the scrapers already read.

    jobecosystem-discover                      # every source, every family
    jobecosystem-discover --from hn            # just the hiring thread
    jobecosystem-discover --family lever,workable
    jobecosystem-discover --dry-run            # print additions, write nothing

A source that fails (an API down, a changed payload) is reported and skipped;
the remaining sources still run. Exit codes:

    0   every source was read (including "nothing new")
    1   at least one source failed; the others still ran
    2   could not start: unknown --from/--family value
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import IO, Sequence

from . import discover

PROGRAM = "jobecosystem-discover"

EXIT_OK = 0
EXIT_SOURCE_FAILED = 1
EXIT_SETUP_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Discover ATS company slugs from Hacker News.",
    )
    parser.add_argument(
        "--from",
        dest="sources",
        metavar="NAME[,NAME...]",
        default="all",
        help="sources to read (default: all): "
        + ", ".join(discover.SOURCES),
    )
    parser.add_argument(
        "--family",
        metavar="NAME[,NAME...]",
        default="all",
        help="ATS families to write (default: all): "
        + ", ".join(family.key for family in discover.FAMILIES),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be added without writing anything",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="write the companies files here instead of the repo root"
        " (mainly for testing or staging a review)",
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
    fetch_json: discover.FetchJson | None = None,
) -> int:
    """Run the discovery described by ``args``; return an exit code.

    ``fetch_json`` is injected for tests, which supply saved payloads instead of
    hitting the network. ``out``/``err`` default to the live streams, resolved at
    call time rather than bound at definition time.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    fetcher = fetch_json or discover.http_get_json

    sources = _split(args.sources) or []
    if "all" in sources:
        sources = list(discover.SOURCES)
    unknown = [name for name in sources if name not in discover.SOURCES]
    if unknown:
        print(f"{PROGRAM}: unknown source(s): {', '.join(unknown)}", file=err)
        return EXIT_SETUP_FAILED

    families = _split(args.family)
    if "all" in families:
        families = [family.key for family in discover.FAMILIES]
    unknown = [name for name in families if name not in discover.FAMILIES_BY_KEY]
    if unknown:
        print(f"{PROGRAM}: unknown family(s): {', '.join(unknown)}", file=err)
        return EXIT_SETUP_FAILED

    target_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else discover.REPO_ROOT
    )

    combined = discover.Harvest()
    failures: list[tuple[str, str]] = []
    for source in sources:
        try:
            combined.merge(discover.harvest(source, fetcher))
        except Exception as error:  # noqa: BLE001 - one dead source is not fatal
            failures.append((source, str(error)))

    outcomes = []
    for name in families:
        family = discover.FAMILIES_BY_KEY[name]
        outcomes.append(discover.merge_into_file(
            target_dir / family.companies_file,
            combined.slugs.get(name, set()),
            names=combined.names.get(name, {}),
            dry_run=args.dry_run,
        ))

    if not args.quiet:
        for outcome in outcomes:
            if outcome.added:
                verb = "would add" if args.dry_run else "added"
                print(
                    f"{outcome.path.name}: {verb} {len(outcome.added)}"
                    f" ({outcome.existing} already present)",
                    file=out,
                    flush=True,
                )
            else:
                print(f"{outcome.path.name}: no new slugs", file=out, flush=True)
        for source, message in failures:
            print(f"{PROGRAM}: {source}: {message}", file=err, flush=True)

    return EXIT_SOURCE_FAILED if failures else EXIT_OK


def _split(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
