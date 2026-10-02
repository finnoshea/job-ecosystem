"""Command-line entry point for the paced embedding batch.

Intended for cron, like ``jobecosystem-scrape`` and ``jobecosystem-describe``:
no prompts, no TTY assumptions, one terse summary, and an exit code the
scheduler can act on. It is a separate command because it needs the embedding
API and a ``VOYAGE_API_KEY``, which the other two deliberately do not.

    jobecosystem-embed --limit 500             # up to 500 descriptions this run
    jobecosystem-embed --db /tmp/jobs.db       # a different database
    jobecosystem-embed --quiet                 # just the exit code

The queue is every job that has a description but no embedding for the requested
model, ordered by id, so an interrupted run resumes where the last one stopped
and switching models re-embeds what the old model produced.

Exit codes:

    0   every attempted description was embedded (including "nothing to do")
    1   at least one embedding failed (the rest still happened)
    2   could not start: bad arguments, unopenable database
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import IO, Sequence

from ..core import db as core_db
from ..core import embedder
from . import embed as embed_batch

PROGRAM = "jobecosystem-embed"

EXIT_OK = 0
EXIT_EMBEDDING_FAILED = 1
EXIT_SETUP_FAILED = 2


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Embed pending job descriptions into the SQLite database.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        metavar="N",
        type=int,
        default=200,
        help=(
            "maximum descriptions to embed this run (default: 200); bounds how"
            " long a scheduled run takes"
        ),
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="database file; defaults to $DB_PATH, then jobs.db at the repo root",
    )
    parser.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=32,
        help="descriptions per model call (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        metavar="ID",
        default=embedder.DEFAULT_MODEL,
        help="embedding model id (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed jobs that already have a vector for this model",
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
    embed=None,
) -> int:
    """Run the embedding described by ``args``; return an exit code.

    Separate from :func:`main` so tests can drive it with a stub ``embed``
    instead of loading the model. ``out``/``err`` default to the live streams,
    resolved at call time rather than bound at definition time.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err

    if args.limit < 0:
        print(f"{PROGRAM}: --limit must be 0 or greater", file=err)
        return EXIT_SETUP_FAILED
    if args.batch_size < 1:
        print(f"{PROGRAM}: --batch-size must be at least 1", file=err)
        return EXIT_SETUP_FAILED

    db_path = core_db.resolve_db_path(args.db)
    try:
        conn = core_db.connect(db_path)
    except sqlite3.Error as error:
        print(f"{PROGRAM}: could not open {db_path}: {error}", file=err)
        return EXIT_SETUP_FAILED

    if embed is None:
        # A missing key is a setup problem, not a per-job failure: fail before
        # touching the database queue so the exit code says "cannot start".
        if not os.environ.get("VOYAGE_API_KEY"):
            print(f"{PROGRAM}: VOYAGE_API_KEY is not set", file=err)
            return EXIT_SETUP_FAILED
        embed = _model_embedder(args.model, args.batch_size)

    try:
        embedder.reset_usage()
        summary = embed_batch.embed_pending(
            conn,
            embed,
            limit=args.limit,
            batch_size=args.batch_size,
            model=args.model,
            force=args.force,
        )
        if not args.quiet:
            _report(summary, tokens=embedder.total_tokens_used(), out=out)
        if summary.failed:
            for job_id, message in summary.failed:
                print(f"{PROGRAM}: job {job_id}: {message}", file=err)
            print(f"{PROGRAM}: {len(summary.failed)} embedding(s) failed", file=err)
            return EXIT_EMBEDDING_FAILED
        return EXIT_OK
    finally:
        conn.close()


def _model_embedder(model_id: str, batch_size: int):
    """The real embedder, bound to a model and mini-batch size."""

    def embed(texts):
        return embedder.embed_documents(
            texts, model_id=model_id, batch_size=batch_size
        )

    # Recorded in job_embeddings.model so a later model change re-embeds.
    embed.model_name = model_id
    return embed


def _report(summary: embed_batch.EmbedSummary, *, tokens: int, out) -> None:
    """Print one summary line, flushed for cron's piped stdout."""
    print(
        f"embedded {summary.written} of {summary.attempted}"
        f" ({summary.written} written, {len(summary.failed)} failed,"
        f" {tokens} tokens)",
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
