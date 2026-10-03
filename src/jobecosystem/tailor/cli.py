"""Command-line entry point for the resume tailor.

    jobecosystem-tailor validate [--overlay PATH]
    jobecosystem-tailor propose --job-id ID [--out PATH]
    jobecosystem-tailor diff    --overlay PATH
    jobecosystem-tailor render  --overlay PATH [--out PATH] --approve
    jobecosystem-tailor search  [--limit N]

``propose`` writes the overlay but renders nothing. ``render`` refuses to emit a
PDF without ``--approve``: the human approval gate is the whole point, so it is
an explicit flag rather than an implication of running the command.

Exit codes:

    0   success
    1   invalid base/overlay, or an operation failed
    2   could not start (missing file, bad database, missing dependency)
    3   render was asked for without --approve
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import IO, Sequence

from ..core import db as core_db
from . import adapter, diff as diff_mod, embed as embed_mod, overlay as overlay_mod
from .models import Resume, TailorError
from .render import render as render_pdf
from .resume import load as load_resume
from .resume import resolve_resume_path
from .validate import validate_overlay, validate_resume

PROGRAM = "jobecosystem-tailor"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_SETUP = 2
EXIT_NOT_APPROVED = 3

#: src/jobecosystem/tailor/cli.py -> tailor -> jobecosystem -> src -> root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OVERLAYS_DIR = _REPO_ROOT / "resumes"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM, description="Tailor the base resume to a job posting."
    )
    parser.add_argument("--resume", metavar="PATH", default=None,
                        help="base resume; defaults to $TAILOR_RESUME_FILE,"
                             " then resume.base.json at the repo root")
    parser.add_argument("--db", metavar="PATH", default=None,
                        help="database; defaults to $DB_PATH, then jobs.db at"
                             " the repo root")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate the base and/or an overlay")
    validate.add_argument("--overlay", metavar="PATH", default=None)

    propose = sub.add_parser("propose", help="ask the LLM for a tailored overlay")
    propose.add_argument("--job-id", type=int, required=True)
    propose.add_argument("--out", metavar="PATH", default=None)

    diff = sub.add_parser("diff", help="show what an overlay changes")
    diff.add_argument("--overlay", metavar="PATH", required=True)

    render = sub.add_parser("render", help="render an approved overlay to PDF")
    render.add_argument("--overlay", metavar="PATH", required=True)
    render.add_argument("--out", metavar="PATH", default=None)
    render.add_argument("--approve", action="store_true",
                        help="confirm you have reviewed the diff")

    search = sub.add_parser("search", help="jobs most similar to the resume")
    search.add_argument("--limit", type=int, default=25)
    search.add_argument("--min-score", type=float, default=0.0)
    return parser


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "resume"


def _load_job(conn: sqlite3.Connection, job_id: int) -> dict:
    row = conn.execute(
        "SELECT id, company, title, description, url FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        raise TailorError(f"no job with id {job_id}")
    return dict(row)


def _load_base(args) -> Resume:
    return load_resume(resolve_resume_path(args.resume))


def _load_and_apply(args):
    """Load the base and overlay, validate the overlay, return both + tailored."""
    base = _load_base(args)
    overlay = overlay_mod.load(args.overlay)
    errors = validate_overlay(base, overlay)
    if errors:
        raise TailorError("invalid overlay:\n  " + "\n  ".join(errors))
    return base, overlay, overlay_mod.apply(base, overlay)


def run(args, *, out: IO[str] | None = None, err: IO[str] | None = None) -> int:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    try:
        return _dispatch(args, out=out, err=err)
    except TailorError as error:
        print(f"{PROGRAM}: {error}", file=err)
        return EXIT_FAILED
    except sqlite3.Error as error:
        print(f"{PROGRAM}: database error: {error}", file=err)
        return EXIT_SETUP
    except FileNotFoundError as error:
        print(f"{PROGRAM}: {error}", file=err)
        return EXIT_SETUP


def _dispatch(args, *, out, err) -> int:
    if args.command == "validate":
        return _cmd_validate(args, out=out)
    if args.command == "propose":
        return _cmd_propose(args, out=out)
    if args.command == "diff":
        return _cmd_diff(args, out=out)
    if args.command == "render":
        return _cmd_render(args, out=out, err=err)
    if args.command == "search":
        return _cmd_search(args, out=out)
    raise TailorError(f"unknown command {args.command!r}")  # pragma: no cover


def _cmd_validate(args, *, out) -> int:
    base = _load_base(args)
    errors = validate_resume(base)
    if args.overlay:
        overlay = overlay_mod.load(args.overlay)
        errors += validate_overlay(base, overlay)
    if errors:
        for message in errors:
            print(f"{PROGRAM}: {message}", file=out)
        return EXIT_FAILED
    print("ok", file=out)
    return EXIT_OK


def _cmd_propose(args, *, out) -> int:
    base = _load_base(args)
    conn = core_db.connect(args.db)
    try:
        job = _load_job(conn, args.job_id)
        overlay = adapter.propose_overlay(
            base,
            company=job["company"],
            title=job["title"],
            description=job["description"],
            job_id=job["id"],
            complete=adapter.default_completer(),
        )
    finally:
        conn.close()

    path = Path(args.out) if args.out else (
        DEFAULT_OVERLAYS_DIR / f"{job['id']}-{_slug(job['company'])}.json"
    )
    overlay_mod.save(overlay, path)
    changes = diff_mod.diff(base, overlay_mod.apply(base, overlay))
    print(diff_mod.render_diff(changes), file=out)
    print(f"\nwrote {path}", file=out)
    return EXIT_OK


def _cmd_diff(args, *, out) -> int:
    base, _overlay, tailored = _load_and_apply(args)
    print(diff_mod.render_diff(diff_mod.diff(base, tailored)), file=out)
    return EXIT_OK


def _cmd_render(args, *, out, err) -> int:
    base, overlay, tailored = _load_and_apply(args)
    changes = diff_mod.diff(base, tailored)
    if not args.approve:
        print(diff_mod.render_diff(changes), file=out)
        print(
            f"\n{PROGRAM}: review the diff above, then re-run with --approve",
            file=err,
        )
        return EXIT_NOT_APPROVED

    path = Path(args.out) if args.out else Path(args.overlay).with_suffix(".pdf")
    written = render_pdf(tailored, path)
    target = overlay.target.title or "resume"
    if overlay.target.company:
        target = f"{target} @ {overlay.target.company}"
    print(f"rendered {target} -> {written}", file=out)
    return EXIT_OK


def _cmd_search(args, *, out) -> int:
    base = _load_base(args)
    conn = core_db.connect(args.db)
    try:
        hits = embed_mod.similar_jobs(
            conn, base, limit=args.limit, min_score=args.min_score
        )
    finally:
        conn.close()
    if not hits:
        print("no embedded jobs yet", file=out)
        return EXIT_OK
    for hit in hits:
        print(f"{hit.score:.4f}  {hit.job.id:>6}  {hit.job.title} @ {hit.job.company}",
              file=out)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
