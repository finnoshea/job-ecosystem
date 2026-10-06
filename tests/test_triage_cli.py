"""Tests for the triage command-line entry point.

Only the non-interactive surface is covered here: the parser and the wiring that
hands options to the TUI. The interface itself is exercised in ``test_tui``.
"""

from __future__ import annotations

import pytest

from jobecosystem.triage import cli

pytest.importorskip("textual")


def test_parser_defaults_resume_to_none():
    assert cli.build_parser().parse_args([]).resume is None


def test_parser_accepts_a_resume_path():
    args = cli.build_parser().parse_args(["--resume", "my-resume.json"])
    assert args.resume == "my-resume.json"


def test_run_forwards_the_resume_path(db_path, monkeypatch):
    # run() imports the TUI lazily, so the module attribute is what to patch.
    import jobecosystem.triage.tui.app as tui_app

    seen = {}

    def fake_run(path, *, page_size, resume_path=None):
        seen["resume_path"] = resume_path
        return 0

    monkeypatch.setattr(tui_app, "run", fake_run)
    args = cli.build_parser().parse_args(
        ["--db", str(db_path), "--resume", "cv.json"]
    )
    assert cli.run(args) == 0
    assert seen["resume_path"] == "cv.json"
