"""Tests for jobecosystem.tailor.cli: the flow driver and approval gate."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobecosystem.core.models import Job
from jobecosystem.ingest import upsert
from jobecosystem.tailor import cli
from jobecosystem.tailor import overlay as O
from jobecosystem.tailor import resume as R


@pytest.fixture
def resume_file(sample_resume, tmp_path):
    path = tmp_path / "resume.base.json"
    R.save(sample_resume, path)
    return path


def parse(argv):
    return cli.build_parser().parse_args(argv)


def add_job(conn):
    result = upsert.upsert_job(
        conn,
        Job(source="ashby:ramp", external_id="1", company="Acme",
            title="Data Engineer", description="Build pipelines with Python."),
    )
    conn.commit()
    return result.job_id


def write_overlay(sample_resume, path, **overrides):
    fields = {"base_hash": R.resume_hash(sample_resume)}
    fields.update(overrides)
    O.save(O.Overlay(**fields), path)
    return path


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_ok(resume_file):
    out = io.StringIO()
    code = cli.run(parse(["--resume", str(resume_file), "validate"]), out=out)
    assert code == cli.EXIT_OK
    assert out.getvalue().strip() == "ok"


def test_validate_reports_a_bad_overlay(resume_file, sample_resume, tmp_path):
    overlay = write_overlay(sample_resume, tmp_path / "o.json",
                            base_hash="sha256:" + "0" * 64)
    out = io.StringIO()
    code = cli.run(
        parse(["--resume", str(resume_file), "validate", "--overlay", str(overlay)]),
        out=out,
    )
    assert code == cli.EXIT_FAILED
    assert "base_hash" in out.getvalue()


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_prints_the_changes(resume_file, sample_resume, tmp_path):
    overlay = write_overlay(sample_resume, tmp_path / "o.json", summary="Tailored.")
    out = io.StringIO()
    code = cli.run(
        parse(["--resume", str(resume_file), "diff", "--overlay", str(overlay)]),
        out=out,
    )
    assert code == cli.EXIT_OK
    assert "[summary]" in out.getvalue()


# ---------------------------------------------------------------------------
# render / approval gate
# ---------------------------------------------------------------------------

def test_render_requires_approval_and_writes_nothing(resume_file, sample_resume,
                                                     tmp_path):
    overlay = write_overlay(sample_resume, tmp_path / "o.json", summary="Tailored.")
    pdf = tmp_path / "out.pdf"
    err = io.StringIO()
    code = cli.run(
        parse(["--resume", str(resume_file), "render", "--overlay", str(overlay),
               "--out", str(pdf)]),
        out=io.StringIO(), err=err,
    )
    assert code == cli.EXIT_NOT_APPROVED
    assert not pdf.exists()
    assert "--approve" in err.getvalue()


def test_render_with_approval_writes_a_pdf(resume_file, sample_resume, tmp_path,
                                           monkeypatch):
    overlay = write_overlay(sample_resume, tmp_path / "o.json", summary="Tailored.")
    pdf = tmp_path / "out.pdf"

    def fake_render(resume, path, **kwargs):
        Path(path).write_text("pdf", encoding="utf-8")
        return Path(path)

    monkeypatch.setattr(cli, "render_pdf", fake_render)
    out = io.StringIO()
    code = cli.run(
        parse(["--resume", str(resume_file), "render", "--overlay", str(overlay),
               "--out", str(pdf), "--approve"]),
        out=out,
    )
    assert code == cli.EXIT_OK
    assert pdf.exists()
    assert "rendered" in out.getvalue()


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------

def test_propose_writes_a_validated_overlay(resume_file, conn, db_path, tmp_path,
                                            monkeypatch):
    job_id = add_job(conn)
    completion = json.dumps({"summary": "Tailored summary.",
                             "skill_ids": ["skill.python"]})
    monkeypatch.setattr(cli.adapter, "default_completer",
                        lambda: (lambda prompt: completion))
    overlay_path = tmp_path / "proposed.json"
    out = io.StringIO()
    code = cli.run(
        parse(["--resume", str(resume_file), "--db", str(db_path), "propose",
               "--job-id", str(job_id), "--out", str(overlay_path)]),
        out=out,
    )
    assert code == cli.EXIT_OK
    assert overlay_path.exists()
    assert O.load(overlay_path).target.job_id == job_id
    assert "wrote" in out.getvalue()


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_prints_hits(resume_file, db_path, monkeypatch):
    hits = [SimpleNamespace(score=0.91,
                            job=SimpleNamespace(id=3, title="Data Engineer",
                                                company="Acme"))]
    monkeypatch.setattr(cli.embed_mod, "similar_jobs", lambda *a, **k: hits)
    out = io.StringIO()
    code = cli.run(
        parse(["--resume", str(resume_file), "--db", str(db_path), "search"]),
        out=out,
    )
    assert code == cli.EXIT_OK
    assert "0.9100" in out.getvalue()
    assert "Data Engineer @ Acme" in out.getvalue()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def test_main_is_callable_without_arguments():
    import inspect

    assert not [
        p for p in inspect.signature(cli.main).parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
