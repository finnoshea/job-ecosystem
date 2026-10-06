"""Shared fixtures for the jobecosystem test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src/` importable without requiring `pip install -e .` first. This is the
# one place that reaches outside the installed package; everything else imports
# `jobecosystem` normally.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A filesystem path to a throwaway database, in pytest's tmp dir."""
    return tmp_path / "jobs.db"


@pytest.fixture
def conn(db_path: Path):
    """An open, migrated connection to a throwaway database.

    Closed on teardown. Tests that need the raw path (to check file creation,
    WAL artifacts, or reopen behavior) should use ``db_path`` instead.
    """
    from jobecosystem.core import db

    connection = db.connect(db_path)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def sample_resume():
    """A small, self-contained base resume for the tailor tests.

    Deliberately not the shipped ``resume.base.json``: tests that care about the
    shipped file load it explicitly, and everything else should not break when
    the fake resume is edited.
    """
    from jobecosystem.tailor import models as m

    return m.Resume(
        basics=m.Basics(
            name="Test Person",
            headline="Backend Engineer",
            email="test@example.com",
            phone="+1-512-555-0000",
            location="Remote",
            links=[m.Link(label="GitHub", url="https://github.com/tp"),
                   m.Link(label="LinkedIn", url="https://linkedin.com/in/tp")],
            summary="Original summary.",
        ),
        skills=[
            m.Skill(id="skill.python", name="Python", category="Languages",
                    keywords=["asyncio"]),
            m.Skill(id="skill.sql", name="SQL", category="Languages"),
            m.Skill(id="skill.aws", name="AWS", category="Cloud"),
        ],
        roles=[
            m.Role(
                id="role.one", company="One Corp", title="Senior Engineer",
                location="Seattle, WA", start="2020-01", end=None,
                bullets=[
                    m.Bullet(id="b.one.1", text="Cut latency 40% using caches.",
                             metrics=["40%"], tags=["perf"]),
                    m.Bullet(id="b.one.2", text="Led 5 engineers on a migration.",
                             metrics=["5"], tags=["lead"]),
                ],
            ),
            m.Role(
                id="role.two", company="Two Inc", title="Engineer",
                start="2017-01", end="2019-12",
                bullets=[m.Bullet(id="b.two.1",
                                  text="Built a service handling 2M requests/day.",
                                  metrics=["2M"])],
            ),
        ],
        projects=[m.Project(
            id="proj.job", name="Job Ecosystem",
            url="https://github.com/tp/job-ecosystem",
            description="A pipeline.",
            bullets=[m.Bullet(id="b.proj.1", text="Built it.")],
        )],
        education=[m.Education(id="edu.x", institution="State U", degree="B.S.",
                               field_of_study="CS", start="2013-08", end="2017-05")],
        certifications=[m.Certification(id="cert.x", name="Cert", issuer="Issuer",
                                        date="2021-01", number="CERT-123")],
        publications=["**A Paper** — A. Author, *Journal* (2022)."],
    )
