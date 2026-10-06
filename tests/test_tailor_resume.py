"""Tests for jobecosystem.tailor.resume: load, serialize, hash, query text."""

from __future__ import annotations

import pytest

from jobecosystem.tailor import resume as R
from jobecosystem.tailor.models import TailorError
from jobecosystem.tailor.validate import validate_resume


def test_round_trips_through_dict(sample_resume):
    assert R.resume_from_dict(R.resume_to_dict(sample_resume)) == sample_resume


def test_publications_round_trip(sample_resume):
    data = R.resume_to_dict(sample_resume)
    assert data["publications"][0].startswith("**A Paper**")
    assert R.resume_from_dict(data).publications == sample_resume.publications


def test_non_string_publication_is_rejected():
    with pytest.raises(TailorError, match="publications"):
        R.resume_from_dict({"basics": {"name": "X"},
                            "publications": [{"id": "pub"}]})


def test_canonical_json_is_stable_and_sorted(sample_resume):
    text = R.canonical_json(R.resume_to_dict(sample_resume))
    assert text.endswith("\n")
    assert text == R.canonical_json(R.resume_to_dict(sample_resume))


def test_hash_is_stable(sample_resume):
    assert R.resume_hash(sample_resume) == R.resume_hash(sample_resume)
    assert R.resume_hash(sample_resume).startswith("sha256:")


def test_hash_changes_when_content_changes(sample_resume):
    before = R.resume_hash(sample_resume)
    sample_resume.basics.summary = "A different summary."
    assert R.resume_hash(sample_resume) != before


def test_save_and_load_round_trip(sample_resume, tmp_path):
    path = tmp_path / "resume.json"
    R.save(sample_resume, path)
    assert R.load(path) == sample_resume


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(TailorError, match="not found"):
        R.load(tmp_path / "nope.json")


def test_load_invalid_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TailorError, match="not valid JSON"):
        R.load(path)


def test_resolve_prefers_the_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("TAILOR_RESUME_FILE", str(tmp_path / "env.json"))
    assert R.resolve_resume_path(tmp_path / "given.json") == tmp_path / "given.json"


def test_resolve_uses_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TAILOR_RESUME_FILE", str(tmp_path / "env.json"))
    assert R.resolve_resume_path() == tmp_path / "env.json"


def test_resolve_defaults_to_the_repo_root(monkeypatch):
    monkeypatch.delenv("TAILOR_RESUME_FILE", raising=False)
    assert R.resolve_resume_path() == R.DEFAULT_RESUME_FILE
    assert R.DEFAULT_RESUME_FILE.name == "resume.base.json"


def test_query_text_includes_the_signal(sample_resume):
    text = R.query_text(sample_resume)
    assert "Backend Engineer" in text
    assert "Python" in text
    assert "Senior Engineer at One Corp" in text
    assert "Cut latency 40% using caches." in text


# ---------------------------------------------------------------------------
# shape errors
# ---------------------------------------------------------------------------

def test_missing_name_is_rejected():
    with pytest.raises(TailorError, match="basics.name"):
        R.resume_from_dict({"basics": {}})


def test_bullets_must_be_a_list():
    data = {"basics": {"name": "X"}, "roles": [
        {"id": "r", "company": "C", "title": "T", "bullets": {"not": "a list"}}]}
    with pytest.raises(TailorError, match="bullets"):
        R.resume_from_dict(data)


# ---------------------------------------------------------------------------
# the shipped file
# ---------------------------------------------------------------------------

def test_the_shipped_resume_loads_and_validates():
    base = R.load()
    assert base.basics.name
    assert validate_resume(base) == []
    assert base.roles and all(role.bullets for role in base.roles)
