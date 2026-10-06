"""Tests for the shared ``*_companies.txt`` parser.

Ashby, Greenhouse, Lever and SmartRecruiters all read the same line shape, so
the tokenizing behaviour is tested once here; each source's tests cover only how
it maps the tokens onto its own spec.
"""

from __future__ import annotations

from jobecosystem.ingest.sources import companies


def test_parse_lines_reads_slug_and_fields():
    outcome = companies.parse_lines("acme Acme Corp\nbeta\n")
    assert [(entry.slug, entry.fields) for entry in outcome.entries] == [
        ("acme", ("Acme", "Corp")),
        ("beta", ()),
    ]
    assert outcome.entries[0].name == "Acme Corp"
    assert outcome.entries[1].name is None


def test_comments_and_blank_lines_are_ignored():
    outcome = companies.parse_lines("# header\n\n  \n\t\nacme  # inline\n")
    assert [entry.slug for entry in outcome.entries] == ["acme"]
    assert outcome.entries[0].name is None


def test_duplicates_keep_the_first_occurrence():
    outcome = companies.parse_lines("acme Acme Corp\nacme\nacme Other\n")
    assert [(entry.slug, entry.name) for entry in outcome.entries] == [
        ("acme", "Acme Corp")
    ]


def test_order_is_preserved():
    outcome = companies.parse_lines("c\na\nb\n")
    assert [entry.slug for entry in outcome.entries] == ["c", "a", "b"]


def test_quoted_names_are_one_field():
    outcome = companies.parse_lines('acme "Acme Corporation"\n')
    assert outcome.entries[0].name == "Acme Corporation"


def test_unbalanced_quotes_are_recorded_not_raised():
    outcome = companies.parse_lines('acme "Unclosed\nbeta\n')
    assert [entry.slug for entry in outcome.entries] == ["beta"]
    assert len(outcome.errors) == 1
    assert outcome.errors[0][0] == 1


def test_empty_text_yields_nothing():
    outcome = companies.parse_lines("\n\n")
    assert not outcome
    assert len(outcome) == 0
    assert outcome.errors == []


def test_resolve_file_prefers_the_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_COMPANIES_FILE", str(tmp_path / "env.txt"))
    assert companies.resolve_file(
        tmp_path / "given.txt", env_var="TEST_COMPANIES_FILE",
        default=tmp_path / "default.txt",
    ) == tmp_path / "given.txt"


def test_resolve_file_uses_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_COMPANIES_FILE", str(tmp_path / "env.txt"))
    assert companies.resolve_file(
        None, env_var="TEST_COMPANIES_FILE", default=tmp_path / "default.txt"
    ) == tmp_path / "env.txt"


def test_resolve_file_falls_back_to_the_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_COMPANIES_FILE", raising=False)
    default = tmp_path / "default.txt"
    assert companies.resolve_file(
        None, env_var="TEST_COMPANIES_FILE", default=default
    ) == default


def test_load_lines_missing_file_is_empty(tmp_path):
    outcome = companies.load_lines(
        tmp_path / "nope.txt", env_var="TEST_COMPANIES_FILE",
        default=tmp_path / "default.txt",
    )
    assert not outcome
    assert outcome.errors == []


def test_load_lines_reads_a_file(tmp_path):
    path = tmp_path / "companies.txt"
    path.write_text("acme Acme Corp\n", encoding="utf-8")
    outcome = companies.load_lines(
        path, env_var="TEST_COMPANIES_FILE", default=tmp_path / "default.txt"
    )
    assert [entry.slug for entry in outcome.entries] == ["acme"]
    assert outcome.entries[0].name == "Acme Corp"
