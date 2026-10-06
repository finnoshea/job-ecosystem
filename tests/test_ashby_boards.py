"""Tests for jobecosystem.ingest.sources.ashby_boards: the editable board list."""

from __future__ import annotations

from jobecosystem.ingest.sources import ashby_boards as boards
from jobecosystem.ingest.sources.ashby import AshbyScraper


# ---------------------------------------------------------------------------
# parse_boards
# ---------------------------------------------------------------------------

def test_parses_a_bare_slug():
    (spec,) = boards.parse_boards("ramp")
    assert spec.slug == "ramp"
    assert spec.company is None


def test_parses_a_slug_with_a_display_name():
    (spec,) = boards.parse_boards("ironcladhq  Ironclad")
    assert spec.slug == "ironcladhq"
    assert spec.company == "Ironclad"


def test_display_name_may_contain_spaces():
    (spec,) = boards.parse_boards("slug  Multi Word Name")
    assert spec.company == "Multi Word Name"


def test_blank_lines_are_ignored():
    assert boards.parse_boards("\n\n  \n\t\n") == []


def test_comment_lines_are_ignored():
    assert boards.parse_boards("# a comment\n#another\n") == []


def test_trailing_comments_are_stripped():
    (spec,) = boards.parse_boards("ramp  # engine board")
    assert spec.slug == "ramp"
    assert spec.company is None


def test_trailing_comment_after_a_display_name():
    (spec,) = boards.parse_boards("dave  Dave  # fintech")
    assert spec.slug == "dave"
    assert spec.company == "Dave"


def test_comments_and_blanks_mix_with_entries():
    text = """
    # Ashby boards

    ramp
    snowflake  Snowflake

    # done
    """
    specs = boards.parse_boards(text)
    assert [s.slug for s in specs] == ["ramp", "snowflake"]
    assert specs[1].company == "Snowflake"


def test_leading_and_trailing_whitespace_is_ignored():
    (spec,) = boards.parse_boards("   spaced-slug   Spaced Co   ")
    assert spec.slug == "spaced-slug"
    assert spec.company == "Spaced Co"


def test_duplicate_slugs_are_collapsed():
    specs = boards.parse_boards("ramp\nramp\nramp")
    assert [s.slug for s in specs] == ["ramp"]


def test_first_occurrence_wins_so_a_display_name_is_kept():
    # A later bare duplicate must not erase an earlier display name.
    specs = boards.parse_boards("ramp  Ramp Inc\nramp")
    assert specs[0].company == "Ramp Inc"


def test_order_is_preserved():
    specs = boards.parse_boards("c\na\nb")
    assert [s.slug for s in specs] == ["c", "a", "b"]


def test_quoted_display_name_is_unquoted():
    (spec,) = boards.parse_boards('slug  "Quoted Name"  # note')
    assert spec.company == "Quoted Name"


def test_unicode_display_name_survives():
    (spec,) = boards.parse_boards("slug  Ünicode Näme")
    assert spec.company == "Ünicode Näme"


def test_source_label_is_prefixed_with_ashby():
    (spec,) = boards.parse_boards("ramp")
    assert spec.source == "ashby:ramp"


def test_hash_only_line_is_a_comment():
    assert boards.parse_boards("#") == []


# ---------------------------------------------------------------------------
# resolve_boards_file
# ---------------------------------------------------------------------------

def test_resolve_prefers_an_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHBY_BOARDS_FILE", str(tmp_path / "env.txt"))
    assert boards.resolve_boards_file(tmp_path / "given.txt") == tmp_path / "given.txt"


def test_resolve_uses_the_environment_when_given(tmp_path, monkeypatch):
    monkeypatch.setenv("ASHBY_BOARDS_FILE", str(tmp_path / "env.txt"))
    assert boards.resolve_boards_file() == tmp_path / "env.txt"


def test_resolve_falls_back_to_the_repo_default(monkeypatch):
    monkeypatch.delenv("ASHBY_BOARDS_FILE", raising=False)
    assert boards.resolve_boards_file() == boards.DEFAULT_BOARDS_FILE
    assert boards.DEFAULT_BOARDS_FILE.name == "ashby_boards.txt"


def test_resolve_expands_user(monkeypatch):
    monkeypatch.delenv("ASHBY_BOARDS_FILE", raising=False)
    assert "~" not in str(boards.resolve_boards_file("~/boards.txt"))


def test_default_file_lives_at_the_repo_root():
    # Repo root, next to pyproject.toml and README.md -- not inside src/, so it
    # stays editable after an install.
    assert boards.DEFAULT_BOARDS_FILE.parent.name == "jobecosystem"
    assert boards.DEFAULT_BOARDS_FILE.parent.name != "src"
    assert boards.DEFAULT_BOARDS_FILE.name == "ashby_boards.txt"


# ---------------------------------------------------------------------------
# load_boards
# ---------------------------------------------------------------------------

def test_load_reads_from_a_file(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("ramp\nsnowflake  Snowflake\n", encoding="utf-8")
    specs = boards.load_boards(path)
    assert [s.slug for s in specs] == ["ramp", "snowflake"]
    assert specs[1].company == "Snowflake"


def test_load_returns_empty_for_a_missing_file(tmp_path):
    # A missing optional config file means "nothing configured", not a crash.
    assert boards.load_boards(tmp_path / "nope.txt") == []


def test_load_returns_empty_for_an_empty_file(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("", encoding="utf-8")
    assert boards.load_boards(path) == []


def test_load_handles_comments_only(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("# nothing here yet\n\n", encoding="utf-8")
    assert boards.load_boards(path) == []


def test_load_uses_the_environment_variable(tmp_path, monkeypatch):
    path = tmp_path / "from-env.txt"
    path.write_text("ramp\n", encoding="utf-8")
    monkeypatch.setenv("ASHBY_BOARDS_FILE", str(path))
    assert [s.slug for s in boards.load_boards()] == ["ramp"]


def test_the_shipped_file_is_loadable():
    specs = boards.load_boards()
    assert specs, "ashby_boards.txt should list at least one board"
    assert len({s.slug for s in specs}) == len(specs)


def test_the_shipped_file_has_no_malformed_lines():
    for spec in boards.load_boards():
        assert spec.slug == spec.slug.strip()
        assert " " not in spec.slug
        assert not spec.slug.startswith("#")


# ---------------------------------------------------------------------------
# build_scrapers
# ---------------------------------------------------------------------------

def test_build_creates_one_scraper_per_board(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("ramp\nsnowflake  Snowflake\n", encoding="utf-8")
    scrapers = boards.build_scrapers(path)
    assert len(scrapers) == 2
    assert all(isinstance(s, AshbyScraper) for s in scrapers)


def test_built_scrapers_carry_slug_and_company(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("ironcladhq  Ironclad\n", encoding="utf-8")
    (scraper,) = boards.build_scrapers(path)
    assert scraper.board == "ironcladhq"
    assert scraper.company == "Ironclad"
    assert scraper.source == "ashby:ironcladhq"


def test_built_scraper_falls_back_to_the_slug_as_company(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("ramp\n", encoding="utf-8")
    (scraper,) = boards.build_scrapers(path)
    assert scraper.company == "ramp"


def test_build_returns_empty_for_a_missing_file(tmp_path):
    assert boards.build_scrapers(tmp_path / "nope.txt") == []


def test_build_passes_fetch_json_through(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("ramp\n", encoding="utf-8")

    def stub(url):
        return {"apiVersion": "1", "jobs": []}

    (scraper,) = boards.build_scrapers(path, fetch_json=stub)
    assert scraper.fetch() == []


def test_built_scrapers_have_distinct_source_labels(tmp_path):
    path = tmp_path / "boards.txt"
    path.write_text("ramp\nsnowflake\nnotion\n", encoding="utf-8")
    sources = [s.source for s in boards.build_scrapers(path)]
    assert sources == ["ashby:ramp", "ashby:snowflake", "ashby:notion"]
    assert len(set(sources)) == 3


# ---------------------------------------------------------------------------
# integration with the runner
# ---------------------------------------------------------------------------

def test_scrapers_from_the_file_run_through_the_runner(conn, tmp_path):
    from jobecosystem.ingest import runner

    path = tmp_path / "boards.txt"
    path.write_text("ramp\nnotion\n", encoding="utf-8")

    payloads = {
        "ramp": {"jobs": [{"id": "r1", "title": "Engineer", "isListed": True}]},
        "notion": {"jobs": [{"id": "n1", "title": "Designer", "isListed": True}]},
    }

    def stub(url):
        return payloads[url.rsplit("/", 1)[-1]]

    summary = runner.run_scrapers(conn, boards.build_scrapers(path, fetch_json=stub))
    assert [o.source for o in summary] == ["ashby:ramp", "ashby:notion"]
    assert summary.ok
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 2


def test_a_missing_board_file_is_not_a_crash(conn, tmp_path):
    from jobecosystem.ingest import runner

    summary = runner.run_scrapers(conn, boards.build_scrapers(tmp_path / "nope.txt"))
    assert len(summary) == 0
    assert summary.ok
