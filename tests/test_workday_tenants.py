"""Tests for jobecosystem.ingest.sources.workday_tenants: the editable site list."""

from __future__ import annotations

from jobecosystem.ingest.sources import workday_tenants as wt
from jobecosystem.ingest.sources.workday import WorkdayScraper


# ---------------------------------------------------------------------------
# parse_tenants
# ---------------------------------------------------------------------------

def test_parses_the_three_required_fields():
    outcome = wt.parse_tenants("nvidia.wd5 nvidia NVIDIAExternalCareerSite")
    (spec,) = outcome.specs
    assert (spec.host, spec.tenant, spec.site) == (
        "nvidia.wd5", "nvidia", "NVIDIAExternalCareerSite"
    )
    assert spec.company is None
    assert outcome.errors == []


def test_parses_an_optional_display_name():
    (spec,) = wt.parse_tenants("asml.wd3 asml ASMLPrivate1 ASML").specs
    assert spec.company == "ASML"


def test_multi_word_display_name_is_joined():
    (spec,) = wt.parse_tenants("h.wd1 t Site Acme Corporation Ltd").specs
    assert spec.company == "Acme Corporation Ltd"


def test_display_name_may_contain_parentheses():
    (spec,) = wt.parse_tenants("asml.wd3 asml ASMLEXT1 ASML (external)").specs
    assert spec.company == "ASML (external)"


def test_blank_lines_are_ignored():
    assert wt.parse_tenants("\n\n   \n\t").specs == []


def test_comment_lines_are_ignored():
    assert wt.parse_tenants("# comment\n# another").specs == []


def test_trailing_comments_are_stripped():
    (spec,) = wt.parse_tenants("h.wd1 t Site  # note").specs
    assert (spec.host, spec.company) == ("h.wd1", None)


def test_commented_out_lines_are_ignored():
    text = "# h.wd1 t Site\nnvidia.wd5 nvidia NVIDIAExternalCareerSite"
    assert [s.host for s in wt.parse_tenants(text).specs] == ["nvidia.wd5"]


def test_whitespace_columns_align_without_effect():
    aligned = "nvidia.wd5       nvidia      NVIDIAExternalCareerSite    Nvidia"
    (spec,) = wt.parse_tenants(aligned).specs
    assert spec.host == "nvidia.wd5"
    assert spec.company == "Nvidia"


def test_duplicate_sites_collapse_to_one():
    text = "h.wd1 t Site\nh.wd1 t Site\nh.wd1 t Site"
    assert len(wt.parse_tenants(text).specs) == 1


def test_the_same_host_with_different_sites_stays_separate():
    # One tenant often exposes several sites; each is its own source.
    text = "sf.wd12 sf Main\nsf.wd12 sf Slack"
    assert [s.label for s in wt.parse_tenants(text).specs] == [
        "workday:sf:Main", "workday:sf:Slack"
    ]


def test_short_lines_are_reported_not_raised():
    outcome = wt.parse_tenants("h.wd1 t\nnvidia.wd5 nvidia Site")
    assert [s.host for s in outcome.specs] == ["nvidia.wd5"]
    (number, message), = outcome.errors
    assert number == 1
    assert "3" in message or "field" in message


def test_unbalanced_quotes_are_reported():
    outcome = wt.parse_tenants('h.wd1 t Site "Unclosed')
    assert outcome.specs == []
    (number, message), = outcome.errors
    assert number == 1
    assert "quotation" in message.lower() or "parse" in message.lower()


def test_errors_do_not_stop_later_lines():
    text = "bad line\nnvidia.wd5 nvidia Site Nvidia"
    outcome = wt.parse_tenants(text)
    assert [s.host for s in outcome.specs] == ["nvidia.wd5"]
    assert len(outcome.errors) == 1


def test_order_is_preserved():
    text = "c.wd1 t S\na.wd1 t S\nb.wd1 t S"
    assert [s.host for s in wt.parse_tenants(text).specs] == ["c.wd1", "a.wd1", "b.wd1"]


def test_label_and_listing_url_are_built():
    (spec,) = wt.parse_tenants("nvidia.wd5 nvidia NVIDIAExternalCareerSite").specs
    assert spec.label == "workday:nvidia:NVIDIAExternalCareerSite"
    assert spec.listing_url == (
        "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia"
        "/NVIDIAExternalCareerSite/jobs"
    )


def test_outcome_truthiness_reflects_content():
    assert bool(wt.parse_tenants("h.wd1 t S")) is True
    assert bool(wt.parse_tenants("# nothing")) is False
    assert len(wt.parse_tenants("h.wd1 t S\nh2.wd1 t S")) == 2


# ---------------------------------------------------------------------------
# resolve_tenants_file
# ---------------------------------------------------------------------------

def test_resolve_prefers_an_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKDAY_TENANTS_FILE", str(tmp_path / "env.txt"))
    assert wt.resolve_tenants_file(tmp_path / "given.txt") == tmp_path / "given.txt"


def test_resolve_uses_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKDAY_TENANTS_FILE", str(tmp_path / "env.txt"))
    assert wt.resolve_tenants_file() == tmp_path / "env.txt"


def test_resolve_falls_back_to_the_repo_default(monkeypatch):
    monkeypatch.delenv("WORKDAY_TENANTS_FILE", raising=False)
    assert wt.resolve_tenants_file() == wt.DEFAULT_TENANTS_FILE
    assert wt.DEFAULT_TENANTS_FILE.name == "workday_tenants.txt"


def test_default_file_lives_at_the_repo_root():
    # Repo root, next to ashby_boards.txt, so it stays editable after an install.
    assert wt.DEFAULT_TENANTS_FILE.parent.name == "jobecosystem"
    assert wt.DEFAULT_TENANTS_FILE.parent.name != "src"


def test_resolve_expands_user(monkeypatch):
    monkeypatch.delenv("WORKDAY_TENANTS_FILE", raising=False)
    assert "~" not in str(wt.resolve_tenants_file("~/t.txt"))


# ---------------------------------------------------------------------------
# load_tenants
# ---------------------------------------------------------------------------

def test_load_reads_a_file(tmp_path):
    path = tmp_path / "tenants.txt"
    path.write_text("nvidia.wd5 nvidia Site Nvidia\n", encoding="utf-8")
    (spec,) = wt.load_tenants(path).specs
    assert spec.host == "nvidia.wd5"


def test_load_returns_empty_for_a_missing_file(tmp_path):
    outcome = wt.load_tenants(tmp_path / "nope.txt")
    assert outcome.specs == []
    assert outcome.errors == []


def test_load_uses_the_environment(tmp_path, monkeypatch):
    path = tmp_path / "from-env.txt"
    path.write_text("h.wd1 t S\n", encoding="utf-8")
    monkeypatch.setenv("WORKDAY_TENANTS_FILE", str(path))
    assert [s.host for s in wt.load_tenants().specs] == ["h.wd1"]


def test_the_shipped_file_loads_cleanly():
    outcome = wt.load_tenants()
    assert outcome.errors == []
    assert len(outcome.specs) >= 1


def test_the_shipped_file_has_no_duplicate_sites():
    specs = wt.load_tenants().specs
    keys = {(s.host, s.tenant, s.site) for s in specs}
    assert len(keys) == len(specs)


def test_the_shipped_file_excludes_the_robots_disallowed_site():
    # ASMLEXT1 is commented out; ASMLPrivate1 is the allowed alternative.
    sites = {s.site for s in wt.load_tenants().specs}
    assert "ASMLEXT1" not in sites
    assert "ASMLPrivate1" in sites


def test_the_shipped_file_has_expected_sites():
    sites = {s.site for s in wt.load_tenants().specs}
    assert {"NVIDIAExternalCareerSite", "disneycareer", "External_Career_Site"} <= sites


# ---------------------------------------------------------------------------
# build_scrapers
# ---------------------------------------------------------------------------

def test_build_creates_one_scraper_per_site(tmp_path):
    path = tmp_path / "tenants.txt"
    path.write_text(
        "nvidia.wd5 nvidia Site Nvidia\nasml.wd3 asml Other ASML\n", encoding="utf-8"
    )
    scrapers = wt.build_scrapers(path)
    assert len(scrapers) == 2
    assert all(isinstance(s, WorkdayScraper) for s in scrapers)


def test_built_scrapers_carry_all_addressing_parts(tmp_path):
    path = tmp_path / "tenants.txt"
    path.write_text("asml.wd3 asml ASMLPrivate1 ASML\n", encoding="utf-8")
    (scraper,) = wt.build_scrapers(path)
    assert (scraper.host, scraper.tenant, scraper.site) == (
        "asml.wd3", "asml", "ASMLPrivate1"
    )
    assert scraper.company == "ASML"
    assert scraper.source == "workday:asml:ASMLPrivate1"


def test_built_scraper_company_falls_back_to_the_tenant(tmp_path):
    path = tmp_path / "tenants.txt"
    path.write_text("h.wd1 acme Site\n", encoding="utf-8")
    (scraper,) = wt.build_scrapers(path)
    assert scraper.company == "acme"


def test_build_returns_empty_for_a_missing_file(tmp_path):
    assert wt.build_scrapers(tmp_path / "nope.txt") == []


def test_build_passes_post_json_through(tmp_path):
    path = tmp_path / "tenants.txt"
    path.write_text("h.wd1 t S\n", encoding="utf-8")

    def stub(url, body):
        return {"total": 0, "jobPostings": []}

    (scraper,) = wt.build_scrapers(path, post_json=stub)
    assert scraper.fetch() == []


def test_built_scrapers_have_distinct_sources(tmp_path):
    path = tmp_path / "tenants.txt"
    path.write_text("sf.wd12 sf Main\nsf.wd12 sf Slack\n", encoding="utf-8")
    sources = [s.source for s in wt.build_scrapers(path)]
    assert sources == ["workday:sf:Main", "workday:sf:Slack"]


# ---------------------------------------------------------------------------
# integration with the runner
# ---------------------------------------------------------------------------

def test_scrapers_from_the_file_run_through_the_runner(conn, tmp_path):
    from jobecosystem.ingest import runner

    path = tmp_path / "tenants.txt"
    # A site name with a space would be ambiguous with the display name, so the
    # two sites here are single tokens.
    path.write_text("a.wd1 alpha Main Alpha\na.wd1 alpha Other Alpha\n", encoding="utf-8")

    payloads = {
        "Main": {"total": 1, "jobPostings": [
            {"title": "Engineer", "externalPath": "/job/X/E_1", "bulletFields": ["A1"]}]},
        "Other": {"total": 1, "jobPostings": [
            {"title": "Designer", "externalPath": "/job/X/D_2", "bulletFields": ["B2"]}]},
    }

    def stub(url, body):
        return payloads[next(k for k in payloads if url.endswith(k + "/jobs"))]

    summary = runner.run_scrapers(conn, wt.build_scrapers(path, post_json=stub))
    assert [o.source for o in summary] == [
        "workday:alpha:Main", "workday:alpha:Other"
    ]
    assert summary.ok
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
