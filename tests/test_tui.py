"""Tests for jobecosystem.triage.tui.app.

The app is driven headlessly through Textual's test harness, which runs the real
widgets without a terminal. Assertions target behaviour -- what the tables
contain, what the writers did -- rather than widget internals.
"""

from __future__ import annotations

import pytest

from jobecosystem.core.models import Job
from jobecosystem.ingest import upsert
from jobecosystem.triage import queries as q
from jobecosystem.triage.tui import app as tui

textual = pytest.importorskip("textual")


@pytest.fixture
def loaded(conn):
    """Five jobs across two sources, one of them triaged."""
    rows = [
        ("1", "ashby:ramp", "Acme", "Senior Python Engineer", "backend python sqlite"),
        ("2", "ashby:ramp", "Acme", "Rust Systems Engineer", "systems rust kernels"),
        ("3", "workday:nvidia:S", "Nvidia", "Data Scientist", "python ml models"),
        ("4", "workday:nvidia:S", "Nvidia", "Registered Nurse", "pediatric care"),
        ("5", "ashby:zapier", "Zapier", "Frontend Engineer", "react typescript"),
    ]
    ids = {}
    for external_id, source, company, title, description in rows:
        result = upsert.upsert_job(
            conn,
            Job(source=source, external_id=external_id, company=company,
                title=title, description=description,
                url=f"https://x.test/{external_id}"),
        )
        ids[external_id] = result.job_id
    conn.commit()
    return ids


def fake_embed(mapping=None):
    """A deterministic embedder, so no model is needed."""
    mapping = mapping or {"python": [1.0, 0.0], "rust": [0.0, 1.0]}
    calls: list[str] = []

    def embed(text):
        calls.append(text)
        lowered = text.lower()
        for key, vector in mapping.items():
            if key in lowered:
                return vector
        return [0.0, 0.0]

    embed.calls = calls  # type: ignore[attr-defined]
    return embed


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def test_app_mounts(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.rows
            return len(app.rows)

    assert _run(scenario()) >= 1


def test_app_loads_unseen_jobs_first(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            return [r.job.external_id for r in app.rows]

    assert sorted(_run(scenario())) == ["1", "2", "3", "4", "5"]


def test_app_respects_the_page_size(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn, page_size=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 2


def test_table_has_a_row_per_job(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            return app.current_table().row_count

    assert _run(scenario()) == 5


def test_subtitle_reports_totals(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            return app.sub_title

    subtitle = _run(scenario())
    assert "5 jobs" in subtitle
    assert "shown" in subtitle


def test_app_starts_with_an_empty_database(conn):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 0


# ---------------------------------------------------------------------------
# tabs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tab", ["today", "reposted", "stale", "search", "similar"])
def test_every_tab_renders(conn, loaded, tab):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = tab
            await pilot.pause()
            return app._active_tab

    assert _run(scenario()) == tab


def test_today_tab_shows_recent_jobs(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "today"
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 5


def test_search_and_similar_tabs_start_empty(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            search_rows = len(app.rows)
            app.query_one("TabbedContent").active = "similar"
            await pilot.pause()
            return search_rows, len(app.rows)

    assert _run(scenario()) == (0, 0)


# ---------------------------------------------------------------------------
# rating and status
# ---------------------------------------------------------------------------

def test_rating_a_job(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+5")
            await pilot.pause()
            return app.selected_job().rating

    assert _run(scenario()) == 5


def test_pressing_the_same_rating_clears_it(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+4")
            await pilot.pause()
            await pilot.press("ctrl+4")
            await pilot.pause()
            return app.selected_job().rating

    assert _run(scenario()) is None


def test_rating_persists_to_the_database(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+3")
            await pilot.pause()
            return app.selected_job().id

    job_id = _run(scenario())
    assert conn.execute("SELECT rating FROM jobs WHERE id = ?", (job_id,)).fetchone()[0] == 3


def test_cycling_status_moves_a_job_out_of_new(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            before = len(app.rows)
            await pilot.press("s")
            await pilot.pause()
            return before, len(app.rows)

    before, after = _run(scenario())
    assert after == before - 1


def test_hiding_a_job_removes_it_from_the_list(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            before = len(app.rows)
            await pilot.press("x")
            await pilot.pause()
            return before, len(app.rows)

    before, after = _run(scenario())
    assert after == before - 1
    assert conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'hidden'"
    ).fetchone()[0] == 1


def test_status_cycling_is_a_no_op_on_an_empty_list(conn):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 0


# ---------------------------------------------------------------------------
# keyword search
# ---------------------------------------------------------------------------

def test_submitting_a_search_filters_the_table(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "python"
            await pilot.press("enter")
            await pilot.pause()
            return [r.job.title for r in app.rows]

    titles = _run(scenario())
    assert "Senior Python Engineer" in titles
    assert "Rust Systems Engineer" not in titles


def test_search_is_case_insensitive(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "PYTHON"
            await pilot.press("enter")
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) >= 1


def test_clearing_the_search_empties_the_table(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "python"
            await pilot.press("enter")
            await pilot.pause()
            # Focus moved to the results, so slash back before re-submitting.
            await pilot.press("slash")
            await pilot.pause()
            app.query_one("#search-input").value = ""
            await pilot.press("enter")
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 0


def test_search_moves_focus_to_the_results(conn, loaded):
    # Landing in the box after a search would trap the arrow keys in the caret.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "python"
            await pilot.press("enter")
            await pilot.pause()
            return app.focused.__class__.__name__

    assert _run(scenario()) == "JobsTable"


def test_arrows_navigate_the_results_right_after_a_search(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "engineer"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("down")
            await pilot.pause()
            return app.current_table().cursor_row

    assert _run(scenario()) == 1


def test_slash_returns_to_the_search_box(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "python"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            return app.query_one("#search-input").has_focus

    assert _run(scenario()) is True


def test_arrows_from_the_box_move_the_list_selection(conn, loaded):
    # The box keeps focus while you edit a query; its single line has nowhere for
    # the caret to go, so up/down must drive the list instead of being swallowed.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "engineer"
            await pilot.press("enter")
            await pilot.pause()
            # Back to the box, where the rows are still loaded.
            await pilot.press("slash")
            await pilot.pause()
            assert app.query_one("#search-input").has_focus
            assert len(app.rows) >= 2
            await pilot.press("down")
            await pilot.pause()
            return app.current_table().cursor_row, app.focused.__class__.__name__

    row, focused = _run(scenario())
    assert row == 1
    assert focused == "JobsTable"


def test_search_with_no_matches_is_not_an_error(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "cobol"
            await pilot.press("enter")
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 0


# ---------------------------------------------------------------------------
# embedding search
# ---------------------------------------------------------------------------

def attach_vectors(conn, loaded):
    """Give two jobs orthogonal vectors, so ranking is deterministic."""
    from jobecosystem.core import similarity

    def store(job_id, vector):
        with conn:
            conn.execute(
                "INSERT INTO job_embeddings (job_id, vector, dim, model, updated_at)"
                " VALUES (?, ?, ?, 'test', '2026-01-01T00:00:00Z')"
                " ON CONFLICT (job_id) DO UPDATE SET vector = excluded.vector,"
                " dim = excluded.dim",
                (job_id, similarity.encode(vector), len(vector)),
            )

    store(loaded["1"], [1.0, 0.0])
    store(loaded["2"], [0.0, 1.0])


def test_similar_search_ranks_by_score(conn, loaded):
    attach_vectors(conn, loaded)

    async def scenario():
        app = tui.JobApp(conn, embed_query=fake_embed())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "python"
            await pilot.press("enter")
            await pilot.pause()
            app.action_search_similar()
            await pilot.pause()
            return [(r.job.external_id, r.score) for r in app.rows]

    rows = _run(scenario())
    assert rows
    assert rows[0][0] == "1"                 # the python job wins
    assert rows[0][1] == pytest.approx(1.0, abs=1e-6)


def test_similar_search_requires_a_query(conn, loaded):
    attach_vectors(conn, loaded)

    async def scenario():
        app = tui.JobApp(conn, embed_query=fake_embed())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_search_similar()
            await pilot.pause()
            # Nothing searched, so the tab did not switch and no results exist.
            return app._active_tab, len(app._similar_rows)

    assert _run(scenario()) == ("new", 0)


def test_similar_search_with_an_empty_query_does_not_embed(conn, loaded):
    attach_vectors(conn, loaded)
    embedder = fake_embed()

    async def scenario():
        app = tui.JobApp(conn, embed_query=embedder)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "   "
            app.action_search_similar()
            await pilot.pause()

    _run(scenario())
    assert embedder.calls == []


def test_similar_search_without_an_embedder_is_a_notification(conn, loaded):
    attach_vectors(conn, loaded)

    async def scenario():
        app = tui.JobApp(conn, embed_query=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "python"
            app.action_search_similar()
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 0        # degrades, does not crash


def test_embedding_is_cached_per_session(conn, loaded):
    attach_vectors(conn, loaded)
    embedder = fake_embed()

    async def scenario():
        app = tui.JobApp(conn, embed_query=embedder)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "python"
            app.action_search_similar()
            await pilot.pause()
            app.action_search_similar()
            await pilot.pause()
            return list(embedder.calls)

    # Two searches, one model call.
    assert _run(scenario()) == ["python"]


def test_similar_to_selected_uses_the_stored_vector(conn, loaded):
    attach_vectors(conn, loaded)

    async def scenario():
        app = tui.JobApp(conn, embed_query=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Select the python job (external id 1) and ask for neighbours.
            target = next(i for i, r in enumerate(app.rows) if r.job.external_id == "1")
            app.current_table().move_cursor(row=target)
            await pilot.pause()
            app.action_similar_to_selected()
            await pilot.pause()
            return [(r.job.external_id, r.score) for r in app.rows]

    rows = _run(scenario())
    assert "1" not in [external_id for external_id, _ in rows]   # not itself
    assert rows[0][0] == "2"


def test_similar_to_a_job_without_an_embedding_reports_it(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn, embed_query=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_similar_to_selected()
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 0


def test_status_change_updates_the_similar_view(conn, loaded):
    attach_vectors(conn, loaded)

    async def scenario():
        app = tui.JobApp(conn, embed_query=fake_embed())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "python"
            app.action_search_similar()
            await pilot.pause()
            before = [row.job.status for row in app.rows]
            app.action_set_status("hidden")
            await pilot.pause()
            after = [row.job.status for row in app.rows]
            return before, after

    before, after = _run(scenario())
    assert before and all(status == "new" for status in before)
    # The Similar tab must show the live status, not the search-time snapshot.
    assert "hidden" in after


def test_resume_match_ranks_jobs_and_switches_tab(conn, loaded, sample_resume, tmp_path):
    from jobecosystem.tailor import resume as tailor_resume

    attach_vectors(conn, loaded)
    path = tmp_path / "resume.json"
    tailor_resume.save(sample_resume, path)

    async def scenario():
        app = tui.JobApp(conn, embed_query=fake_embed(), resume_path=str(path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_resume_match()
            await pilot.pause()
            return app._active_tab, app._similar_for, list(app.rows)

    tab, label, rows = _run(scenario())
    assert tab == "similar"
    assert label == f"resume: {path.name}"
    # The resume query text contains "Python", so job 1's [1, 0] vector wins.
    assert [row.job.external_id for row in rows][0] == "1"
    assert rows[0].score == pytest.approx(1.0, abs=1e-6)


def test_resume_match_reports_a_missing_file_without_stopping(conn, loaded, tmp_path):
    async def scenario():
        app = tui.JobApp(
            conn, embed_query=fake_embed(), resume_path=str(tmp_path / "nope.json")
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_resume_match()
            await pilot.pause()
            return app._active_tab, len(app._similar_rows)

    # A bad path is reported and ignored: the app stays where it was.
    assert _run(scenario()) == ("new", 0)


def test_resume_match_without_an_embedder_is_a_notification(conn, loaded, sample_resume,
                                                            tmp_path):
    from jobecosystem.tailor import resume as tailor_resume

    path = tmp_path / "resume.json"
    tailor_resume.save(sample_resume, path)

    async def scenario():
        app = tui.JobApp(conn, embed_query=None, resume_path=str(path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_resume_match()
            await pilot.pause()
            return app._active_tab, len(app._similar_rows)

    assert _run(scenario()) == ("new", 0)


def test_resume_match_caches_the_vector_per_session(conn, loaded, sample_resume, tmp_path):
    from jobecosystem.tailor import resume as tailor_resume

    attach_vectors(conn, loaded)
    path = tmp_path / "resume.json"
    tailor_resume.save(sample_resume, path)
    embedder = fake_embed()

    async def scenario():
        app = tui.JobApp(conn, embed_query=embedder, resume_path=str(path))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_resume_match()
            await pilot.pause()
            app.action_resume_match()
            await pilot.pause()
            return list(embedder.calls)

    # Re-pressing ctrl+r on the same file is free: one model call, not two.
    assert len(_run(scenario())) == 1


# ---------------------------------------------------------------------------
# reload
# ---------------------------------------------------------------------------

def test_reload_reflects_external_changes(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            before = len(app.rows)
            # Something else triages a job -- a cron job, or another session.
            q.set_status(conn, loaded["1"], "hidden")
            await pilot.press("r")
            await pilot.pause()
            return before, len(app.rows)

    before, after = _run(scenario())
    assert after == before - 1


def test_reload_keeps_the_cursor_position(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.current_table().move_cursor(row=2)
            await pilot.pause()
            selected = app.selected_job().external_id
            await pilot.press("r")
            await pilot.pause()
            return selected, app.selected_job().external_id

    assert _run(scenario()) == ("3", "3")


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# detail pane
# ---------------------------------------------------------------------------

def test_detail_pane_shows_the_selected_job(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.current_table().move_cursor(row=1)
            await pilot.pause()
            # Re-query rather than holding the widget: show() replaces children.
            detail = app.query_one("#detail", tui.JobDetail)
            return len(detail.children), app.selected_job()

    child_count, job = _run(scenario())
    assert job is not None
    assert child_count >= 1


def test_detail_pane_handles_an_empty_selection(conn):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            detail = app.query_one("#detail", tui.JobDetail)
            detail.show(None)
            await pilot.pause()
            return len(detail.children)

    assert _run(scenario()) >= 1


# ---------------------------------------------------------------------------
# formatting, without a running app
# ---------------------------------------------------------------------------

def test_format_row_includes_a_score_only_when_present():
    job = Job(source="a", external_id="1", company="Acme", title="Eng", id=7)
    plain = tui.JobApp.format_row(tui.Row(job))
    scored = tui.JobApp.format_row(tui.Row(job, 0.5123))
    assert plain[0] == "7"
    assert plain[3] == ""
    assert scored[3] == "0.512"


def test_format_row_marks_an_unrated_job():
    job = Job(source="a", external_id="1", company="Acme", title="Eng")
    assert tui.JobApp.format_row(tui.Row(job))[1] == "—"


def test_format_row_truncates_a_long_company():
    job = Job(source="a", external_id="1", company="N" * 40, title="Eng")
    shown = tui.JobApp.format_row(tui.Row(job))[4]
    assert tui._cell_len(shown) <= tui.COMPANY_WIDTH
    assert shown.endswith(tui.ELLIPSIS)


def test_a_long_company_is_truncated_in_the_table(conn):
    long_company = "Nvidia Corporation International Holdings"
    upsert.upsert_job(
        conn,
        Job(source="ashby:ramp", external_id="1", company=long_company,
            title="Engineer", description="body"),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            return str(app.current_table().get_row_at(0)[4])

    shown = _run(scenario())
    assert tui._cell_len(shown) <= tui.COMPANY_WIDTH
    assert shown.endswith(tui.ELLIPSIS)
    # The store keeps the whole name; only the list display is cut.
    stored = conn.execute("SELECT company FROM jobs").fetchone()[0]
    assert stored == long_company


def test_every_status_has_a_key():
    """One binding per status, so no status is unreachable from the UI."""
    actions = {
        b.action for b in tui.JobApp.BINDINGS if b.action.startswith("set_status(")
    }
    keys = {
        b.action.split("'")[1]
        for b in tui.JobApp.BINDINGS
        if b.action.startswith("set_status(")
    }
    assert keys == set(q.VALID_STATUSES)
    assert len(actions) == len(q.VALID_STATUSES)


def test_tabs_are_declared_in_order():
    assert [key for key, _ in tui.TABS] == [
        "new", "today", "applied", "all", "reposted", "stale", "search", "similar"
    ]


def test_build_embedder_never_raises():
    # It returns None when the model is unavailable, so the TUI still starts.
    result = tui.build_embedder()
    assert result is None or callable(result)


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine on a fresh event loop.

    Textual's harness is async; pytest is not, and no async plugin is installed.
    """
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# tab navigation
# ---------------------------------------------------------------------------

def test_bracket_keys_cycle_forward_through_every_tab(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            seen = [app._active_tab]
            for _ in range(len(tui.TABS) - 1):
                await pilot.press("]")
                await pilot.pause()
                seen.append(app._active_tab)
            return seen

    assert _run(scenario()) == [key for key, _ in tui.TABS]


def test_bracket_keys_wrap_around(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Once per tab, plus one to come back to the start.
            for _ in range(len(tui.TABS)):
                await pilot.press("]")
                await pilot.pause()
            return app._active_tab

    assert _run(scenario()) == "new"


def test_prev_tab_wraps_backwards(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("[")
            await pilot.pause()
            return app._active_tab

    assert _run(scenario()) == "similar"


def test_bracket_keys_work_while_the_search_box_has_focus(conn, loaded):
    # The box swallows printable keys, so this is the case that needs
    # forwarding; otherwise there is no way back off the Search tab.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            assert app.query_one("#search-input").has_focus
            await pilot.press("]")
            await pilot.pause()
            return app._active_tab

    assert _run(scenario()) == "similar"


def test_bracket_characters_are_not_typed_into_the_search_box(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            await pilot.press("]")
            await pilot.press("[")
            await pilot.pause()
            return app.query_one("#search-input").value

    assert _run(scenario()) == ""


def test_typing_in_the_search_box_still_works(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            await pilot.press(*"python")
            await pilot.pause()
            return app.query_one("#search-input").value

    assert _run(scenario()) == "python"


def test_digits_type_into_the_search_box(conn, loaded):
    # Digits are no longer tab jumps: they belong to the query.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            await pilot.press(*"senior 2")
            await pilot.pause()
            return app.query_one("#search-input").value

    assert _run(scenario()) == "senior 2"


def test_rating_is_on_ctrl_digit(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+2")
            await pilot.pause()
            return app.selected_job().rating

    assert _run(scenario()) == 2


def test_footer_shows_the_tab_keys():
    keys = {
        b.key_display or b.key
        for b in tui.JobApp.BINDINGS
        if b.show
    }
    assert "]" in keys
    assert "[" in keys


# ---------------------------------------------------------------------------
# detail pane: layout and content
# ---------------------------------------------------------------------------

def detail_text(app) -> str:
    """The detail pane's rendered text, as the user would see it."""
    pane = app.query_one("#detail", tui.JobDetail)
    parts = []
    for child in pane.children:
        content = child.render()
        parts.append(getattr(content, "plain", str(content)))
    return "\n".join(parts)


def test_detail_pane_is_visible_beside_the_list(conn, loaded):
    # Regression: the tabs were a sibling of the pane rather than sharing a
    # horizontal container, so the tabs filled the screen and the pane was
    # squeezed to a one-row strip off the bottom.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            return (
                app.query_one("#tabs").region,
                app.query_one("#detail").region,
                app.query_one("Footer").region,
            )

    tabs, detail, footer = _run(scenario())
    assert detail.width > 20, "the detail pane is too narrow to read"
    assert detail.height > 5, "the detail pane is too short to read"
    assert detail.x >= tabs.right - 1, "the pane should sit beside the list"
    assert detail.bottom <= footer.y + 1, "the pane must not overlap the footer"


def test_detail_pane_shows_the_job_title(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            return detail_text(app)

    text = _run(scenario())
    assert "Senior" in text or "Engineer" in text or "Nurse" in text
    assert "source:" in text


def test_detail_pane_shows_the_description(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            # Select the job with a known description.
            target = next(
                i for i, r in enumerate(app.rows) if r.job.external_id == "1"
            )
            app.current_table().move_cursor(row=target)
            await pilot.pause()
            await pilot.pause()
            return detail_text(app)

    assert "backend python sqlite" in _run(scenario())


def test_detail_pane_says_so_when_a_description_is_missing(conn):
    upsert.upsert_job(
        conn,
        Job(source="workday:x:S", external_id="1", company="Acme",
            title="No Description Here", description=None),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            return detail_text(app)

    text = _run(scenario())
    assert "No description yet" in text
    assert "press d" in text.lower()


def test_detail_pane_updates_when_the_selection_changes(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            app.current_table().move_cursor(row=0)
            await pilot.pause()
            await pilot.pause()
            first = detail_text(app)
            app.current_table().move_cursor(row=1)
            await pilot.pause()
            await pilot.pause()
            return first, detail_text(app)

    first, second = _run(scenario())
    assert first != second


# ---------------------------------------------------------------------------
# fetching a description from the TUI
# ---------------------------------------------------------------------------

def test_fetching_a_description_from_the_tui(conn, monkeypatch):
    # Regression: the worker thread used to call the whole fetch+store helper,
    # so SQLite raised "objects created in a thread can only be used in that
    # same thread". Only the HTTP call may run off-thread now.
    from jobecosystem.ingest.sources import description as desc

    monkeypatch.setattr(
        desc, "_http_get_json",
        lambda url, **kw: {"jobPostingInfo": {"jobDescription": "<p>Body text.</p>"}},
    )
    upsert.upsert_job(
        conn,
        Job(source="workday:x:S", external_id="1", company="Acme", title="Engineer",
            description=None, description_url="https://x.test/detail/1"),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("d")
            for _ in range(30):
                await pilot.pause(0.05)
                job = app.selected_job()
                if job is not None and job.description:
                    break
            return app.selected_job()

    job = _run(scenario())
    assert job.description == "Body text."
    assert job.content_hash is not None
    assert job.description_fetched_at is not None


def test_fetching_a_description_writes_to_the_database(conn, monkeypatch):
    from jobecosystem.ingest.sources import description as desc

    monkeypatch.setattr(
        desc, "_http_get_json",
        lambda url, **kw: {"jobPostingInfo": {"jobDescription": "<p>Stored.</p>"}},
    )
    result = upsert.upsert_job(
        conn,
        Job(source="workday:x:S", external_id="1", company="Acme", title="Engineer",
            description=None, description_url="https://x.test/detail/1"),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("d")
            for _ in range(30):
                await pilot.pause(0.05)
                if conn.execute(
                    "SELECT description FROM jobs WHERE id = ?", (result.job_id,)
                ).fetchone()[0]:
                    break

    _run(scenario())
    stored = conn.execute(
        "SELECT description FROM jobs WHERE id = ?", (result.job_id,)
    ).fetchone()[0]
    assert stored == "Stored."


def test_pressing_d_counts_an_attempt_and_ignores_the_cap(conn, monkeypatch):
    from jobecosystem.ingest.sources import description as desc

    monkeypatch.setattr(
        desc, "_http_get_json",
        lambda url, **kw: {"jobPostingInfo": {"jobDescription": "<p>Retried.</p>"}},
    )
    result = upsert.upsert_job(
        conn,
        Job(source="workday:x:S", external_id="1", company="Acme", title="Engineer",
            description=None, description_url="https://x.test/detail/1"),
    )
    conn.execute(
        "UPDATE jobs SET description_attempts = 5 WHERE id = ?", (result.job_id,)
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("d")
            for _ in range(30):
                await pilot.pause(0.05)
                if conn.execute(
                    "SELECT description FROM jobs WHERE id = ?", (result.job_id,)
                ).fetchone()[0]:
                    break

    _run(scenario())
    row = conn.execute(
        "SELECT description, description_attempts FROM jobs WHERE id = ?",
        (result.job_id,),
    ).fetchone()
    assert row["description"] == "Retried."
    assert row["description_attempts"] == 6      # 'd' bypasses the cap but counts


def test_fetching_skips_a_job_that_already_has_a_description(conn, loaded):
    def exploding(url, **kwargs):
        raise AssertionError("should not have made a request")

    from jobecosystem.ingest.sources import description as desc

    original = desc._http_get_json
    desc._http_get_json = exploding
    try:
        async def scenario():
            app = tui.JobApp(conn)
            async with app.run_test() as pilot:
                await pilot.pause()
                target = next(
                    i for i, r in enumerate(app.rows) if r.job.external_id == "1"
                )
                app.current_table().move_cursor(row=target)
                await pilot.pause()
                await pilot.press("d")
                await pilot.pause()
                return app.selected_job().description

        assert _run(scenario()) == "backend python sqlite"
    finally:
        desc._http_get_json = original


def test_fetching_without_a_detail_url_warns(conn):
    upsert.upsert_job(
        conn,
        Job(source="ashby:ramp", external_id="1", company="Acme", title="Eng",
            description=None, description_url=None),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()
            return app.selected_job().description

    # Must not raise, and must leave the row alone.
    assert _run(scenario()) is None


# ---------------------------------------------------------------------------
# status keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key, status",
    [("s", "seen"), ("a", "applied"), ("z", "new"), ("x", "hidden")],
)
def test_status_keys_set_the_expected_status(conn, loaded, key, status):
    def scenario():
        async def run():
            app = tui.JobApp(conn)
            async with app.run_test() as pilot:
                await pilot.pause()
                # Track the job we act on: it leaves the New tab, so the list
                # contents afterwards are not where to look for it.
                job_id = app.selected_job().id
                await pilot.press(key)
                await pilot.pause()
                return conn.execute(
                    "SELECT status FROM jobs WHERE id = ?", (job_id,)
                ).fetchone()[0]
        return run()

    assert _run(scenario()) == status


def test_applied_key_moves_the_job_to_the_applied_tab(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            job_id = app.selected_job().id
            await pilot.press("a")
            await pilot.pause()
            app.query_one("TabbedContent").active = "applied"
            await pilot.pause()
            applied = [r.job.id for r in app.rows]
            app.query_one("TabbedContent").active = "new"
            await pilot.pause()
            return job_id, applied, [r.job.id for r in app.rows]

    job_id, applied, new_rows = _run(scenario())
    assert job_id in applied
    assert job_id not in new_rows


def test_z_returns_a_job_to_new(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            job_id = app.selected_job().id
            await pilot.press("a")
            await pilot.pause()
            # The job left New, so it is no longer the selection there. Put it
            # back via the All tab, which always contains everything.
            app.query_one("TabbedContent").active = "all"
            await pilot.pause()
            target = next(i for i, r in enumerate(app.rows) if r.job.id == job_id)
            app.current_table().move_cursor(row=target)
            await pilot.pause()
            await pilot.press("z")
            await pilot.pause()
            app.query_one("TabbedContent").active = "new"
            await pilot.pause()
            return job_id, [r.job.id for r in app.rows]

    job_id, new_rows = _run(scenario())
    assert job_id in new_rows


def test_pressing_the_current_status_is_harmless(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("s")      # new -> seen
            await pilot.pause()
            app.query_one("TabbedContent").active = "all"
            await pilot.pause()
            await pilot.press("s")      # already seen
            await pilot.pause()
            return conn.execute("SELECT COUNT(*) FROM jobs WHERE status='seen'").fetchone()[0]

    assert _run(scenario()) == 1


def test_status_keys_are_all_shown_in_the_footer():
    shown = {b.key for b in tui.JobApp.BINDINGS if b.show}
    assert {"s", "a", "z", "x"} <= shown


def test_open_url_binding_is_gone():
    # Removed by request; the URL is still visible in the detail pane.
    assert "o" not in {b.key for b in tui.JobApp.BINDINGS}


# ---------------------------------------------------------------------------
# the Applied and All tabs
# ---------------------------------------------------------------------------

def test_applied_tab_starts_empty(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "applied"
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 0


def test_applied_tab_shows_only_applied(conn, loaded):
    q.set_status(conn, loaded["1"], "applied")
    q.set_status(conn, loaded["2"], "seen")

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "applied"
            await pilot.pause()
            return [r.job.id for r in app.rows]

    assert _run(scenario()) == [loaded["1"]]


def test_all_tab_shows_every_job_regardless_of_status(conn, loaded):
    q.set_status(conn, loaded["1"], "applied")
    q.set_status(conn, loaded["2"], "hidden")
    q.set_status(conn, loaded["3"], "seen")

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "all"
            await pilot.pause()
            return sorted(r.job.id for r in app.rows)

    assert _run(scenario()) == sorted(loaded.values())


def test_all_tab_respects_the_page_size(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn, page_size=2)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("TabbedContent").active = "all"
            await pilot.pause()
            return len(app.rows)

    assert _run(scenario()) == 2


def test_hidden_jobs_appear_in_all_but_not_new(conn, loaded):
    q.set_status(conn, loaded["1"], "hidden")

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            new_ids = [r.job.id for r in app.rows]
            app.query_one("TabbedContent").active = "all"
            await pilot.pause()
            return new_ids, [r.job.id for r in app.rows]

    new_ids, all_ids = _run(scenario())
    assert loaded["1"] not in new_ids
    assert loaded["1"] in all_ids


# ---------------------------------------------------------------------------
# stale and repost flags in the detail pane
# ---------------------------------------------------------------------------

def test_detail_text_shows_no_flags_for_a_plain_job():
    job = Job(source="a", external_id="1", company="Acme", title="Eng")
    text = tui._detail_text(tui.Row(job))
    assert "stale" not in text
    assert "reposted" not in text


def test_detail_text_flags_a_stale_job():
    job = Job(source="a", external_id="1", company="Acme", title="Eng")
    assert "stale" in tui._detail_text(tui.Row(job), stale=True)


def test_detail_text_flags_a_duplicate_content_repost():
    # repost_count is 0 here: the evidence is the duplicate row count, which is
    # exactly the case that used to render no flag at all.
    job = Job(source="a", external_id="1", company="Acme", title="Eng")
    text = tui._detail_text(
        tui.Row(job), is_reposted=True, duplicate_count=1
    )
    assert "reposted" in text
    assert "1 duplicate" in text
    assert "0\u00d7 seen" in text


def test_detail_text_reports_both_kinds_of_repost_evidence():
    job = Job(source="a", external_id="1", company="Acme", title="Eng",
              repost_count=3)
    text = tui._detail_text(
        tui.Row(job), is_reposted=True, duplicate_count=2
    )
    assert "3\u00d7 seen" in text
    assert "2 duplicate" in text


def test_detail_text_can_show_both_flags():
    job = Job(source="a", external_id="1", company="Acme", title="Eng")
    text = tui._detail_text(
        tui.Row(job), stale=True, is_reposted=True, duplicate_count=1
    )
    assert "stale" in text
    assert "reposted" in text


def test_detail_pane_flags_the_stale_tab_rows(conn, loaded):
    q.set_status(conn, loaded["1"], "seen")
    conn.execute(
        "UPDATE jobs SET last_seen_at = '2026-01-01T00:00:00Z' WHERE id = ?",
        (loaded["1"],),
    )
    conn.execute("UPDATE jobs SET last_seen_at = '2099-01-01T00:00:00Z' WHERE id != ?",
                 (loaded["1"],))
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.query_one("TabbedContent").active = "stale"
            await pilot.pause()
            await pilot.pause()
            return [r.job.id for r in app.rows], detail_text(app)

    stale_ids, text = _run(scenario())
    assert loaded["1"] in stale_ids
    assert "stale" in text


def test_detail_pane_flags_a_duplicate_content_row(conn):
    # Two ids, identical text: the Reposted tab shows both, and each must say
    # why it is there.
    for external_id in ("A", "B"):
        upsert.upsert_job(
            conn,
            Job(source="ashby:ramp", external_id=external_id, company="Acme",
                title="Senior Engineer", description="identical body text"),
        )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.query_one("TabbedContent").active = "reposted"
            await pilot.pause()
            await pilot.pause()
            return len(app.rows), detail_text(app)

    rows, text = _run(scenario())
    assert rows == 2
    assert "reposted" in text
    assert "duplicate" in text


def test_detail_pane_shows_no_flags_for_an_ordinary_job(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            return detail_text(app)

    text = _run(scenario())
    assert "stale" not in text
    assert "reposted" not in text


# ---------------------------------------------------------------------------
# copying the URL
# ---------------------------------------------------------------------------

def test_copy_url_puts_the_url_on_the_clipboard(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        copied = []
        async with app.run_test() as pilot:
            await pilot.pause()
            # copy_to_clipboard ultimately writes an OSC 52 escape sequence; on
            # a headless driver it is recorded on the app.
            app.copy_to_clipboard = copied.append
            await pilot.press("c")
            await pilot.pause()
            return app.selected_job().url, copied

    url, copied = _run(scenario())
    assert copied == [url]
    assert url.startswith("https://")


def test_copy_url_warns_when_there_is_no_url(conn):
    upsert.upsert_job(
        conn,
        Job(source="ashby:ramp", external_id="1", company="Acme", title="Eng",
            description="body", url=None),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        copied = []
        async with app.run_test() as pilot:
            await pilot.pause()
            app.copy_to_clipboard = copied.append
            await pilot.press("c")
            await pilot.pause()
            return copied

    # Must not raise, and must not copy an empty string.
    assert _run(scenario()) == []


def test_copy_url_on_an_empty_list_is_harmless(conn):
    async def scenario():
        app = tui.JobApp(conn)
        copied = []
        async with app.run_test() as pilot:
            await pilot.pause()
            app.copy_to_clipboard = copied.append
            await pilot.press("c")
            await pilot.pause()
            return copied

    assert _run(scenario()) == []


def test_copy_binding_is_shown_in_the_footer():
    shown = {b.key for b in tui.JobApp.BINDINGS if b.show}
    assert "c" in shown


# ---------------------------------------------------------------------------
# the URL bar
# ---------------------------------------------------------------------------

def url_bar_text(app) -> str:
    """The URL bar's content as plain text."""
    bar = app.query_one("#url-bar", tui.Static)
    rendered = bar.render()
    return getattr(rendered, "plain", str(rendered)).strip()


def test_url_bar_shows_only_the_url(conn, loaded):
    # Its whole purpose: a drag should capture the URL and nothing else, so no
    # markup, padding text, or second field may appear on that line.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            target = next(
                i for i, r in enumerate(app.rows) if r.job.external_id == "1"
            )
            app.current_table().move_cursor(row=target)
            await pilot.pause()
            return url_bar_text(app), app.selected_job().url

    shown, expected = _run(scenario())
    assert shown == expected
    assert shown.startswith("https://")


def test_url_bar_is_a_single_row(conn, loaded):
    # Fixed height, never auto: a wrapped URL would put half the text on a
    # second row and break the drag.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            return app.query_one("#url-bar").region

    region = _run(scenario())
    assert region.height == 1


def test_url_bar_spans_the_full_width(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            return app.query_one("#url-bar").region, app.screen.size

    region, screen = _run(scenario())
    assert region.width == screen.width


def test_url_bar_sits_above_the_footer(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            return (
                app.query_one("#url-bar").region,
                app.query_one("Footer").region,
                app.query_one("#body").region,
            )

    bar, footer, body = _run(scenario())
    assert bar.y < footer.y
    assert body.bottom <= bar.y + 1


def test_url_bar_updates_with_the_selection(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            app.current_table().move_cursor(row=0)
            await pilot.pause()
            first = url_bar_text(app)
            app.current_table().move_cursor(row=1)
            await pilot.pause()
            return first, url_bar_text(app)

    first, second = _run(scenario())
    assert first != second
    assert first.startswith("https://") and second.startswith("https://")


def test_url_bar_says_so_when_no_url_is_stored(conn):
    upsert.upsert_job(
        conn,
        Job(source="ashby:ramp", external_id="1", company="Acme", title="Eng",
            description="body", url=None),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            return url_bar_text(app)

    assert "no URL" in _run(scenario())


def test_url_bar_is_empty_with_no_selection(conn):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            return url_bar_text(app)

    assert _run(scenario()) == ""


def test_detail_pane_no_longer_shows_the_url(conn, loaded):
    # It moved to the bar so the line can be selected on its own.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            return detail_text(app), app.selected_job().url

    pane_text, url = _run(scenario())
    assert url not in pane_text


# ---------------------------------------------------------------------------
# paging
# ---------------------------------------------------------------------------

@pytest.fixture
def paged_db(tmp_path):
    """A separate database with 25 jobs.

    Deliberately NOT the shared ``conn`` fixture: inserting rows into a database
    other tests share is what made unrelated tests fail intermittently when this
    feature was first attempted.
    """
    from jobecosystem.core import db as core_db

    conn = core_db.connect(tmp_path / "paged.db")
    for i in range(25):
        upsert.upsert_job(
            conn,
            Job(source="ashby:ramp", external_id=f"{i:02d}", company="Acme",
                title=f"Engineer {i:02d}", description="body text",
                url=f"https://x.test/{i:02d}"),
        )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def page_ids(app) -> list[str]:
    """External ids on the currently displayed page."""
    return [r.job.external_id for r in app.rows]


def test_paging_forward_shows_the_next_rows(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            first = page_ids(app)
            await pilot.press(">")
            await pilot.pause()
            return first, page_ids(app), app._page

    first, second, page = _run(scenario())
    assert page == 1
    assert len(first) == len(second) == 10
    assert not set(first) & set(second)


def test_paging_reaches_every_row_once(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        seen = []
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            for _ in range(3):
                seen.extend(page_ids(app))
                await pilot.press(">")
                await pilot.pause()
        return seen

    seen = _run(scenario())
    assert len(seen) == 25
    assert len(set(seen)) == 25


def test_paging_stops_at_the_last_page(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            for _ in range(8):
                await pilot.press(">")
                await pilot.pause()
            return app._page, app.page_count, len(app.rows)

    page, pages, rows = _run(scenario())
    assert pages == 3
    assert page == pages - 1
    assert rows == 5          # the partial last page


def test_paging_backwards_returns_to_the_start(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            await pilot.press(">")
            await pilot.pause()
            await pilot.press(">")
            await pilot.pause()
            middle = app._page          # zero-based: two forwards is page 2
            for _ in range(5):
                await pilot.press("<")
                await pilot.pause()
            return middle, app._page, page_ids(app)

    middle, first, rows = _run(scenario())
    assert middle == 2
    assert first == 0
    assert len(rows) == 10


def test_header_reports_the_page(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            first = app.sub_title
            await pilot.press(">")
            await pilot.pause()
            return first, app.sub_title

    first, second = _run(scenario())
    assert "10 of 25 shown (page 1/3)" in first
    assert "page 2/3" in second


def test_switching_tabs_returns_to_the_first_page(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            await pilot.press(">")
            await pilot.pause()
            moved = app._page
            await pilot.press("]")
            await pilot.pause()
            return moved, app._page

    moved, after = _run(scenario())
    assert moved == 1
    assert after == 0


def test_searching_returns_to_the_first_page(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            await pilot.press(">")
            await pilot.pause()
            app.query_one("TabbedContent").active = "search"
            await pilot.pause()
            app.query_one("#search-input").value = "engineer"
            await pilot.press("enter")
            await pilot.pause()
            return app._page

    assert _run(scenario()) == 0


def test_a_status_change_keeps_the_page(paged_db):
    # Acting on a job used to reset _page to 0, throwing a paged list back to
    # the top. The page is not part of the mutation.
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            await pilot.press(">")
            await pilot.pause()
            first_before = page_ids(app)[0]
            app.current_table().move_cursor(row=3)
            await pilot.press("s")
            await pilot.pause()
            return app._page, first_before, page_ids(app)

    page, first_before, after = _run(scenario())
    assert page == 1
    assert len(after) == 10
    assert after[0] == first_before


def test_a_status_change_keeps_the_scroll_offset(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=25)
        async with app.run_test(size=(150, 16)) as pilot:
            await pilot.pause()
            table = app.current_table()
            table.scroll_y = 8
            table.move_cursor(row=8, scroll=False)
            await pilot.pause()
            before = table.scroll_y
            app.action_set_status("seen")
            await pilot.pause()
            await pilot.pause()
            return before, table.scroll_y

    before, after = _run(scenario())
    assert before > 0, "the fixture must actually be scrolled"
    assert after == before


def test_acting_on_the_last_row_keeps_the_cursor_near_the_end(conn, loaded):
    # The restore used to fall through to row 0 when the acted-on row was the
    # last one and the list shrank by one.
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.current_table().move_cursor(row=4)
            await pilot.pause()
            app.action_set_status("seen")
            await pilot.pause()
            return len(app.rows), app.current_table().cursor_row

    remaining, cursor_row = _run(scenario())
    assert remaining == 4
    assert cursor_row == 3          # clamped to the new last row, not 0


def test_a_status_change_selects_the_neighbouring_job(paged_db):
    # The cursor follows the position, not the row key: the acted-on job leaves
    # the view, so the row that shifts into its place is the one to select.
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.current_table().move_cursor(row=4)
            await pilot.pause()
            acting_on = app.selected_job().external_id
            neighbour = app.rows[5].job.external_id
            app.action_set_status("seen")
            await pilot.pause()
            return acting_on, neighbour, app.selected_job().external_id

    acting_on, neighbour, selected = _run(scenario())
    assert selected == neighbour
    assert selected != acting_on


def test_paging_past_the_end_is_a_no_op(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            await pilot.press("<")
            await pilot.pause()
            return app._page

    assert _run(scenario()) == 0


def test_a_result_set_smaller_than_a_page_is_one_page(conn, loaded):
    # Uses the existing small fixture deliberately: five jobs, one page.
    async def scenario():
        app = tui.JobApp(conn, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            await pilot.press(">")
            await pilot.pause()
            return app._page, app.page_count, app.sub_title

    page, pages, subtitle = _run(scenario())
    assert page == 0
    assert pages == 1
    assert "1 page" in subtitle
    assert "page 1/1" not in subtitle


def test_url_bar_follows_the_page(paged_db):
    # The bar must not keep showing a job from the previous page.
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            first = app.query_one("#url-bar").render().plain
            await pilot.press(">")
            await pilot.pause()
            return first, app.query_one("#url-bar").render().plain

    first, second = _run(scenario())
    assert first.startswith("https://")
    assert second.startswith("https://")
    assert first != second


def test_paging_keys_are_shown_in_the_footer():
    shown = {
        (b.key_display or b.key)
        for b in tui.JobApp.BINDINGS
        if b.show
    }
    assert {">", "<"} <= shown


# ---------------------------------------------------------------------------
# header counting is cached
# ---------------------------------------------------------------------------

def test_header_count_is_not_recomputed_on_every_refresh(paged_db, monkeypatch):
    # Counting a view was a full scan and ran twice per refresh, adding about a
    # second per keypress on a large database.
    from jobecosystem.triage import queries as qq

    calls = []
    original = qq.count_jobs

    def counting(*args, **kwargs):
        calls.append(kwargs.get("view") or kwargs.get("status") or "all")
        return original(*args, **kwargs)

    monkeypatch.setattr(tui.q, "count_jobs", counting)

    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            after_mount = len(calls)
            app.refresh_view()
            app.refresh_view()
            app.refresh_view()
            return after_mount, len(calls)

    after_mount, later = _run(scenario())
    assert after_mount >= 1
    assert later == after_mount, "the count should be cached across refreshes"


def test_a_status_change_invalidates_the_cached_count(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            before = app.count_rows("new")
            await pilot.press("a")          # mark the selected job applied
            await pilot.pause()
            return before, app.count_rows("new")

    before, after = _run(scenario())
    assert after == before - 1


def test_count_cache_is_scoped_to_the_tab(paged_db):
    async def scenario():
        app = tui.JobApp(paged_db, page_size=10)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            new = app.count_rows("new")
            app.query_one("TabbedContent").active = "all"
            await pilot.pause()
            everything = app.count_rows("all")
            app.query_one("TabbedContent").active = "new"
            await pilot.pause()
            return new, everything, app.count_rows("new")

    new, everything, new_again = _run(scenario())
    assert new_again == new
    assert everything >= new


# ---------------------------------------------------------------------------
# title truncation
# ---------------------------------------------------------------------------

def test_truncate_leaves_short_text_alone():
    assert tui.truncate("Short", 80) == "Short"


def test_truncate_leaves_exactly_the_width_alone():
    text = "x" * 80
    assert tui.truncate(text, 80) == text


def test_truncate_cuts_to_the_width_including_the_marker():
    result = tui.truncate("x" * 200, 80)
    assert tui._cell_len(result) == 80
    assert result.endswith(tui.ELLIPSIS)


def test_truncate_marks_the_cut_so_it_is_visible():
    # A silently clipped title reads as a complete one.
    assert tui.truncate("A" * 100, 20)[-1] == tui.ELLIPSIS


def test_truncate_does_not_leave_a_trailing_space_before_the_marker():
    # The character at the cut point is often a space; keeping it would look
    # like the title ended early.
    assert not tui.truncate("word " * 40, 20).endswith(" " + tui.ELLIPSIS)


def test_truncate_of_empty_text():
    assert tui.truncate("", 80) == ""


def test_truncate_with_a_zero_width():
    assert tui.truncate("anything", 0) == ""


def test_title_column_is_fixed_at_the_configured_width(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            table = app.current_table()
            return table.columns[list(table.columns)[5]].width

    assert _run(scenario()) == tui.TITLE_WIDTH


def test_score_column_precedes_the_wide_columns(conn, loaded):
    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test() as pilot:
            await pilot.pause()
            return [str(column.label) for column in app.current_table().ordered_columns]

    labels = _run(scenario())
    # Scrollable-overflow must never be what stands between a row and its score.
    assert labels.index("Score") < labels.index("Company")
    assert labels.index("Score") < labels.index("Title")


def test_a_long_title_is_truncated_in_the_table(conn):
    long_title = "Manager Technical Consulting " + "and more detail " * 12
    upsert.upsert_job(
        conn,
        Job(source="ashby:ramp", external_id="long", company="Acme",
            title=long_title, description="body"),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            table = app.current_table()
            for index in range(table.row_count):
                cells = table.get_row_at(index)
                if str(cells[5]).startswith("Manager"):
                    return str(cells[5])
            return None

    shown = _run(scenario())
    assert shown is not None
    assert tui._cell_len(shown) <= tui.TITLE_WIDTH
    assert shown.endswith(tui.ELLIPSIS)


def test_truncation_does_not_touch_the_stored_title(conn):
    long_title = "x" * 200
    result = upsert.upsert_job(
        conn,
        Job(source="ashby:ramp", external_id="1", company="Acme",
            title=long_title, description="body"),
    )
    conn.commit()

    async def scenario():
        app = tui.JobApp(conn)
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()

    _run(scenario())
    # The list shows a cut title; the database and the detail pane keep it whole.
    stored = conn.execute(
        "SELECT title FROM jobs WHERE id = ?", (result.job_id,)
    ).fetchone()[0]
    assert stored == long_title


def test_truncate_measures_display_columns_not_characters():
    # A CJK character is two columns wide, so 60 of them fill a 120-column field.
    # Counting characters would overflow by one column each.
    text = "消" * 100
    result = tui.truncate(text, 60)
    # At most the width, and never over it: with 2-column characters the last
    # one that would land exactly on the boundary is left off, so the result can
    # be a column short. Under-filling is safe; overflowing is the bug.
    assert tui._cell_len(result) <= 60
    assert tui._cell_len(result) >= 60 - 2
    assert len(result) < 60, "wide characters mean fewer of them fit"


def test_truncate_of_mixed_width_text_fits():
    text = "Account Executive, Enterprise Sales (BtoC/消費財、流通、物流、通信、メディア)"
    result = tui.truncate(text, 60)
    assert tui._cell_len(result) <= 60
    assert result.endswith(tui.ELLIPSIS)


def test_truncate_never_splits_a_wide_character_across_the_boundary():
    # An odd budget with a wide character next must stop before it, not take
    # half of it.
    text = "ab" + "消" * 40
    result = tui.truncate(text, 5)
    assert tui._cell_len(result) <= 5


def test_truncate_leaves_short_cjk_text_alone():
    text = "消費財"
    assert tui.truncate(text, 60) == text


def test_cell_len_agrees_with_len_for_ascii():
    assert tui._cell_len("plain ascii") == len("plain ascii")
