"""Textual triage interface.

A thin shell over :mod:`jobecosystem.triage.queries`: the app holds no SQL and
no business rules, it calls the query layer and renders the result. That keeps
the interesting behaviour testable without a terminal.

Layout
------

    ┌─ header: database, totals, description backlog ─────────────┐
    ├─ tabs: New | Today | Reposted | Stale | Search | Similar ────┤
    │  a DataTable of jobs, one row each                          │
    ├─ detail pane: the selected job's full text ──────────────────┤
    └─ footer: keys ───────────────────────────────────────────────┘

Keys
----

    /        focus the search box (on the Search tab)
    ]  [     next / previous tab
    s        mark the selected job seen
    a        mark the selected job applied
    z        mark the selected job new again
    x        hide the selected job
    ctrl+0-5 rate the selected job (same key again clears it)
    d        fetch the selected job's description (if missing)
    c        copy the selected job's URL to the local clipboard
    ctrl+e   find jobs similar to the selected one
    r        reload the current tab
    q        quit

Statuses are set directly rather than cycled: one key per state, so the result
is never ambiguous and "back to new" is a single press. Marking a job changes
which tab it appears in, but never destroys it -- see the New, Applied and All
tabs.

``c`` exists because copying text out of a full-screen application is otherwise
awkward: Textual enables mouse reporting, so the terminal cannot select text
itself, and whether Shift or Option suspends that depends on the terminal. OSC 52
(the mechanism behind ``copy_to_clipboard``) puts the text on the clipboard of
the machine you are sitting at, over SSH included.

Embedding search is a separate action, not a live filter: typing in the Search
box matches keywords, while ``ctrl+e`` on a job finds others like it. The query
vector is embedded once per submit and cached for the session, per the earlier
decision that live search is not wanted.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, Sequence

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Static, TabbedContent, TabPane

try:  # pragma: no cover - exercised by the import, not by a test
    from textual._cells import cell_len as _cell_len
except ImportError:  # pragma: no cover - only on a Textual that moved it
    def _cell_len(text: str) -> int:
        """Fallback width measure, counting every character as one cell.

        Wrong for wide CJK characters, which occupy two columns, so truncation
        would overflow by one column per wide character. Only reached if
        Textual's private ``_cells`` module disappears.
        """
        return len(text)

from ...core import db as core_db
from ...core.models import Job
from .. import queries as q

#: Rows loaded per view. The store holds thousands of jobs and the table is for
#: reading, so a bounded load keeps the UI responsive; narrowing happens through
#: search and filters rather than scrolling.
DEFAULT_PAGE_SIZE = 300

#: Characters of a job title shown in the list before it is cut off.
#:
#: Sized to fit the list pane, not to show the whole title: the pane is about
#: 3/5 of the terminal, so ~90 columns at a 150-wide terminal, and the other six
#: columns need roughly 34 of those. A larger value makes DataTable scroll
#: horizontally, which hides the leading columns and shows a window into the
#: title rather than its start. Titles are also long (one Workday posting runs to
#: 180 characters), so truncation is expected; the detail pane shows it in full.
TITLE_WIDTH = 60

#: Shown in place of the removed characters.
ELLIPSIS = "\u2026"

TABS = (
    ("new", "New"),
    ("today", "Today"),
    ("applied", "Applied"),
    ("all", "All"),
    ("reposted", "Reposted"),
    ("stale", "Stale"),
    ("search", "Search"),
    ("similar", "Similar"),
)

#: No longer a cycle: each status has its own key (see BINDINGS), so setting a
#: status is one keystroke rather than pressing ``s`` until the right one shows.
VALID_STATUSES = ("new", "seen", "applied", "hidden")


def truncate(text: str, width: int) -> str:
    """Cut ``text`` to ``width`` display columns, marking the cut with an ellipsis.

    Measured in terminal cells rather than characters, because they differ: a CJK
    character occupies two columns, so 60 characters of Japanese is about 85
    columns and would overflow a 60-column field. ``len()`` is wrong for exactly
    the titles this app sees most.

    The marker is counted in the width, so the result never exceeds it. Text that
    already fits is returned unchanged, and a cut never leaves a trailing space
    before the marker.
    """
    if width <= 0:
        return ""
    if _cell_len(text) <= width:
        return text

    budget = width - _cell_len(ELLIPSIS)
    kept: list[str] = []
    used = 0
    for character in text:
        size = _cell_len(character)
        if used + size > budget:
            break
        kept.append(character)
        used += size
    return "".join(kept).rstrip() + ELLIPSIS


@dataclass(slots=True)
class Row:
    """A job plus the score shown beside it, if any."""

    job: Job
    score: float | None = None


class SearchInput(Input):
    """The keyword box, with navigation keys forwarded to the list.

    Two problems solved here, both because ``Input`` claims keys for the text
    cursor:

    * ``]``/``[`` must switch tabs even while the box has focus, or there is no
      way back off the Search tab. A public ``on_key`` wins this race because
      Textual dispatches public handlers before the private ``_on_key`` that
      ``Input`` uses to swallow printable keys -- ordinary bindings lose it.
    * Up/down (and page keys) must move the selection in the table rather than
      the caret, since the box is a single line and there is nowhere to move to.
    """

    #: Keys that belong to the list, not the caret.
    FORWARDED = {
        "up": "cursor_up",
        "down": "cursor_down",
        "pageup": "page_up",
        "pagedown": "page_down",
        "home": "scroll_home",
        "end": "scroll_end",
    }

    def on_key(self, event) -> None:
        """Handle tab switching and forwarding, stopping anything consumed."""
        if event.key == "right_square_bracket":
            self.app.action_next_tab()
        elif event.key == "left_square_bracket":
            self.app.action_prev_tab()
        elif event.key in self.FORWARDED:
            table = self.app.current_table()
            # Focus follows the keypress: once you start navigating results the
            # caret is no longer of interest, so re-submitting means pressing
            # slash again.
            table.focus()
            getattr(table, f"action_{self.FORWARDED[event.key]}")()
        else:
            return
        event.stop()
        event.prevent_default()


class JobsTable(DataTable):
    """DataTable preconfigured for job rows."""

    def on_mount(self) -> None:
        self.cursor_type = "row"
        self.zebra_stripes = True
        keys = self.add_columns(
            "ID",
            "Rating",
            "Status",
            "Company",
            "Title",
            "Location",
            "Score",
        )
        # Fixed rather than auto: the cap is the point, and a fixed column keeps
        # the freed width available to the columns that were also being clipped.
        self.columns[keys[4]].width = TITLE_WIDTH


class JobDetail(VerticalScroll):
    """The detail pane: whatever is known about the selected job.

    Owns a single ``Static`` and rewrites its content, rather than mounting and
    removing children per selection. Two reasons, both learned the hard way:
    ``remove_children()`` returns an awaitable, and leaving it pending made the
    pane render blank; and a mount/remove cycle per keystroke churns the DOM for
    no benefit when only the text changes.
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="detail-text")

    def show(
        self,
        row: Row | None,
        *,
        stale: bool = False,
        duplicate_count: int = 0,
        is_reposted: bool = False,
    ) -> None:
        """Render a job, or the empty-state prompt. Synchronous by design."""
        self.query_one("#detail-text", Static).update(
            _detail_text(
                row,
                stale=stale,
                duplicate_count=duplicate_count,
                is_reposted=is_reposted,
            )
        )


def _detail_text(
    row: Row | None,
    *,
    stale: bool = False,
    duplicate_count: int = 0,
    is_reposted: bool = False,
) -> str:
    """The detail pane's content for a row, as Textual markup.

    A free function so the formatting can be tested without a running app.

    ``stale``, ``is_reposted`` and ``duplicate_count`` are passed in rather than
    read off the job because none of them is a column: staleness is decided
    against the scrape anchor, and a repost can be either a row re-seen after a
    gap (``repost_count``) or the same posting listed under a second id
    (``duplicate_count``). Both come from the views, so the pane agrees with the
    tabs rather than reimplementing them.
    """
    if row is None:
        return "Select a job to see its details."

    job = row.job
    lines = [
        f"[b]{job.title}[/b]",
        f"{job.company}  \u00b7  {job.location or 'location unknown'}",
        f"source: {job.source}   id: {job.external_id}",
    ]
    if row.score is not None:
        lines.append(f"similarity: {row.score:.4f}")
    if job.salary_range:
        lines.append(f"salary: {job.salary_range}")
    rating = "\u2014" if job.rating is None else str(job.rating)
    lines.append(f"status: {job.status}   rating: {rating}")

    # Flags the row cannot answer on its own. Shown only when true, so the pane
    # stays short; a warning style because both mean "look before you spend time
    # here".
    flags = []
    if stale:
        flags.append("stale")
    if is_reposted:
        flags.append(f"reposted [{job.repost_count}\u00d7 seen, {duplicate_count} duplicate]")
    if flags:
        lines.append(f"[yellow]{'   '.join(flags)}[/yellow]")

    meta = []
    if job.posted_at:
        meta.append(f"posted {job.posted_at}")
    meta.append(f"first seen {job.first_seen_at}")
    if job.repost_count:
        meta.append(f"reseen {job.repost_count}\u00d7")
    lines.append("   ".join(meta))
    lines.append("")

    if job.description:
        lines.append(job.description)
    else:
        lines.append("[dim]No description yet. Press d to fetch it.[/dim]")

    return "\n".join(lines)


class JobApp(App[None]):
    """The triage application."""

    CSS = """
    Screen { layout: vertical; }
    /* The tabs and the detail pane sit side by side; both must be children of
       #body for that to happen, or the tabs take the full height and squash
       the pane to nothing. */
    #body { height: 1fr; }
    #tabs { width: 3fr; }
    #detail { width: 2fr; border-left: solid $panel; padding: 0 1; }
    #search-bar { height: auto; padding: 0 1; }
    JobsTable { height: 1fr; }
    TabPane { height: 1fr; }
    /* The URL bar exists to be selected with the mouse: one short line, the URL
       and nothing else, so a drag captures exactly it and no line wrapping
       splits it. keep it fixed-height (never auto) so long URLs are truncated
       rather than wrapped onto a second row. */
    #url-bar {
        height: 1;
        padding: 0 1;
        background: $panel;
        text-overflow: ellipsis;
    }
    """

    # Plain digits are left alone: the search box wants them, and rating lives
    # on ctrl+digit so the two cannot collide.
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reload", "Reload"),
        Binding("slash", "focus_search", "Search"),
        Binding("s", "set_status('seen')", "Seen"),
        Binding("a", "set_status('applied')", "Applied"),
        Binding("z", "set_status('new')", "New"),
        Binding("x", "set_status('hidden')", "Hide"),
        Binding("d", "fetch_description", "Describe"),
        Binding("c", "copy_url", "Copy URL"),
        Binding("ctrl+e", "similar_to_selected", "More like this"),
        Binding("right_square_bracket", "next_tab", "Next tab", key_display="]"),
        Binding("left_square_bracket", "prev_tab", "Prev tab", key_display="["),
        Binding("greater_than_sign", "next_page", "Next page", key_display=">"),
        Binding("less_than_sign", "prev_page", "Prev page", key_display="<"),
        # One footer entry for the whole rating range; six would crowd the bar.
        Binding("ctrl+0", "rate(0)", "Rate 0-5", key_display="ctrl+0..5", show=True),
        *[
            Binding(f"ctrl+{n}", f"rate({n})", f"Rate {n}", show=False)
            for n in range(1, 6)
        ],
    ]

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        embed_query: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        super().__init__()
        self.conn = conn
        self.page_size = page_size
        self._embed_query = embed_query
        self.rows: list[Row] = []
        self._query_cache: dict[str, list[float]] = {}
        #: Ids the Stale view selects, and repost evidence per id. Populated on
        #: refresh; see queries.stale_job_ids and queries.repost_info.
        self._stale_ids: set[int] = set()
        self._repost_counts: dict[int, int] = {}
        self._active_tab = "new"
        self._search_text = ""
        self._similar_to: int | None = None
        self._similar_rows: list[Row] = []
        self._similar_for: str | None = None
        #: Zero-based page within the active tab. Reset whenever the result set
        #: changes -- a page number means nothing across different queries.
        self._page = 0
        #: Cached row count for the active tab, and the key it was computed for.
        #: Counting a view is a full scan (a second on a large table), and the
        #: header used to run it twice per refresh; the key means a stale count
        #: can never be shown for a different tab or query.
        self._count_cache: tuple[tuple[str, str], int] | None = None

    # -- composition -------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # Tabs and detail pane are siblings inside #body, so they lay out
        # horizontally. Yielding TabbedContent at the top level instead made it
        # consume the whole height and pushed the pane off-screen.
        with Horizontal(id="body"):
            with TabbedContent(initial="new", id="tabs"):
                for key, label in TABS:
                    with TabPane(label, id=key):
                        if key == "search":
                            with Horizontal(id="search-bar"):
                                yield SearchInput(
                                    placeholder="keywords (Enter to search)",
                                    id="search-input",
                                )
                        yield JobsTable(id=f"table-{key}")
            yield JobDetail(id="detail")
        yield Static("", id="url-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Job ecosystem"
        self.refresh_view()

    # -- data loading ------------------------------------------------------

    def refresh_view(
        self,
        *,
        keep_cursor: bool = False,
        focus_table: bool = True,
        reset_page: bool = True,
    ) -> None:
        """Reload the active tab at the current page and re-render.

        ``focus_table=False`` when a caller wants to leave focus elsewhere --
        notably the search box, which must keep focus after a search returns.
        ``reset_page=False`` for a pager move, which has already set the page.

        With ``keep_cursor`` the viewport is pinned across the rebuild: the
        highlighted job is found again by id and the scroll offset is put back.
        A mutation removes a row and shifts every later one, so without this the
        table would jump back to the top on each keypress.
        """
        tab = self._active_tab
        if reset_page:
            self._page = 0

        table = self.query_one(f"#table-{tab}", JobsTable)
        if keep_cursor:
            selected = self.selected_row()
            previous = self.selected_row_id()
            selected_id = selected.job.id if selected is not None else None
            saved_scroll = table.scroll_offset
        else:
            previous = None
            selected_id = None
            saved_scroll = None

        try:
            rows = self.load_rows(tab)
        except q.QueryError as error:
            self.notify(str(error), severity="error")
            rows = []

        self.rows = rows
        # Refreshed alongside the rows: stale-ness and repost-ness are decided by
        # the views, not by anything on the row, so the pane needs them looked up.
        self._stale_ids = q.stale_job_ids(self.conn)
        self._repost_counts = q.repost_info(self.conn)
        table.clear()
        for row in rows:
            # Keyed by job id, not position: after a mutation the rows shift, and
            # a positional key would point at a different job than before.
            table.add_row(*self.format_row(row), key=str(row.job.id))

        # Prefer the same job over the same index -- the row it occupied may now
        # hold its neighbour. When the job has left the view, keep the cursor at
        # the nearest surviving row rather than snapping to the top.
        restore = self.index_of_job(selected_id) if selected_id is not None else None
        if restore is None and previous is not None and rows:
            restore = min(previous, len(rows) - 1)
        if restore is not None:
            table.move_cursor(row=restore)
        elif rows:
            table.move_cursor(row=0)

        # move_cursor scrolls the cursor into view on the next refresh; run after
        # it so the restored offset, not the cursor, is what the viewport shows.
        if saved_scroll is not None:
            self.call_after_refresh(self._restore_scroll, table, saved_scroll.y)

        if focus_table:
            table.focus()
        self.update_detail()
        self.update_header()

    def index_of_job(self, job_id: int | None) -> int | None:
        """Row index of a job in the current result set, or ``None`` if absent."""
        if job_id is None:
            return None
        for index, row in enumerate(self.rows):
            if row.job.id == job_id:
                return index
        return None

    def _restore_scroll(self, table: JobsTable, y: float) -> None:
        """Put a saved scroll offset back after a rebuild.

        Scheduled after the cursor's own deferred scroll so it has the last word;
        the new content may be shorter, which the ``scroll_y`` setter clamps.
        """
        table.scroll_y = y

    def load_rows(self, tab: str) -> list[Row]:
        """Fetch the current page for one tab, through the query layer only."""
        size = self.page_size
        offset = self._page * size
        if tab == "new":
            return [Row(j) for j in q.jobs_unseen(self.conn, limit=size, offset=offset)]
        if tab == "today":
            return [Row(j) for j in q.jobs_today(self.conn, limit=size, offset=offset)]
        if tab == "applied":
            return [
                Row(j)
                for j in q.jobs_by_status(
                    self.conn, "applied", limit=size, offset=offset
                )
            ]
        if tab == "all":
            return [Row(j) for j in q.jobs_all(self.conn, limit=size, offset=offset)]
        if tab == "reposted":
            return [
                Row(j) for j in q.jobs_reposted(self.conn, limit=size, offset=offset)
            ]
        if tab == "stale":
            return [Row(j) for j in q.jobs_stale(self.conn, limit=size, offset=offset)]
        if tab == "search":
            if not self._search_text:
                return []
            return [
                Row(j)
                for j in q.search_jobs(
                    self.conn, self._search_text, limit=size, offset=offset
                )
            ]
        if tab == "similar":
            # Similarity is ranked in Python, so this tab is the one whose
            # results are paged after the fact rather than by the query.
            return self.similar_rows()[offset : offset + size]
        return []

    def similar_rows(self) -> list[Row]:
        """The whole Similar-tab result set, before paging.

        "More like this job" re-queries from the stored vector; a text search
        re-renders the results it already computed.
        """
        if self._similar_to is not None:
            return [
                Row(found.job, found.score)
                for found in q.similar_to_job(self.conn, self._similar_to)
            ]
        return list(self._similar_rows)

    def count_rows(self, tab: str) -> int:
        """Total rows the tab holds, for the page indicator.

        Cached per (tab, query), because counting a view is a full scan -- about
        a second on a 15k-row database -- and the header needs it on every
        refresh. The cache key includes the search text so switching queries
        cannot show a stale total.
        """
        key = (tab, self._search_text if tab == "search" else "")
        if self._count_cache is not None and self._count_cache[0] == key:
            return self._count_cache[1]

        total = self.count_rows_uncached(tab)
        self._count_cache = (key, total)
        return total

    def count_rows_uncached(self, tab: str) -> int:
        """Count the rows a tab holds, straight from the query layer."""
        if tab == "new":
            return q.count_jobs(self.conn, view="jobs_unseen")
        if tab == "today":
            return q.count_jobs(self.conn, view="jobs_today")
        if tab == "applied":
            return q.count_jobs(self.conn, status="applied")
        if tab == "all":
            return q.count_jobs(self.conn)
        if tab == "reposted":
            return q.count_jobs(self.conn, view="jobs_reposted")
        if tab == "stale":
            return q.count_jobs(self.conn, view="jobs_stale")
        if tab == "search":
            return (
                q.count_jobs(self.conn, text=self._search_text)
                if self._search_text
                else 0
            )
        if tab == "similar":
            return len(self.similar_rows())
        return 0

    @property
    def page_count(self) -> int:
        """Pages in the active tab. Never zero, so indicators read 1/1."""
        total = self.count_rows(self._active_tab)
        return max(1, -(-total // self.page_size))

    @staticmethod
    def format_row(row: Row) -> tuple[str, str, str, str, str, str, str]:
        job = row.job
        return (
            str(job.id),
            "—" if job.rating is None else str(job.rating),
            job.status,
            job.company,
            truncate(job.title, TITLE_WIDTH),
            job.location or "",
            "" if row.score is None else f"{row.score:.3f}",
        )

    def update_header(self) -> None:
        """Say where you are: this page, and how much is in the database.

        The page indicator is the point of the change it accompanies -- a capped
        list is otherwise indistinguishable from a short one, which is exactly
        the confusion a silent 300-row default caused.
        """
        progress = q.description_progress(self.conn)
        total = progress.get("total_jobs") or 0
        pending = progress.get("pending") or 0
        shown = len(self.rows)
        in_tab = self.count_rows(self._active_tab)
        pages = max(1, -(-in_tab // self.page_size))
        where = f"page {self._page + 1}/{pages}" if pages > 1 else "1 page"
        self.sub_title = (
            f"{shown} of {in_tab} shown ({where}) · {total} jobs ·"
            f" {pending} awaiting descriptions"
        )

    def update_detail(self) -> None:
        """Re-render the detail pane and the URL bar for the current selection."""
        pane = self.query_one("#detail", JobDetail)
        bar = self.query_one("#url-bar", Static)
        row = self.selected_row()
        if row is None:
            pane.show(None)
            bar.update("")
            return
        job = row.job
        # Plain text, no markup and no dim styling: this line is meant to be
        # selected with the mouse, and styles add nothing to a copy.
        bar.update(job.url or "[dim]no URL stored for this job[/dim]")
        pane.show(
            row,
            stale=job.id in self._stale_ids,
            duplicate_count=self._repost_counts.get(job.id, 0),
            is_reposted=job.id in self._repost_counts,
        )

    # -- selection helpers -------------------------------------------------

    def current_table(self) -> JobsTable:
        return self.query_one(f"#table-{self._active_tab}", JobsTable)

    def selected_row_id(self) -> int | None:
        """Index of the highlighted row, or ``None``.

        ``cursor_row`` is -1 on an empty or freshly-cleared table, which is not
        an error -- it just means nothing is selected.
        """
        if not self.rows:
            return None
        table = self.current_table()
        row = table.cursor_row
        if row < 0 or row >= len(self.rows):
            return None
        return row

    def selected_row(self) -> Row | None:
        index = self.selected_row_id()
        return self.rows[index] if index is not None else None

    def selected_job(self) -> Job | None:
        row = self.selected_row()
        return row.job if row is not None else None

    # -- events ------------------------------------------------------------

    @on(TabbedContent.TabActivated)
    def tab_changed(self, event: TabbedContent.TabActivated) -> None:
        tab_id = (event.pane.id or "new").replace("table-", "")
        self._active_tab = tab_id
        # On Search the box is the only thing to do, so focus it and leave it
        # focused: refresh_view must not steal focus back to the table.
        if tab_id == "search":
            self.refresh_view(focus_table=False)
            self.query_one("#search-input", Input).focus()
        else:
            self.refresh_view()

    @on(Input.Submitted, "#search-input")
    def run_search(self, event: Input.Submitted) -> None:
        self._search_text = event.value.strip()
        # Focus moves to the results: having just searched, that is what you want
        # to act on, and leaving focus in the box would trap the arrow keys in
        # the text cursor. Slash (or Ctrl+E, or clicking) returns to the box.
        self.refresh_view()
    @on(DataTable.RowHighlighted)
    def row_highlighted(self) -> None:
        self.update_detail()

    # -- actions -----------------------------------------------------------

    def action_reload(self) -> None:
        self.refresh_view(keep_cursor=True)

    def go_to_tab(self, index: int) -> None:
        """Switch to the tab at ``index``, clamped to the available tabs."""
        index = max(0, min(index, len(TABS) - 1))
        key = TABS[index][0]
        if key == self._active_tab:
            return
        self.query_one(TabbedContent).active = key

    def action_next_page(self) -> None:
        """Show the next page, if there is one."""
        self.go_to_page(self._page + 1)

    def action_prev_page(self) -> None:
        """Show the previous page, if there is one."""
        self.go_to_page(self._page - 1)

    def go_to_page(self, page: int) -> None:
        """Move to a page, clamped to the range."""
        last = self.page_count - 1
        page = max(0, min(page, last))
        if page == self._page:
            self.notify("Already on the first page" if page == 0 else "Already on the last page")
            return
        self._page = page
        self.refresh_view(reset_page=False)
        self.notify(f"Page {page + 1} of {last + 1} ({len(self.rows)} rows)")

    def action_next_tab(self) -> None:
        current = [key for key, _ in TABS].index(self._active_tab)
        self.go_to_tab((current + 1) % len(TABS))

    def action_prev_tab(self) -> None:
        current = [key for key, _ in TABS].index(self._active_tab)
        self.go_to_tab((current - 1) % len(TABS))

    def action_focus_search(self) -> None:
        self.query_one(TabbedContent).active = "search"
        self.query_one("#search-input", Input).focus()

    def action_set_status(self, status: str) -> None:
        """Mark the selected job with one status.

        One key per status rather than a cycle: cycling meant pressing ``s``
        repeatedly and never being sure which status you had landed on, and it
        made "back to new" as many presses as there are statuses.
        """
        job = self.selected_job()
        if job is None:
            self.notify("No job selected", severity="warning")
            return
        if job.status == status:
            self.notify(f"Already {status}")
            return
        q.set_status(self.conn, job.id, status)
        # A status change moves the row between tabs, so the cached totals are
        # no longer true. Only mutations invalidate; a plain refresh does not.
        self._count_cache = None
        self.notify(f"{job.title[:40]}: {status}")
        # A mutation changes the rows but not which page you are reading: keeping
        # the page and the scroll offset is what stops the jump to the top.
        self.refresh_view(keep_cursor=True, reset_page=False)

    def action_rate(self, rating: int) -> None:
        job = self.selected_job()
        if job is None:
            return
        # Pressing the current rating clears it, so a mistake is undoable.
        new_rating = None if job.rating == rating else rating
        q.set_rating(self.conn, job.id, new_rating)
        self.refresh_view(keep_cursor=True, reset_page=False)

    def action_similar_to_selected(self) -> None:
        job = self.selected_job()
        if job is None:
            return
        self._similar_to = job.id
        self._similar_rows = []
        self._active_tab = "similar"
        self.query_one(TabbedContent).active = "similar"
        self.refresh_view()
        if not self.rows:
            self.notify(
                "No embedding for this job yet; run the embedder first",
                severity="warning",
            )

    def action_copy_url(self) -> None:
        """Copy the selected job's URL to the local clipboard.

        Uses OSC 52, so the text lands on the clipboard of whatever terminal you
        are sitting at, even over SSH -- which is the only way to get text out of
        a full-screen app reliably. Mouse reporting stops the terminal from
        selecting text itself, and neither Shift nor Option bypasses that in
        every terminal.
        """
        job = self.selected_job()
        if job is None:
            self.notify("No job selected", severity="warning")
            return
        if not job.url:
            self.notify(
                f"No URL stored for {job.source}:{job.external_id}",
                severity="warning",
            )
            return
        self.copy_to_clipboard(job.url)
        self.notify(f"Copied {job.url}")

    def action_fetch_description(self) -> None:
        """Fetch a missing description without blocking the UI.

        The lookup and the eventual write both happen here, on the thread that
        owns the connection; only the HTTP call runs off-thread, since SQLite
        objects may not be used from another thread.
        """
        job = self.selected_job()
        if job is None:
            return
        if job.description:
            self.notify("Already have the description")
            return
        if not job.description_url:
            self.notify(
                f"Nothing to fetch: no detail link for {job.source}",
                severity="warning",
            )
            return

        self.notify("Fetching description…")
        self.fetch_description(
            job.id, job.title, job.company, job.description_url, job.source
        )

    @work(exclusive=True, thread=True)
    def fetch_description(
        self, job_id: int, title: str, company: str, url: str, source: str
    ) -> None:
        """Worker: the network call only, with no database access.

        Everything needed was read on the owning thread and passed in, so this
        touches no SQLite object. ``source`` selects the payload parser. The
        result is handed back for the main thread to store -- see
        :func:`_store_fetched_description`.

        Imported inside the worker so the TUI does not require the vendor
        modules (or httpx) unless a fetch is actually requested.
        """
        from ...ingest.sources.description import (
            fetch_description_from_known_url,
        )

        outcome = fetch_description_from_known_url(
            job_id, title, company, url, source=source
        )
        self.call_from_thread(self._store_fetched_description, outcome)

    def _store_fetched_description(self, outcome) -> None:
        """Write a fetch result, on the main thread.

        A short write, so doing it here does not perceptibly block the UI, and it
        keeps every connection use on one thread.
        """
        self._count_cache = None
        if outcome.error:
            self.notify(outcome.error, severity="error")
        elif outcome.fetched is not None:
            from ...ingest.sources.description import store_description

            if store_description(self.conn, outcome.fetched):
                self.notify("Description stored")
            else:
                self.notify("Already had the description")
        self.refresh_view(keep_cursor=True, reset_page=False)

    # -- embedding search --------------------------------------------------

    def action_search_similar(self) -> None:
        """Embed the search box contents and rank by similarity.

        Only called on explicit submit (ctrl+e while the search box is focused),
        never per keystroke: embedding costs a model call.
        """
        text = self.query_one("#search-input", Input).value.strip()
        if not text:
            self.notify("Type a query first", severity="warning")
            return
        if self._embed_query is None:
            self.notify("No embedder configured", severity="error")
            return

        vector = self.embed_cached(text)
        if vector is None:
            return

        found = q.similar_jobs(
            self.conn, vector, scope="filtered", limit=self.page_size
        )
        # Cache the results before switching tabs: switching fires
        # TabActivated, whose refresh would otherwise read an empty Similar tab
        # and discard these.
        self._similar_to = None
        self._similar_rows = [Row(hit.job, hit.score) for hit in found]
        self._similar_for = text
        self._active_tab = "similar"
        self.query_one(TabbedContent).active = "similar"
        self.refresh_view()

    def embed_cached(self, text: str) -> list[float] | None:
        """Embed a query, memoised for the session.

        One model call per distinct query; re-submitting the same text is free,
        which matters since re-running after a filter change is the common case.
        """
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        try:
            vector = list(self._embed_query(text))  # type: ignore[misc]
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            self.notify(f"Embedding failed: {error}", severity="error")
            return None
        self._query_cache[text] = vector
        return vector


def build_embedder() -> Callable[[str], Sequence[float]] | None:
    """The real query embedder, or ``None`` when the model is unavailable.

    Imported lazily so the TUI starts without torch; a missing embedder degrades
    embedding search to a notification rather than a crash on startup.
    """
    try:
        from ...core.embedder import embed_query
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    return embed_query


def run(
    db_path: str | None = None,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    embed_query: Callable[[str], Sequence[float]] | None = None,
) -> int:
    """Open the database and run the TUI. Returns a process exit code."""
    conn = core_db.connect(db_path)
    try:
        app = JobApp(
            conn,
            page_size=page_size,
            embed_query=embed_query if embed_query is not None else build_embedder(),
        )
        app.run()
    finally:
        conn.close()
    return 0
