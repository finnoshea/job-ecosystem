"""Tests for jobecosystem.ingest.sources.workday_description.

The getter is deliberately standalone, so these tests exercise it both directly
and through the batch path the cron job would use. No sleeping happens: the
batch takes an injected ``sleep``.
"""

from __future__ import annotations

import sqlite3

import pytest

from jobecosystem.core.models import Job
from jobecosystem.ingest import upsert
from jobecosystem.ingest.sources import workday_description as wd


def store_job(conn, external_id="A", **overrides):
    """Insert a Workday-shaped row: no description, but a URL to fetch it from."""
    fields = {
        "source": "workday:asml:ASMLPrivate1",
        "external_id": external_id,
        "company": "ASML",
        "title": "Software Engineer",
        "url": f"https://asml.wd3.myworkdayjobs.com/ASMLPrivate1/job/X/S_{external_id}",
        "description_url": (
            f"https://asml.wd3.myworkdayjobs.com/wday/cxs/asml/ASMLPrivate1/job/X/S_{external_id}"
        ),
    }
    fields.update(overrides)
    job = Job(**fields)
    result = upsert.upsert_job(conn, job)
    conn.commit()
    return result.job_id


def detail_payload(description="<p>A real description.</p>"):
    return {"jobPostingInfo": {"title": "Software Engineer", "jobDescription": description}}


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def test_detail_url_joins_the_parts():
    assert wd.detail_url("nvidia.wd5", "nvidia", "Site", "/job/X/Y_1") == (
        "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/Site/job/X/Y_1"
    )


def test_detail_url_adds_a_missing_leading_slash():
    assert wd.detail_url("h.wd1", "t", "s", "job/X") == (
        "https://h.wd1.myworkdayjobs.com/wday/cxs/t/s/job/X"
    )


# ---------------------------------------------------------------------------
# HTML conversion
# ---------------------------------------------------------------------------

def test_html_paragraphs_become_newlines():
    assert wd.html_to_text("<p>One</p><p>Two</p>") == "One\nTwo"


def test_html_list_items_become_lines():
    assert wd.html_to_text("<ul><li>A</li><li>B</li></ul>") == "A\nB"


def test_br_becomes_a_newline():
    assert wd.html_to_text("One<br>Two") == "One\nTwo"


def test_self_closing_br_is_handled():
    assert wd.html_to_text("One<br/>Two") == "One\nTwo"


def test_entities_are_unescaped():
    assert wd.html_to_text("<p>R&amp;D &lt;team&gt;</p>") == "R&D <team>"


def test_non_breaking_spaces_become_regular_spaces():
    assert wd.html_to_text("<p>Hello&nbsp;world</p>") == "Hello world"


def test_script_and_style_content_is_dropped():
    text = wd.html_to_text("<script>evil()</script><p>Keep</p><style>a{}</style>")
    assert text == "Keep"
    assert "evil" not in text


def test_inline_tags_do_not_break_lines():
    assert wd.html_to_text("<p>Hello <strong>bold</strong> text</p>") == "Hello bold text"


def test_indentation_is_collapsed():
    assert wd.html_to_text("<p>   lots     of space   </p>") == "lots of space"


def test_newlines_inside_a_block_are_preserved():
    # Only horizontal whitespace is collapsed; a line break in the source is
    # kept, since some vendors put meaningful line breaks inside a paragraph.
    assert wd.html_to_text("<p>lots of\nspace</p>") == "lots of\nspace"


def test_excess_blank_lines_are_collapsed():
    assert "\n\n\n" not in wd.html_to_text("<p>A</p><p></p><p></p><p>B</p>")


def test_headers_become_their_own_lines():
    assert wd.html_to_text("<h1>Title</h1><p>Body</p>") == "Title\nBody"


def test_empty_markup_yields_empty_text():
    assert wd.html_to_text("<p></p><div>  </div>") == ""


def test_crlf_is_normalized():
    assert "\r" not in wd.html_to_text("<p>A\r\nB</p>")


# ---------------------------------------------------------------------------
# parse_description
# ---------------------------------------------------------------------------

def test_parse_description_extracts_plain_text():
    assert wd.parse_description(detail_payload()) == "A real description."


def test_parse_description_rejects_a_non_object():
    with pytest.raises(wd.DescriptionError, match="expected an object"):
        wd.parse_description(["nope"])


def test_parse_description_rejects_a_missing_info_block():
    with pytest.raises(wd.DescriptionError, match="jobPostingInfo"):
        wd.parse_description({"somethingElse": {}})


def test_parse_description_rejects_an_empty_description():
    with pytest.raises(wd.DescriptionError, match="missing or empty"):
        wd.parse_description(detail_payload(""))


def test_parse_description_rejects_markup_with_no_text():
    # Present but useless: better a loud failure than an empty row.
    with pytest.raises(wd.DescriptionError, match="no text"):
        wd.parse_description(detail_payload("<p>   </p><div></div>"))


def test_parse_description_rejects_a_non_string_description():
    with pytest.raises(wd.DescriptionError, match="missing or empty"):
        wd.parse_description(detail_payload(None))


# ---------------------------------------------------------------------------
# fetch_description: the happy path
# ---------------------------------------------------------------------------

def test_fetch_description_writes_everything(conn):
    job_id = store_job(conn)
    result = wd.fetch_description(conn, job_id, fetch_json=lambda url: detail_payload())

    assert result.ok and result.written and result.fetched
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row["description"] == "A real description."
    assert row["description_fetched_at"] is not None
    assert row["content_hash"] is not None
    assert len(row["content_hash"]) == 64


def test_content_hash_matches_the_shared_helper(conn):
    from jobecosystem.core.models import content_hash

    job_id = store_job(conn, title="Engineer")
    wd.fetch_description(conn, job_id, fetch_json=lambda url: detail_payload("Body"))
    stored = conn.execute("SELECT content_hash FROM jobs").fetchone()[0]
    assert stored == content_hash("Engineer", "ASML", "Body")


def test_fetched_job_leaves_the_work_queue(conn):
    job_id = store_job(conn)
    assert conn.execute("SELECT COUNT(*) FROM jobs_needing_descriptions").fetchone()[0] == 1
    wd.fetch_description(conn, job_id, fetch_json=lambda url: detail_payload())
    assert conn.execute("SELECT COUNT(*) FROM jobs_needing_descriptions").fetchone()[0] == 0


def test_fetched_job_can_be_embedded(conn):
    # End to end: the description is what makes embedding and similarity possible.
    from jobecosystem.core import similarity

    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda url: detail_payload())
    row = conn.execute("SELECT vector FROM jobs_with_embeddings").fetchone()
    assert row is None or similarity.decode(row["vector"]) is not None


def test_the_request_url_is_the_stored_one(conn):
    job_id = store_job(conn)
    seen = []
    wd.fetch_description(
        conn, job_id, fetch_json=lambda url: seen.append(url) or detail_payload()
    )
    assert seen == [conn.execute(
        "SELECT description_url FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()[0]]


# ---------------------------------------------------------------------------
# fetch_description: only once
# ---------------------------------------------------------------------------

def test_a_second_call_does_not_refetch(conn):
    job_id = store_job(conn)
    calls = []

    def fetch_json(url):
        calls.append(url)
        return detail_payload()

    wd.fetch_description(conn, job_id, fetch_json=fetch_json)
    second = wd.fetch_description(conn, job_id, fetch_json=fetch_json)

    assert len(calls) == 1        # the network was touched once
    assert second.skipped and not second.written
    assert second.ok              # skipping is normal, not an error


def test_force_refetches(conn):
    job_id = store_job(conn)
    calls = []

    def fetch_json(url):
        calls.append(url)
        return detail_payload(f"<p>Version {len(calls)}</p>")

    wd.fetch_description(conn, job_id, fetch_json=fetch_json)
    wd.fetch_description(conn, job_id, force=True, fetch_json=fetch_json)

    assert len(calls) == 2
    assert conn.execute("SELECT description FROM jobs").fetchone()[0] == "Version 2"


def test_force_updates_the_hash_when_text_changes(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload("<p>First</p>"))
    before = conn.execute("SELECT content_hash FROM jobs").fetchone()[0]
    wd.fetch_description(
        conn, job_id, force=True, fetch_json=lambda u: detail_payload("<p>Second</p>")
    )
    assert conn.execute("SELECT content_hash FROM jobs").fetchone()[0] != before


def test_a_race_loses_gracefully(conn):
    # The conditional UPDATE is what makes "only once" hold when a TUI request
    # and a cron batch target the same row.
    job_id = store_job(conn)
    other = sqlite3.connect(conn.execute("PRAGMA database_list").fetchone()[2])
    other.row_factory = sqlite3.Row
    other.execute("PRAGMA foreign_keys=ON")
    try:
        # Simulate the other writer finishing first.
        other.execute(
            "UPDATE jobs SET description = 'theirs', content_hash = 'x',"
            " description_fetched_at = 'now' WHERE id = ?",
            (job_id,),
        )
        other.commit()
        result = wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload())
        assert result.skipped is True
        assert conn.execute("SELECT description FROM jobs").fetchone()[0] == "theirs"
    finally:
        other.close()


def test_existing_description_is_never_clobbered(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload("<p>Keep me</p>"))
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload("<p>Other</p>"))
    assert conn.execute("SELECT description FROM jobs").fetchone()[0] == "Keep me"


# ---------------------------------------------------------------------------
# fetch_description: failures
# ---------------------------------------------------------------------------

def test_missing_job_is_reported_not_raised(conn):
    result = wd.fetch_description(conn, 9999)
    assert result.ok is False
    assert "no job" in result.error


def test_a_row_without_a_description_url_is_reported(conn):
    job = Job(source="ashby", external_id="X", company="Acme", title="Eng",
              description=None)
    upsert.upsert_job(conn, job)
    conn.commit()
    result = wd.fetch_description(conn, job.id or 1)
    # An Ashby row has no description_url; only Workday needs one.
    assert result.ok is False
    assert "description_url" in result.error


def test_http_failure_is_reported_not_raised(conn):
    job_id = store_job(conn)

    def boom(url):
        raise wd.DescriptionError("HTTP 503")

    result = wd.fetch_description(conn, job_id, fetch_json=boom)
    assert result.ok is False
    assert "503" in result.error
    assert conn.execute("SELECT description FROM jobs").fetchone()[0] is None


def test_unexpected_exception_is_reported_not_raised(conn):
    def boom(url):
        raise RuntimeError("kaboom")

    result = wd.fetch_description(conn, store_job(conn), fetch_json=boom)
    assert result.ok is False
    assert "kaboom" in result.error


def test_a_failed_fetch_leaves_the_job_in_the_queue(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: {"nope": True})
    assert conn.execute("SELECT COUNT(*) FROM jobs_needing_descriptions").fetchone()[0] == 1


def test_a_failed_fetch_does_not_set_description_fetched_at(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: {"nope": True})
    assert conn.execute(
        "SELECT description_fetched_at FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()[0] is None


def test_a_bad_payload_error_mentions_the_response(conn):
    result = wd.fetch_description(conn, store_job(conn), fetch_json=lambda u: {"x": 1})
    assert "jobPostingInfo" in result.error


# ---------------------------------------------------------------------------
# fetch_descriptions (batch)
# ---------------------------------------------------------------------------

def test_batch_fills_the_queue(conn):
    for i in range(3):
        store_job(conn, external_id=f"J{i}")
    summary = wd.fetch_descriptions(conn, sleep=lambda s: None, fetch_json=lambda u: detail_payload())
    assert (summary.attempted, summary.written, summary.skipped) == (3, 3, 0)
    assert summary.ok
    assert conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE content_hash IS NOT NULL"
    ).fetchone()[0] == 3


def test_batch_respects_the_limit(conn):
    for i in range(5):
        store_job(conn, external_id=f"J{i}")
    summary = wd.fetch_descriptions(
        conn, limit=2, sleep=lambda s: None, fetch_json=lambda u: detail_payload()
    )
    assert summary.attempted == 2
    assert conn.execute("SELECT COUNT(*) FROM jobs_needing_descriptions").fetchone()[0] == 3


def test_batch_resumes_where_it_left_off(conn):
    for i in range(4):
        store_job(conn, external_id=f"J{i}")
    wd.fetch_descriptions(conn, limit=2, sleep=lambda s: None, fetch_json=lambda u: detail_payload())
    second = wd.fetch_descriptions(
        conn, limit=2, sleep=lambda s: None, fetch_json=lambda u: detail_payload()
    )
    assert second.written == 2
    assert conn.execute("SELECT COUNT(*) FROM jobs_needing_descriptions").fetchone()[0] == 0


def test_batch_sleeps_between_requests_but_not_before_the_first(conn):
    for i in range(3):
        store_job(conn, external_id=f"J{i}")
    sleeps = []
    wd.fetch_descriptions(
        conn, delay=1.0, jitter=0.0, sleep=sleeps.append,
        fetch_json=lambda u: detail_payload(),
    )
    # One pause between each pair, none before the first request.
    assert sleeps == [1.0, 1.0]


def test_batch_jitter_stays_within_bounds(conn):
    for i in range(4):
        store_job(conn, external_id=f"J{i}")
    sleeps = []
    wd.fetch_descriptions(
        conn, delay=0.5, jitter=0.25, sleep=sleeps.append,
        fetch_json=lambda u: detail_payload(),
    )
    assert all(0.5 <= s <= 0.75 for s in sleeps)


def test_batch_records_failures_without_stopping(conn):
    for i in range(3):
        store_job(conn, external_id=f"J{i}")

    def flaky(url):
        if "_J1" in url:
            raise wd.DescriptionError("HTTP 500")
        return detail_payload()

    summary = wd.fetch_descriptions(conn, sleep=lambda s: None, fetch_json=flaky)
    assert summary.written == 2
    assert len(summary.failed) == 1
    assert summary.ok is False


def test_batch_marks_already_fetched_rows_as_skipped(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload())
    # Empty the queue check by passing the id explicitly.
    summary = wd.fetch_descriptions(
        conn, job_ids=[job_id], sleep=lambda s: None, fetch_json=lambda u: detail_payload()
    )
    assert (summary.attempted, summary.written, summary.skipped) == (1, 0, 1)


def test_batch_with_an_empty_queue_does_nothing(conn):
    summary = wd.fetch_descriptions(conn, sleep=lambda s: None)
    assert (summary.attempted, summary.written) == (0, 0)
    assert summary.ok
    assert len(summary) == 0


def test_batch_job_ids_are_truncated_to_limit(conn):
    ids = [store_job(conn, external_id=f"J{i}") for i in range(4)]
    summary = wd.fetch_descriptions(
        conn, job_ids=ids, limit=2, sleep=lambda s: None,
        fetch_json=lambda u: detail_payload(),
    )
    assert summary.attempted == 2


def test_batch_orders_the_queue_newest_first(conn):
    ids = [store_job(conn, external_id=f"J{i}") for i in range(3)]
    summary = wd.fetch_descriptions(
        conn, sleep=lambda s: None, fetch_json=lambda u: detail_payload()
    )
    assert summary.written == len(ids)


# ---------------------------------------------------------------------------
# integration: scrape then fetch
# ---------------------------------------------------------------------------

def test_scrape_then_fetch_end_to_end(conn):
    from jobecosystem.ingest import runner
    from jobecosystem.ingest.sources.workday import WorkdayScraper

    listing = {
        "title": "Software Engineer",
        "externalPath": "/job/X/Software-Engineer_J-1",
        "locationsText": "Veldhoven, Netherlands",
        "bulletFields": ["J-1"],
    }
    page = {"total": 1, "jobPostings": [listing]}
    scraper = WorkdayScraper(
        "asml.wd3", "asml", "ASMLPrivate1", company="ASML",
        post_json=lambda url, body: page,
    )
    runner.run_scrapers(conn, [scraper])

    row = conn.execute("SELECT id, content_hash FROM jobs").fetchone()
    assert row["content_hash"] is None

    summary = wd.fetch_descriptions(
        conn, sleep=lambda s: None, fetch_json=lambda u: detail_payload()
    )
    assert summary.written == 1

    row = conn.execute(
        "SELECT description, content_hash, description_fetched_at FROM jobs"
    ).fetchone()
    assert row["description"] == "A real description."
    assert row["content_hash"] is not None
    assert row["description_fetched_at"] is not None


def test_scraped_rows_then_fetched_do_not_report_false_reposts(conn):
    from jobecosystem.ingest import runner
    from jobecosystem.ingest.sources.workday import WorkdayScraper

    def page_for(external_path):
        return {"total": 2, "jobPostings": [
            {"title": "Software Engineer", "externalPath": external_path,
             "locationsText": "X", "bulletFields": ["J-1"]},
            {"title": "Software Engineer", "externalPath": "/job/X/Other_J-2",
             "locationsText": "X", "bulletFields": ["J-2"]},
        ]}

    scraper = WorkdayScraper(
        "asml.wd3", "asml", "Site", company="ASML",
        post_json=lambda url, body: page_for("/job/X/SWE_J-1"),
    )
    runner.run_scrapers(conn, [scraper])
    wd.fetch_descriptions(
        conn, sleep=lambda s: None, fetch_json=lambda u: detail_payload("<p>Same text</p>")
    )

    # Identical text under two ids is a genuine duplicate-content signal.
    assert conn.execute("SELECT COUNT(*) FROM jobs_reposted").fetchone()[0] == 2


# ---------------------------------------------------------------------------
# the fetch/store split
# ---------------------------------------------------------------------------

def test_known_url_fetch_touches_no_database(conn):
    # The whole point of the split: this must be callable from a thread that
    # does not own the connection, so it takes no connection at all.
    outcome = wd.fetch_description_from_known_url(
        42, "Engineer", "Acme", "https://x.test/j",
        fetch_json=lambda url: detail_payload("<p>Body</p>"),
    )
    assert outcome.ok
    assert outcome.fetched is not None
    assert outcome.fetched.text == "Body"
    assert outcome.fetched.job_id == 42


def test_known_url_fetch_computes_the_hash():
    from jobecosystem.core.models import content_hash

    outcome = wd.fetch_description_from_known_url(
        1, "Engineer", "Acme", "https://x.test/j",
        fetch_json=lambda url: detail_payload("<p>Body</p>"),
    )
    assert outcome.fetched.content_hash == content_hash("Engineer", "Acme", "Body")


def test_known_url_fetch_reports_transport_errors(conn):
    def boom(url):
        raise wd.DescriptionError("HTTP 503")

    outcome = wd.fetch_description_from_known_url(
        1, "Engineer", "Acme", "https://x.test/j", fetch_json=boom
    )
    assert outcome.fetched is None
    assert outcome.ok is False
    assert "503" in outcome.error


def test_from_url_reads_but_does_not_write(conn):
    job_id = store_job(conn)
    outcome = wd.fetch_description_from_url(
        conn, job_id, fetch_json=lambda u: detail_payload("<p>Body</p>")
    )
    assert outcome.fetched is not None
    # Nothing written yet.
    row = conn.execute("SELECT description, content_hash FROM jobs").fetchone()
    assert row["description"] is None
    assert row["content_hash"] is None


def test_store_description_writes_the_row(conn):
    job_id = store_job(conn)
    outcome = wd.fetch_description_from_url(
        conn, job_id, fetch_json=lambda u: detail_payload("<p>Body</p>")
    )
    assert wd.store_description(conn, outcome.fetched) is True
    row = conn.execute(
        "SELECT description, content_hash, description_fetched_at FROM jobs"
    ).fetchone()
    assert row["description"] == "Body"
    assert row["content_hash"] is not None
    assert row["description_fetched_at"] is not None


def test_store_description_loses_the_race_gracefully(conn):
    job_id = store_job(conn)
    outcome = wd.fetch_description_from_url(
        conn, job_id, fetch_json=lambda u: detail_payload("<p>Body</p>")
    )
    assert wd.store_description(conn, outcome.fetched) is True
    # A second store of the same fetch must not overwrite.
    assert wd.store_description(conn, outcome.fetched) is False


def test_store_description_force_overwrites(conn):
    job_id = store_job(conn)
    outcome = wd.fetch_description_from_url(
        conn, job_id, fetch_json=lambda u: detail_payload("<p>First</p>")
    )
    wd.store_description(conn, outcome.fetched)
    second = wd.fetch_description_from_url(
        conn, job_id, force=True, fetch_json=lambda u: detail_payload("<p>Second</p>")
    )
    assert wd.store_description(conn, second.fetched, force=True) is True
    assert conn.execute("SELECT description FROM jobs").fetchone()[0] == "Second"


def test_the_split_reproduces_fetch_description(conn):
    # Composing the halves must behave the same as the one-shot call.
    a = store_job(conn, external_id="A")
    combined = wd.fetch_description(
        conn, a, fetch_json=lambda u: detail_payload("<p>Via combined</p>")
    )

    b = store_job(conn, external_id="B")
    outcome = wd.fetch_description_from_url(
        conn, b, fetch_json=lambda u: detail_payload("<p>Via split</p>")
    )
    written = wd.store_description(conn, outcome.fetched)

    assert combined.written is True and combined.fetched is True
    assert written is True
    rows = dict(conn.execute("SELECT external_id, description FROM jobs"))
    assert rows["A"] == "Via combined"
    assert rows["B"] == "Via split"


def test_from_url_reports_a_missing_job(conn):
    outcome = wd.fetch_description_from_url(conn, 999)
    assert outcome.error is not None
    assert "no job" in outcome.error


def test_from_url_skips_an_already_described_job(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload())
    outcome = wd.fetch_description_from_url(conn, job_id)
    assert outcome.skipped is True
    assert outcome.fetched is None


def test_from_url_reports_a_missing_description_url(conn):
    job = Job(source="ashby", external_id="X", company="Acme", title="Eng",
              description=None)
    upsert.upsert_job(conn, job)
    conn.commit()
    outcome = wd.fetch_description_from_url(conn, job.id or 1)
    assert outcome.error is not None
    assert "description_url" in outcome.error


def test_fetch_outcome_ok_reflects_the_error():
    assert wd.FetchOutcome(job_id=1).ok is True
    assert wd.FetchOutcome(job_id=1, skipped=True).ok is True
    assert wd.FetchOutcome(job_id=1, error="boom").ok is False


def test_fetched_description_is_a_plain_carrier():
    fetched = wd.FetchedDescription(job_id=1, text="body", content_hash="h")
    assert (fetched.job_id, fetched.text, fetched.content_hash) == (1, "body", "h")


# ---------------------------------------------------------------------------
# retry cap
# ---------------------------------------------------------------------------

def attempts(conn, job_id):
    return conn.execute(
        "SELECT description_attempts FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()[0]


def test_a_successful_fetch_counts_one_attempt(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload())
    assert attempts(conn, job_id) == 1


def test_a_failed_fetch_counts_an_attempt(conn):
    job_id = store_job(conn)

    def boom(url):
        raise wd.DescriptionError("HTTP 403")

    result = wd.fetch_description(conn, job_id, fetch_json=boom)
    assert result.error is not None
    assert attempts(conn, job_id) == 1


def test_the_attempt_is_counted_before_the_request(conn):
    # The stub runs during the request, so it sees the counter already bumped.
    job_id = store_job(conn)
    seen = {}

    def fetch(url):
        seen["attempts"] = attempts(conn, job_id)
        return detail_payload("<p>Body</p>")

    wd.fetch_description(conn, job_id, fetch_json=fetch)
    assert seen["attempts"] == 1


def test_a_skipped_fetch_does_not_count_an_attempt(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload())
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload())
    assert attempts(conn, job_id) == 1


def test_force_counts_another_attempt(conn):
    job_id = store_job(conn)
    wd.fetch_description(conn, job_id, fetch_json=lambda u: detail_payload("<p>A</p>"))
    wd.fetch_description(
        conn, job_id, force=True, fetch_json=lambda u: detail_payload("<p>B</p>")
    )
    assert attempts(conn, job_id) == 2


def test_a_missing_description_url_still_counts_an_attempt(conn):
    # Otherwise the row would sit in the queue forever, never able to request.
    job = Job(source="ashby", external_id="X", company="Acme", title="Eng",
              description=None)
    upsert.upsert_job(conn, job)
    conn.commit()
    job_id = job.id or 1
    wd.fetch_description(conn, job_id)
    assert attempts(conn, job_id) == 1


def test_a_row_at_the_cap_is_not_attempted(conn):
    job_id = store_job(conn)
    conn.execute("UPDATE jobs SET description_attempts = 5 WHERE id = ?", (job_id,))
    conn.commit()
    summary = wd.fetch_descriptions(
        conn, sleep=lambda s: None, fetch_json=lambda u: detail_payload()
    )
    assert summary.attempted == 0
    assert conn.execute("SELECT description FROM jobs").fetchone()[0] is None


def test_force_bypasses_the_cap(conn):
    job_id = store_job(conn)
    conn.execute("UPDATE jobs SET description_attempts = 5 WHERE id = ?", (job_id,))
    conn.commit()
    summary = wd.fetch_descriptions(
        conn, force=True, sleep=lambda s: None,
        fetch_json=lambda u: detail_payload(),
    )
    assert summary.written == 1
    assert attempts(conn, job_id) == 6


def test_a_rescrape_does_not_reset_the_attempts(conn):
    job_id = store_job(conn)
    conn.execute("UPDATE jobs SET description_attempts = 3 WHERE id = ?", (job_id,))
    conn.commit()
    store_job(conn)                  # upsert the same (source, external_id) again
    assert attempts(conn, job_id) == 3
