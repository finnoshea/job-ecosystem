"""Tests for jobecosystem.ingest.runner: orchestrating scrapers and run rows."""

from __future__ import annotations

import pytest

from jobecosystem.core.models import Job
from jobecosystem.ingest import runner
from jobecosystem.ingest.base import FetchError, ParseError, Scraper


# ---------------------------------------------------------------------------
# scraper stubs
# ---------------------------------------------------------------------------

def make_job(external_id="1", source="ok", description=None, **overrides):
    fields = {
        "source": source,
        "external_id": external_id,
        "company": "Acme",
        "title": f"Job {external_id}",
        "description": description,
    }
    fields.update(overrides)
    return Job(**fields)


class OkScraper(Scraper):
    source = "ok"

    def __init__(self, count=2, descriptions=True):
        self.count = count
        self.descriptions = descriptions

    def fetch(self):
        return [
            make_job(str(i), description=f"desc {i}" if self.descriptions else None)
            for i in range(1, self.count + 1)
        ]


class FailingScraper(Scraper):
    def __init__(self, source, error):
        self.source = source
        self.error = error

    def fetch(self):
        raise self.error


class ErroringScraper(Scraper):
    """Returns jobs but also records per-listing errors."""

    source = "erroring"

    def fetch(self):
        return [make_job("1", source="erroring")]


class EmptyScraper(Scraper):
    source = "empty"

    def fetch(self):
        return []


def embed_constant(texts, model_name="test-model"):
    """Fake embedder: one 3-dim vector per text."""
    return [b"\x00" * 12 for _ in texts]


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------

def test_run_returns_one_outcome_per_scraper(conn):
    summary = runner.run_scrapers(conn, [OkScraper(), EmptyScraper()])
    assert [o.source for o in summary] == ["ok", "empty"]
    assert len(summary) == 2


def test_run_inserts_jobs_and_tallies(conn):
    summary = runner.run_scrapers(conn, [OkScraper(count=3)])
    outcome = summary.outcomes[0]
    assert outcome.fetched == 3
    assert outcome.inserted == 3
    assert outcome.ok
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 3


def test_run_marks_unchanged_reruns(conn):
    runner.run_scrapers(conn, [OkScraper()])
    summary = runner.run_scrapers(conn, [OkScraper()])
    outcome = summary.outcomes[0]
    assert outcome.inserted == 0
    assert outcome.updated == 0
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_run_counts_updates(conn):
    runner.run_scrapers(conn, [OkScraper(count=1)])
    class Renamed(OkScraper):
        def fetch(self):
            return [make_job("1", description="desc 1", title="Renamed")]

    summary = runner.run_scrapers(conn, [Renamed()])
    assert summary.outcomes[0].updated == 1


def test_ok_summary_when_every_source_succeeds(conn):
    summary = runner.run_scrapers(conn, [OkScraper(), EmptyScraper()])
    assert summary.ok
    assert summary.failed_sources == []


def test_empty_source_fetches_nothing_but_is_ok(conn):
    summary = runner.run_scrapers(conn, [EmptyScraper()])
    outcome = summary.outcomes[0]
    assert outcome.fetched == 0
    assert outcome.ok
    assert outcome.run_id is not None


def test_sources_run_in_the_order_given(conn):
    a = OkScraper()
    b = FailingScraper("z", FetchError("down"))
    summary = runner.run_scrapers(conn, [a, b])
    assert [o.source for o in summary] == ["ok", "z"]


def test_on_outcome_is_called_per_source_in_order(conn):
    seen = []
    runner.run_scrapers(
        conn,
        [OkScraper(), FailingScraper("z", FetchError("down")), EmptyScraper()],
        on_outcome=seen.append,
    )
    assert [o.source for o in seen] == ["ok", "z", "empty"]
    # Failures are reported too, not just successes.
    assert seen[1].error is not None


def test_on_outcome_receives_the_same_outcomes_as_the_summary(conn):
    seen = []
    summary = runner.run_scrapers(
        conn, [OkScraper(), EmptyScraper()], on_outcome=seen.append
    )
    assert seen == list(summary.outcomes)


# ---------------------------------------------------------------------------
# failure handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", [FetchError("refused"), ParseError("bad json")])
def test_scraper_error_is_recorded_not_raised(conn, error):
    summary = runner.run_scrapers(conn, [FailingScraper("down", error)])
    outcome = summary.outcomes[0]
    assert outcome.ok is False
    assert "refused" in outcome.error or "bad json" in outcome.error
    assert outcome.run_id is not None


def test_unexpected_scraper_error_is_recorded_with_traceback(conn):
    # A parser bug must not take down the run, but the traceback must survive.
    summary = runner.run_scrapers(conn, [FailingScraper("boom", ZeroDivisionError("bug"))])
    outcome = summary.outcomes[0]
    assert outcome.ok is False
    assert "ZeroDivisionError" in outcome.error
    assert "Traceback" in (outcome.exc_text or "")


def test_a_failing_scraper_does_not_stop_the_others(conn):
    summary = runner.run_scrapers(conn, [
        FailingScraper("first", FetchError("down")),
        OkScraper(),
        FailingScraper("last", FetchError("down")),
    ])
    assert summary.failed_sources == ["first", "last"]
    # The middle source still wrote its jobs.
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert len(summary) == 3


def test_failed_run_is_recorded_even_with_nothing_fetched(conn):
    # A dead source must be visible rather than looking like a quiet board.
    runner.run_scrapers(conn, [FailingScraper("down", FetchError("refused"))])
    row = conn.execute("SELECT * FROM scrape_runs").fetchone()
    assert (row["source"], row["fetched"], row["errors"]) == ("down", 0, 1)
    assert row["finished_at"] is not None


def test_failure_message_lands_in_scrape_errors(conn):
    runner.run_scrapers(conn, [FailingScraper("down", FetchError("connection refused"))])
    row = conn.execute("SELECT tenant, error FROM scrape_errors").fetchone()
    assert row["tenant"] is None
    assert "connection refused" in row["error"]


def test_broken_source_attribute_still_records_a_run(conn):
    class NoSource(Scraper):
        def fetch(self):
            return []

    # source_name raises NotImplementedError; bookkeeping must still happen.
    summary = runner.run_scrapers(conn, [NoSource()])
    assert summary.outcomes[0].source == "NoSource"
    assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 1


def test_keyboard_interrupt_is_not_swallowed(conn):
    class Interrupted(Scraper):
        source = "interrupted"

        def fetch(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        runner.run_scrapers(conn, [Interrupted()])


# ---------------------------------------------------------------------------
# run rows
# ---------------------------------------------------------------------------

def test_every_source_gets_its_own_run_row(conn):
    runner.run_scrapers(conn, [OkScraper(), EmptyScraper()])
    sources = [r[0] for r in conn.execute("SELECT source FROM scrape_runs ORDER BY id")]
    assert sources == ["ok", "empty"]


def test_run_row_matches_the_upsert_tally(conn):
    summary = runner.run_scrapers(conn, [OkScraper(count=3)])
    row = conn.execute("SELECT * FROM scrape_runs").fetchone()
    assert row["fetched"] == summary.outcomes[0].fetched
    assert row["inserted"] == summary.outcomes[0].inserted


def test_run_id_is_returned_on_the_outcome(conn):
    summary = runner.run_scrapers(conn, [OkScraper()])
    run_id = conn.execute("SELECT id FROM scrape_runs").fetchone()[0]
    assert summary.outcomes[0].run_id == run_id


def test_jobs_are_committed_before_the_run_row(conn):
    # If jobs were left uncommitted, a later rollback could orphan the tallies.
    runner.run_scrapers(conn, [OkScraper()])
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------

def test_no_embedder_means_no_embedding(conn):
    summary = runner.run_scrapers(conn, [OkScraper()])
    assert summary.outcomes[0].embeddings_written == 0
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 0


def test_embedder_writes_vectors_for_each_job(conn):
    summary = runner.run_scrapers(conn, [OkScraper(count=2)], embed=embed_constant)
    assert summary.outcomes[0].embeddings_written == 2
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 2


def test_jobs_without_descriptions_are_not_embedded(conn):
    # An empty document would match every other empty document.
    summary = runner.run_scrapers(
        conn, [OkScraper(count=2, descriptions=False)], embed=embed_constant
    )
    assert summary.outcomes[0].embeddings_written == 0
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 0


def test_embedding_uses_the_stored_description(conn):
    seen = []

    def recording_embed(texts, model_name="test-model"):
        seen.extend(texts)
        return [b"\x00" * 12 for _ in texts]

    runner.run_scrapers(conn, [OkScraper(count=2)], embed=recording_embed)
    assert sorted(seen) == ["desc 1", "desc 2"]


def test_embedding_failure_does_not_fail_the_scrape(conn):
    def broken_embed(texts):
        raise RuntimeError("model file missing")

    summary = runner.run_scrapers(conn, [OkScraper(count=2)], embed=broken_embed)
    outcome = summary.outcomes[0]
    assert outcome.ok                      # jobs still landed
    assert outcome.fetched == 2
    assert outcome.embedding_failures == 2
    assert any("model file missing" in message for _, message in outcome.errors)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_embedder_returning_wrong_count_is_reported(conn):
    def short_embed(texts):
        return [b"\x00" * 12]

    summary = runner.run_scrapers(conn, [OkScraper(count=3)], embed=short_embed)
    outcome = summary.outcomes[0]
    assert outcome.embedding_failures == 3
    assert any("vectors for" in message for _, message in outcome.errors)
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 0


def test_embedding_model_column_falls_back_to_unknown(conn):
    runner.run_scrapers(conn, [OkScraper(count=1)], embed=embed_constant)
    assert conn.execute("SELECT model FROM job_embeddings").fetchone()[0] == "unknown"


def test_embedding_model_column_uses_model_name_when_present(conn):
    def named_embed(texts):
        return [b"\x00" * 12 for _ in texts]

    # _model_name reads DEFAULT_MODEL off the embedder object.
    named_embed.model_name = "nomic-ai/nomic-embed-text-v1.5"
    runner.run_scrapers(conn, [OkScraper(count=1)], embed=named_embed)
    assert conn.execute("SELECT model FROM job_embeddings").fetchone()[0] == \
        "nomic-ai/nomic-embed-text-v1.5"


def test_dim_is_derived_from_the_vector_length(conn):
    def embed_768(texts):
        return [b"\x00" * (768 * 4) for _ in texts]

    runner.run_scrapers(conn, [OkScraper(count=1)], embed=embed_768)
    assert conn.execute("SELECT dim FROM job_embeddings").fetchone()[0] == 768


def test_embedding_failure_does_not_roll_back_jobs(conn):
    # The commit before embedding is what makes this hold.
    def broken_embed(texts):
        raise RuntimeError("boom")

    runner.run_scrapers(conn, [OkScraper(count=2)], embed=broken_embed)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_embeddings_are_only_written_for_the_current_batch(conn):
    runner.run_scrapers(conn, [OkScraper(count=2)], embed=embed_constant)
    before = conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0]

    # A rerun with identical content rewrites the same two rows, not four.
    runner.run_scrapers(conn, [OkScraper(count=2)], embed=embed_constant)
    after = conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0]
    assert before == after == 2


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_run_produces_a_usable_database(conn):
    summary = runner.run_scrapers(
        conn,
        [OkScraper(count=3), FailingScraper("down", FetchError("refused")), EmptyScraper()],
        embed=embed_constant,
    )
    assert summary.ok is False
    assert summary.failed_sources == ["down"]

    # Jobs, embeddings, run rows, and errors all consistent.
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM job_embeddings").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM scrape_errors").fetchone()[0] == 1

    # And the views built on top of them work.
    assert conn.execute("SELECT COUNT(*) FROM jobs_unseen").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM jobs_with_embeddings").fetchone()[0] == 3


def test_run_records_reposts_on_a_later_pass(conn):
    # A repost needs: the row unseen for >= REPOST_GAP_DAYS, and a table anchor
    # that is not this run. The anchor is derived from the freshest last_seen_at,
    # so job 1 is set to "yesterday" rather than to now.
    runner.run_scrapers(conn, [OkScraper(count=2)])
    conn.execute(
        "UPDATE jobs SET last_seen_at = datetime('now','-1 day') WHERE external_id = '1'"
    )
    conn.execute(
        "UPDATE jobs SET last_seen_at = datetime('now','-30 days') WHERE external_id = '2'"
    )
    conn.commit()

    summary = runner.run_scrapers(conn, [OkScraper(count=2)])
    outcome = summary.outcomes[0]
    assert outcome.reposted == 1
    counts = conn.execute(
        "SELECT external_id, repost_count FROM jobs ORDER BY external_id"
    ).fetchall()
    assert [(r[0], r[1]) for r in counts] == [("1", 0), ("2", 1)]


def test_run_scrapers_with_no_scrapers(conn):
    summary = runner.run_scrapers(conn, [])
    assert len(summary) == 0
    assert summary.ok
    assert conn.execute("SELECT COUNT(*) FROM scrape_runs").fetchone()[0] == 0


def test_run_scrapers_accepts_a_generator(conn):
    summary = runner.run_scrapers(conn, (OkScraper() for _ in range(2)))
    assert len(summary) == 2


def test_outcome_len_is_fetched_count(conn):
    summary = runner.run_scrapers(conn, [OkScraper(count=4)])
    assert len(summary.outcomes[0]) == 4
