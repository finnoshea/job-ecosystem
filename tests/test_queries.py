"""Tests for jobecosystem.triage.queries.

No model is required: the embedding tests inject a deterministic fake embedder,
and the "more like this" path reuses stored vectors.
"""

from __future__ import annotations

import pytest

from jobecosystem.core import similarity
from jobecosystem.core.models import Job
from jobecosystem.ingest import upsert
from jobecosystem.triage import queries as q


def add_job(
    conn,
    external_id,
    *,
    source="ashby:ramp",
    company="Acme",
    title="Engineer",
    location=None,
    description="unremarkable body text",
    status="new",
    rating=None,
    salary_min=None,
    salary_max=None,
):
    job = Job(
        source=source,
        external_id=external_id,
        company=company,
        title=title,
        location=location,
        description=description,
        salary_min=salary_min,
        salary_max=salary_max,
    )
    result = upsert.upsert_job(conn, job)
    if status != "new" or rating is not None:
        conn.execute(
            "UPDATE jobs SET status = ?, rating = ? WHERE id = ?",
            (status, rating, result.job_id),
        )
    conn.commit()
    return result.job_id


def embed_vector(conn, job_id, vector, *, model="test"):
    payload = similarity.encode(vector)
    with conn:
        conn.execute(
            "INSERT INTO job_embeddings (job_id, vector, dim, model, updated_at)"
            " VALUES (?, ?, ?, ?, '2026-01-01T00:00:00Z')"
            " ON CONFLICT (job_id) DO UPDATE SET vector = excluded.vector,"
            " dim = excluded.dim, model = excluded.model",
            (job_id, payload, len(vector), model),
        )
    return payload


def fake_embedder(mapping):
    """Returns an embed function that maps text to a preset vector."""

    def embed(text):
        for key, vector in mapping.items():
            if key in text.lower():
                return vector
        return [0.0] * len(next(iter(mapping.values())))

    return embed


# ---------------------------------------------------------------------------
# JobFilter: validation
# ---------------------------------------------------------------------------

def test_filter_defaults_are_empty():
    f = q.JobFilter()
    assert f.where() == ("", [])
    assert f.matches(Job(source="a", external_id="1", company="c", title="t"))


def test_filter_rejects_a_rating_out_of_range():
    with pytest.raises(q.QueryError, match="0-5"):
        q.JobFilter(rated_at_least=6)
    with pytest.raises(q.QueryError, match="0-5"):
        q.JobFilter(rated_at_least=-1)


def test_filter_allows_rated_at_least_zero():
    # 0 means "has any rating", not "unrated".
    assert q.JobFilter(rated_at_least=0).rated_at_least == 0


def test_filter_rejects_an_unknown_status():
    with pytest.raises(q.QueryError, match="unknown status"):
        q.JobFilter(statuses=("bogus",))


def test_filter_accepts_every_valid_status():
    assert q.JobFilter(statuses=q.VALID_STATUSES).statuses == q.VALID_STATUSES


# ---------------------------------------------------------------------------
# JobFilter: where() and matches() agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filters, job_kwargs, expected",
    [
        (q.JobFilter(sources=("ashby:ramp",)), {"source": "ashby:ramp"}, True),
        (q.JobFilter(sources=("ashby:ramp",)), {"source": "workday:x:S"}, False),
        (q.JobFilter(source_prefixes=("ashby",)), {"source": "ashby:zapier"}, True),
        (q.JobFilter(source_prefixes=("ashby",)), {"source": "workday:x:S"}, False),
        (q.JobFilter(companies=("Acme",)), {"company": "Acme"}, True),
        (q.JobFilter(companies=("Acme",)), {"company": "Beta"}, False),
        (q.JobFilter(statuses=("applied",)), {"status": "applied"}, True),
        (q.JobFilter(statuses=("applied",)), {"status": "new"}, False),
        (q.JobFilter(rated_at_least=3), {"rating": 4}, True),
        (q.JobFilter(rated_at_least=3), {"rating": 2}, False),
        (q.JobFilter(rated_at_least=3), {"rating": None}, False),
        (q.JobFilter(rated_at_least=0), {"rating": 0}, True),
        (q.JobFilter(min_salary=100000), {"salary_min": 100000}, True),
        (q.JobFilter(min_salary=100000), {"salary_min": 90000}, False),
        (q.JobFilter(min_salary=100000), {"salary_max": 120000}, True),
        (q.JobFilter(min_salary=100000), {"salary_min": None, "salary_max": None}, False),
    ],
)
def test_matches_semantics(conn, filters, job_kwargs, expected):
    defaults = {"source": "a", "external_id": "1", "company": "c", "title": "t",
                "description": "d"}
    defaults.update(job_kwargs)
    job = Job(**defaults)
    assert filters.matches(job) is expected


def test_where_and_matches_agree_on_many_rows(conn):
    # The two implementations must not drift: one builds SQL, one works in
    # Python, and both define what a filter means. Compared against a plain
    # table scan, since a view adds its own WHERE clause that would also
    # exclude rows.
    for i, (source, company, status) in enumerate([
        ("ashby:ramp", "Acme", "new"),
        ("workday:x:S", "Beta", "applied"),
        ("ashby:zapier", "Acme", "new"),
    ], start=1):
        add_job(conn, str(i), source=source, company=company, status=status)
    conn.execute("UPDATE jobs SET rating = 5 WHERE external_id = '1'")
    conn.execute("UPDATE jobs SET salary_min = 200000 WHERE external_id = '2'")
    conn.execute("UPDATE jobs SET description = NULL WHERE external_id = '3'")
    conn.commit()

    all_jobs = [
        Job.from_row(row) for row in conn.execute("SELECT * FROM jobs ORDER BY id")
    ]

    for filters in (
        q.JobFilter(),
        q.JobFilter(sources=("ashby:ramp",)),
        q.JobFilter(source_prefixes=("ashby",)),
        q.JobFilter(companies=("Acme",)),
        q.JobFilter(statuses=("applied",)),
        q.JobFilter(rated_at_least=1),
        q.JobFilter(min_salary=100000),
        q.JobFilter(only_with_descriptions=True),
        q.JobFilter(source_prefixes=("ashby",), companies=("Acme",)),
        q.JobFilter(statuses=("new",), rated_at_least=0),
    ):
        where_sql, params = filters.where(alias="j")
        sql = "SELECT j.* FROM jobs j"
        if where_sql:
            sql += " WHERE " + where_sql
        sql_rows = {row["external_id"] for row in conn.execute(sql, params)}
        python_rows = {job.external_id for job in all_jobs if filters.matches(job)}
        assert sql_rows == python_rows, filters


# ---------------------------------------------------------------------------
# view readers
# ---------------------------------------------------------------------------

def test_jobs_unseen_excludes_triaged_rows(conn):
    add_job(conn, "1", status="new")
    add_job(conn, "2", status="applied")
    add_job(conn, "3", status="hidden")
    assert sorted(j.external_id for j in q.jobs_unseen(conn)) == ["1"]


def test_jobs_today_returns_recent(conn):
    add_job(conn, "1")
    assert [j.external_id for j in q.jobs_today(conn)] == ["1"]


def test_jobs_needing_descriptions_is_the_queue(conn):
    add_job(conn, "1", description="has one")
    add_job(conn, "2", description=None)
    assert [j.external_id for j in q.jobs_needing_descriptions(conn)] == ["2"]


def test_jobs_needing_descriptions_caps_attempts(conn):
    add_job(conn, "1", description=None)
    exhausted = add_job(conn, "2", description=None)
    conn.execute("UPDATE jobs SET description_attempts = 5 WHERE id = ?", (exhausted,))
    conn.commit()
    assert [j.external_id for j in q.jobs_needing_descriptions(conn)] == ["1"]


def test_readers_accept_a_limit(conn):
    for i in range(5):
        add_job(conn, str(i))
    assert len(q.jobs_unseen(conn, limit=2)) == 2


def test_a_limit_of_zero_returns_nothing(conn):
    add_job(conn, "1")
    assert q.jobs_unseen(conn, limit=0) == []


def test_readers_apply_filters(conn):
    add_job(conn, "1", company="Acme")
    add_job(conn, "2", company="Beta")
    result = q.jobs_unseen(conn, filters=q.JobFilter(companies=("Beta",)))
    assert [j.external_id for j in result] == ["2"]


def test_filter_and_limit_combine(conn):
    for i in range(6):
        add_job(conn, str(i), company="Acme" if i % 2 == 0 else "Beta")
    result = q.jobs_unseen(
        conn, filters=q.JobFilter(companies=("Acme",)), limit=2
    )
    assert len(result) == 2
    assert all(j.company == "Acme" for j in result)


def test_unknown_view_is_rejected(conn):
    with pytest.raises(q.QueryError, match="unknown view"):
        q._read_view(conn, "not_a_view", filters=None, limit=None)


def test_all_named_views_exist(conn):
    # A typo in _VIEWS should fail here, not at runtime.
    existing = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'view'"
        )
    }
    assert set(q._VIEWS) <= existing


# ---------------------------------------------------------------------------
# get_job / get_jobs
# ---------------------------------------------------------------------------

def test_get_job_returns_none_for_an_unknown_id(conn):
    assert q.get_job(conn, 999) is None


def test_get_job_returns_the_job(conn):
    job_id = add_job(conn, "1", title="Specific Title")
    assert q.get_job(conn, job_id).title == "Specific Title"


def test_get_jobs_preserves_order(conn):
    first = add_job(conn, "1")
    second = add_job(conn, "2")
    assert [j.id for j in q.get_jobs(conn, [second, first])] == [second, first]


def test_get_jobs_skips_unknown_ids(conn):
    job_id = add_job(conn, "1")
    assert [j.id for j in q.get_jobs(conn, [job_id, 999])] == [job_id]


def test_get_jobs_of_nothing(conn):
    assert q.get_jobs(conn, []) == []


# ---------------------------------------------------------------------------
# keyword search
# ---------------------------------------------------------------------------

def test_search_matches_the_title(conn):
    add_job(conn, "1", title="Senior Python Engineer", description="backend work")
    add_job(conn, "2", title="Rust Engineer", description="systems work")
    assert [j.external_id for j in q.search_jobs(conn, "python")] == ["1"]


def test_search_matches_the_description(conn):
    add_job(conn, "1", title="Engineer", description="expert in kubernetes")
    add_job(conn, "2", title="Engineer", description="expert in rust")
    assert [j.external_id for j in q.search_jobs(conn, "kubernetes")] == ["1"]


def test_search_is_case_insensitive(conn):
    add_job(conn, "1", title="Senior PYTHON Engineer", description="backend work")
    assert [j.external_id for j in q.search_jobs(conn, "python")] == ["1"]
    assert [j.external_id for j in q.search_jobs(conn, "PYTHON")] == ["1"]


def test_search_requires_every_term(conn):
    add_job(conn, "1", title="Python Engineer", description="remote friendly")
    add_job(conn, "2", title="Python Engineer", description="on site only")
    assert [j.external_id for j in q.search_jobs(conn, "python remote")] == ["1"]


def test_search_terms_may_span_columns(conn):
    add_job(conn, "1", title="Python Engineer", description="fully remote")
    assert [j.external_id for j in q.search_jobs(conn, "python remote")] == ["1"]


def test_search_matches_the_company(conn):
    add_job(conn, "1", company="Nvidia", title="Engineer")
    add_job(conn, "2", company="Acme", title="Engineer")
    assert [j.external_id for j in q.search_jobs(conn, "nvidia")] == ["1"]


def test_search_matches_the_location(conn):
    add_job(conn, "1", location="Remote - US", title="Engineer")
    add_job(conn, "2", location="London, UK", title="Engineer")
    assert [j.external_id for j in q.search_jobs(conn, "london")] == ["2"]


def test_search_terms_may_span_title_and_company(conn):
    add_job(conn, "1", company="Nvidia", title="Python Engineer")
    add_job(conn, "2", company="Acme", title="Python Engineer")
    assert [j.external_id for j in q.search_jobs(conn, "python nvidia")] == ["1"]


def test_count_jobs_uses_the_same_search_predicate(conn):
    # The page indicator has to agree with the results, so make sure the two
    # share the company/location columns, not just title/description.
    add_job(conn, "1", company="Nvidia", title="Engineer")
    add_job(conn, "2", company="Acme", title="Engineer")
    assert q.count_jobs(conn, text="nvidia") == 1
    assert len(q.search_jobs(conn, "nvidia")) == 1


def test_blank_search_returns_nothing(conn):
    add_job(conn, "1", title="Python")
    assert q.search_jobs(conn, "   ") == []
    assert q.search_jobs(conn, "") == []


def test_search_respects_filters(conn):
    add_job(conn, "1", title="Python", source="ashby:ramp")
    add_job(conn, "2", title="Python", source="workday:x:S")
    result = q.search_jobs(
        conn, "python", filters=q.JobFilter(source_prefixes=("workday",))
    )
    assert [j.external_id for j in result] == ["2"]


def test_search_handles_a_null_description(conn):
    add_job(conn, "1", title="Python Engineer", description=None)
    assert [j.external_id for j in q.search_jobs(conn, "python")] == ["1"]


def test_search_finds_nothing_gracefully(conn):
    add_job(conn, "1", title="Python")
    assert q.search_jobs(conn, "cobol") == []


def test_search_respects_the_limit(conn):
    for i in range(5):
        add_job(conn, str(i), title="Python Engineer")
    assert len(q.search_jobs(conn, "python", limit=2)) == 2


def test_search_with_sql_metacharacters_is_safe(conn):
    # A user typing % or a quote must not break the query. The DROP attempt must
    # not match anything: the input is a parameter, never interpolated.
    add_job(conn, "1", title="Python Engineer", description="plain body")
    assert q.search_jobs(conn, "'") == []
    assert q.search_jobs(conn, "'; DROP TABLE jobs; --") == []
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# embedding search
# ---------------------------------------------------------------------------

VECTORS = {
    "python": [1.0, 0.0, 0.0],
    "rust": [0.0, 1.0, 0.0],
    "nurse": [0.0, 0.0, 1.0],
}


@pytest.fixture
def embedded(conn):
    """Three jobs with orthogonal vectors, keyed by language."""
    ids = {}
    for key, vector in VECTORS.items():
        job_id = add_job(conn, key, title=f"{key.title()} Job")
        embed_vector(conn, job_id, vector)
        ids[key] = job_id
    return ids


def test_similar_jobs_ranks_by_cosine(conn, embedded):
    results = q.similar_jobs(
        conn, [1.0, 0.0, 0.0], limit=3, embed=None
    )
    assert [r.job.external_id for r in results][0] == "python"
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_similar_jobs_returns_scores_in_descending_order(conn, embedded):
    results = q.similar_jobs(conn, [1.0, 1.0, 0.0], limit=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_similar_jobs_respects_the_limit(conn, embedded):
    assert len(q.similar_jobs(conn, [1.0, 0.0, 0.0], limit=2)) == 2


def test_similar_jobs_filters_by_min_score(conn, embedded):
    results = q.similar_jobs(conn, [1.0, 0.0, 0.0], min_score=0.5)
    assert [r.job.external_id for r in results] == ["python"]


def test_similar_jobs_with_no_embeddings_returns_nothing(conn):
    add_job(conn, "1")
    assert q.similar_jobs(conn, [1.0, 0.0]) == []


def test_similar_jobs_excludes_the_given_job(conn, embedded):
    results = q.similar_jobs(
        conn, [1.0, 0.0, 0.0], exclude_job_id=embedded["python"], min_score=-1
    )
    assert "python" not in [r.job.external_id for r in results]


def test_scope_all_ignores_filters(conn, embedded):
    results = q.similar_jobs(
        conn, [0.0, 1.0, 0.0],
        filters=q.JobFilter(source_prefixes=("workday",)),
        scope="all", limit=3, min_score=-1,
    )
    # The python job is on ashby:ramp and still appears.
    assert "python" in [r.job.external_id for r in results]


def test_scope_filtered_respects_filters(conn):
    ashby = add_job(conn, "a", source="ashby:ramp")
    workday = add_job(conn, "w", source="workday:x:S")
    embed_vector(conn, ashby, [1.0, 0.0])
    embed_vector(conn, workday, [0.0, 1.0])

    results = q.similar_jobs(
        conn, [1.0, 0.0],
        filters=q.JobFilter(source_prefixes=("workday",)),
        scope="filtered", min_score=-1,
    )
    assert [r.job.external_id for r in results] == ["w"]


def test_invalid_scope_is_rejected(conn, embedded):
    with pytest.raises(q.QueryError, match="scope"):
        q.similar_jobs(conn, [1.0, 0.0, 0.0], scope="everything")


def test_text_query_uses_the_injected_embedder(conn, embedded):
    calls = []

    def embed(text):
        calls.append(text)
        return VECTORS["rust"]

    results = q.similar_jobs(conn, "systems programming", limit=1, embed=embed)
    assert calls == ["systems programming"]
    assert results[0].job.external_id == "rust"


def test_text_query_without_an_embedder_raises(conn, embedded):
    with pytest.raises(q.QueryError, match="needs an embedder"):
        q.similar_jobs(conn, "python")


def test_blank_text_query_returns_nothing_without_embedding(conn, embedded):
    calls = []
    assert q.similar_jobs(conn, "   ", embed=lambda t: calls.append(t)) == []
    assert calls == []


def test_similar_to_job_reuses_the_stored_vector(conn, embedded):
    # No embedder is passed, so a model call would raise.
    results = q.similar_to_job(conn, embedded["python"], min_score=-1)
    assert results[0].job.external_id == "rust" or results[0].score < 1.0
    assert "python" not in [r.job.external_id for r in results]


def test_similar_to_job_without_an_embedding_returns_nothing(conn):
    job_id = add_job(conn, "1")
    assert q.similar_to_job(conn, job_id) == []


def test_similar_to_job_never_returns_itself(conn, embedded):
    for key, job_id in embedded.items():
        results = q.similar_to_job(conn, job_id, min_score=-1)
        assert key not in [r.job.external_id for r in results]


def test_similarity_handles_a_dimension_change_gracefully(conn):
    # Vectors of differing widths must not silently produce a wrong score.
    a = add_job(conn, "a")
    b = add_job(conn, "b")
    embed_vector(conn, a, [1.0, 0.0, 0.0])
    embed_vector(conn, b, [1.0, 0.0])
    with pytest.raises(ValueError):
        q.similar_jobs(conn, [1.0, 0.0, 0.0], min_score=-1)


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------

def test_set_status_updates_the_row(conn):
    job_id = add_job(conn, "1")
    assert q.set_status(conn, job_id, "applied") is True
    assert q.get_job(conn, job_id).status == "applied"


def test_set_status_returns_false_for_an_unknown_id(conn):
    assert q.set_status(conn, 999, "seen") is False


def test_set_status_rejects_an_invalid_value(conn):
    job_id = add_job(conn, "1")
    with pytest.raises(q.QueryError, match="invalid status"):
        q.set_status(conn, job_id, "bogus")


def test_set_status_removes_a_job_from_unseen(conn):
    job_id = add_job(conn, "1")
    q.set_status(conn, job_id, "seen")
    assert q.jobs_unseen(conn) == []


def test_set_rating_stores_a_value(conn):
    job_id = add_job(conn, "1")
    assert q.set_rating(conn, job_id, 4) is True
    assert q.get_job(conn, job_id).rating == 4


def test_set_rating_clears_with_none(conn):
    job_id = add_job(conn, "1", rating=4)
    q.set_rating(conn, job_id, None)
    assert q.get_job(conn, job_id).rating is None


def test_set_rating_accepts_the_ends_of_the_range(conn):
    job_id = add_job(conn, "1")
    assert q.set_rating(conn, job_id, 0) is True
    assert q.set_rating(conn, job_id, 5) is True


def test_set_rating_rejects_out_of_range(conn):
    job_id = add_job(conn, "1")
    with pytest.raises(q.QueryError, match="0-5"):
        q.set_rating(conn, job_id, 6)
    with pytest.raises(q.QueryError, match="0-5"):
        q.set_rating(conn, job_id, -1)


def test_set_rating_returns_false_for_an_unknown_id(conn):
    assert q.set_rating(conn, 999, 3) is False


def test_set_status_many_updates_several(conn):
    ids = [add_job(conn, str(i)) for i in range(3)]
    assert q.set_status_many(conn, ids, "hidden") == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'hidden'"
    ).fetchone()[0] == 3


def test_set_status_many_with_no_ids(conn):
    assert q.set_status_many(conn, [], "seen") == 0


def test_set_status_many_rejects_an_invalid_status(conn):
    ids = [add_job(conn, "1")]
    with pytest.raises(q.QueryError, match="invalid status"):
        q.set_status_many(conn, ids, "bogus")


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def test_counts_by_source(conn):
    add_job(conn, "1", source="ashby:ramp")
    add_job(conn, "2", source="ashby:ramp")
    add_job(conn, "3", source="workday:x:S")
    assert q.counts_by_source(conn) == {"ashby:ramp": 2, "workday:x:S": 1}


def test_counts_by_source_is_empty_without_jobs(conn):
    assert q.counts_by_source(conn) == {}


def test_counts_by_status(conn):
    add_job(conn, "1", status="new")
    add_job(conn, "2", status="applied")
    assert q.counts_by_status(conn) == {"new": 1, "applied": 1}


def test_companies_are_sorted_and_distinct(conn):
    add_job(conn, "1", company="Beta")
    add_job(conn, "2", company="Acme")
    add_job(conn, "3", company="Acme")
    assert q.companies(conn) == ["Acme", "Beta"]


def test_companies_respects_a_limit(conn):
    for name in ("C", "A", "B"):
        add_job(conn, name, company=name)
    assert q.companies(conn, limit=2) == ["A", "B"]


def test_sources_are_sorted(conn):
    add_job(conn, "1", source="workday:x:S")
    add_job(conn, "2", source="ashby:ramp")
    assert q.sources(conn) == ["ashby:ramp", "workday:x:S"]


def test_unresolved_count(conn):
    add_job(conn, "1", description="has one")
    add_job(conn, "2", description=None)
    assert q.unresolved_count(conn) == 1


def test_description_progress_shape(conn):
    add_job(conn, "1", description="body")
    progress = q.description_progress(conn)
    assert progress["total_jobs"] == 1
    assert progress["described"] == 1


def test_description_progress_on_an_empty_database(conn):
    assert q.description_progress(conn)["total_jobs"] == 0


def test_run_summary_is_empty_without_runs(conn):
    assert q.run_summary(conn) == []


def test_run_summary_returns_recent_runs_newest_first(conn):
    conn.execute(
        "INSERT INTO scrape_runs (source, started_at, finished_at, fetched)"
        " VALUES ('a', '2026-01-01T00:00:00Z', '2026-01-01T00:01:00Z', 1)"
    )
    conn.execute(
        "INSERT INTO scrape_runs (source, started_at, finished_at, fetched)"
        " VALUES ('b', '2026-06-01T00:00:00Z', '2026-06-01T00:01:00Z', 2)"
    )
    conn.commit()
    rows = q.run_summary(conn)
    assert [r["source"] for r in rows] == ["b", "a"]
    assert rows[0]["status"] == "finished"


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------

def test_triage_workflow(conn):
    """Add, search, triage, and re-query -- the loop the TUI drives."""
    add_job(conn, "1", title="Senior Python Engineer", description="python sqlite")
    add_job(conn, "2", title="Rust Engineer", description="rust systems")
    add_job(conn, "3", title="Python Data Analyst", description="python sql dashboards")

    matches = q.search_jobs(conn, "python")
    assert sorted(j.external_id for j in matches) == ["1", "3"]

    # Search returns newest first, so pick rows by id rather than by position.
    by_external = {job.external_id: job.id for job in matches}
    q.set_rating(conn, by_external["1"], 5)
    q.set_status(conn, by_external["1"], "applied")
    q.set_status(conn, by_external["3"], "seen")
    # The non-matching job is still untriaged.
    q.set_status(conn, 2, "seen")

    assert q.jobs_unseen(conn) == []
    assert q.counts_by_status(conn) == {"applied": 1, "seen": 2}
    assert q.counts_by_source(conn) == {"ashby:ramp": 3}

    rated = q.jobs_unseen(conn, filters=q.JobFilter(rated_at_least=5))
    assert rated == []          # the rated job is applied, so not in unseen
    applied = q.jobs_today(conn, filters=q.JobFilter(statuses=("applied",)))
    assert [j.external_id for j in applied] == ["1"]


# ---------------------------------------------------------------------------
# jobs_by_status / jobs_all
# ---------------------------------------------------------------------------

def test_jobs_by_status_filters(conn):
    add_job(conn, "1", status="applied")
    add_job(conn, "2", status="seen")
    add_job(conn, "3", status="applied")
    assert sorted(j.external_id for j in q.jobs_by_status(conn, "applied")) == ["1", "3"]


def test_jobs_by_status_rejects_an_unknown_status(conn):
    with pytest.raises(q.QueryError, match="invalid status"):
        q.jobs_by_status(conn, "bogus")


def test_jobs_by_status_respects_a_limit(conn):
    for i in range(4):
        add_job(conn, str(i), status="applied")
    assert len(q.jobs_by_status(conn, "applied", limit=2)) == 2


def test_jobs_by_status_with_a_zero_limit(conn):
    add_job(conn, "1", status="applied")
    assert q.jobs_by_status(conn, "applied", limit=0) == []


def test_jobs_by_status_is_empty_when_none_match(conn):
    add_job(conn, "1", status="new")
    assert q.jobs_by_status(conn, "applied") == []


def test_jobs_by_status_combines_with_a_filter(conn):
    add_job(conn, "1", status="applied", source="ashby:ramp")
    add_job(conn, "2", status="applied", source="workday:x:S")
    result = q.jobs_by_status(
        conn, "applied", filters=q.JobFilter(source_prefixes=("ashby",))
    )
    assert [j.external_id for j in result] == ["1"]


def test_jobs_by_status_ignores_a_contradictory_status_filter(conn):
    # The status comes from the argument; a filter carrying its own statuses
    # would contradict it, so it is dropped rather than AND-ed in.
    add_job(conn, "1", status="applied")
    result = q.jobs_by_status(
        conn, "applied", filters=q.JobFilter(statuses=("hidden",))
    )
    assert [j.external_id for j in result] == ["1"]


def test_jobs_all_returns_every_status(conn):
    add_job(conn, "1", status="new")
    add_job(conn, "2", status="seen")
    add_job(conn, "3", status="applied")
    add_job(conn, "4", status="hidden")
    assert sorted(j.external_id for j in q.jobs_all(conn)) == ["1", "2", "3", "4"]


def test_jobs_all_respects_a_limit(conn):
    for i in range(5):
        add_job(conn, str(i))
    assert len(q.jobs_all(conn, limit=3)) == 3


def test_jobs_all_with_a_zero_limit(conn):
    add_job(conn, "1")
    assert q.jobs_all(conn, limit=0) == []


def test_jobs_all_is_empty_without_rows(conn):
    assert q.jobs_all(conn) == []


def test_jobs_all_applies_filters(conn):
    add_job(conn, "1", company="Acme")
    add_job(conn, "2", company="Beta")
    result = q.jobs_all(conn, filters=q.JobFilter(companies=("Beta",)))
    assert [j.external_id for j in result] == ["2"]


def test_jobs_all_orders_newest_seen_first(conn):
    add_job(conn, "1")
    add_job(conn, "2")
    conn.execute("UPDATE jobs SET last_seen_at = '2099-01-01T00:00:00Z' WHERE external_id='1'")
    conn.commit()
    assert [j.external_id for j in q.jobs_all(conn)][0] == "1"


# ---------------------------------------------------------------------------
# stale_job_ids / repost_info
# ---------------------------------------------------------------------------

def test_stale_job_ids_matches_the_view(conn):
    add_job(conn, "fresh")
    add_job(conn, "old")
    conn.execute("UPDATE jobs SET last_seen_at = '2026-01-01T00:00:00Z' WHERE external_id = 'old'")
    conn.execute("UPDATE jobs SET last_seen_at = '2099-01-01T00:00:00Z' WHERE external_id = 'fresh'")
    conn.commit()

    ids = q.stale_job_ids(conn)
    from_view = {row[0] for row in conn.execute("SELECT id FROM jobs_stale")}
    assert ids == from_view
    old_id = conn.execute("SELECT id FROM jobs WHERE external_id='old'").fetchone()[0]
    assert old_id in ids


def test_stale_job_ids_is_empty_without_stale_rows(conn):
    add_job(conn, "1")
    assert q.stale_job_ids(conn) == set()


def test_repost_info_reports_duplicate_content(conn):
    # The case the row cannot answer: same text, two ids, never re-seen.
    add_job(conn, "A", title="Senior Engineer", description="identical body")
    add_job(conn, "B", title="Senior Engineer", description="identical body")

    info = q.repost_info(conn)
    assert len(info) == 2
    assert set(info.values()) == {1}
    # And the rows themselves say nothing about it.
    assert conn.execute("SELECT SUM(repost_count) FROM jobs").fetchone()[0] == 0


def test_repost_info_is_empty_when_nothing_is_reposted(conn):
    add_job(conn, "1", title="One", description="first body")
    add_job(conn, "2", title="Two", description="second body")
    assert q.repost_info(conn) == {}


def test_repost_info_includes_reseen_rows(conn):
    add_job(conn, "1", title="Engineer", description="body")
    conn.execute("UPDATE jobs SET repost_count = 3, content_hash = NULL")
    conn.commit()
    # With no hash the view excludes it, which is the documented limitation.
    assert q.repost_info(conn) == {}


def test_repost_info_agrees_with_the_view(conn):
    add_job(conn, "A", title="Same", description="same body")
    add_job(conn, "B", title="Same", description="same body")
    add_job(conn, "C", title="Different", description="other body")
    from_view = {
        row["id"]
        for row in conn.execute("SELECT id FROM jobs_reposted")
    }
    assert set(q.repost_info(conn)) == from_view


# ---------------------------------------------------------------------------
# paging
# ---------------------------------------------------------------------------

def test_offset_walks_jobs_all_without_gaps_or_repeats(conn):
    for i in range(10):
        add_job(conn, f"{i:02d}")
    seen = []
    for offset in (0, 4, 8):
        seen += [j.external_id for j in q.jobs_all(conn, limit=4, offset=offset)]
    assert len(seen) == 10
    assert len(set(seen)) == 10


def test_offset_past_the_end_is_empty(conn):
    add_job(conn, "1")
    assert q.jobs_all(conn, limit=10, offset=99) == []


def test_offset_without_a_limit_is_not_ignored(conn):
    # SQLite ignores OFFSET unless a LIMIT is present, which would make paging
    # silently repeat the first page.
    for i in range(5):
        add_job(conn, str(i))
    assert len(q.jobs_all(conn, offset=3)) == 2


def test_offset_applies_to_a_view_reader(conn):
    for i in range(6):
        add_job(conn, str(i))
    page1 = [j.external_id for j in q.jobs_unseen(conn, limit=2, offset=0)]
    page2 = [j.external_id for j in q.jobs_unseen(conn, limit=2, offset=2)]
    assert len(page1) == len(page2) == 2
    assert not set(page1) & set(page2)


def test_offset_applies_to_jobs_by_status(conn):
    for i in range(4):
        job_id = add_job(conn, str(i))
        q.set_status(conn, job_id, "applied")
    page1 = q.jobs_by_status(conn, "applied", limit=2, offset=0)
    page2 = q.jobs_by_status(conn, "applied", limit=2, offset=2)
    assert len(page1) == len(page2) == 2
    assert not {j.id for j in page1} & {j.id for j in page2}


def test_offset_applies_to_search(conn):
    for i in range(6):
        add_job(conn, f"{i:02d}", title="Python Engineer", description="python")
    page1 = q.search_jobs(conn, "python", limit=2, offset=0)
    page2 = q.search_jobs(conn, "python", limit=2, offset=2)
    assert len(page1) == len(page2) == 2
    assert not {j.id for j in page1} & {j.id for j in page2}


def test_filtered_paging_does_not_skip_rows(conn):
    # The filter runs in Python after the SQL, so the offset has to be applied
    # after filtering too -- otherwise each page boundary loses a row.
    for i in range(10):
        add_job(conn, f"{i:02d}", company="Acme" if i % 2 == 0 else "Beta")
    filters = q.JobFilter(companies=("Acme",))
    seen = []
    for offset in (0, 2, 4):
        seen += [
            j.external_id
            for j in q.jobs_all(conn, filters=filters, limit=2, offset=offset)
        ]
    assert len(seen) == 5
    assert len(set(seen)) == 5


# ---------------------------------------------------------------------------
# count_jobs
# ---------------------------------------------------------------------------

def test_count_jobs_counts_everything(conn):
    for i in range(7):
        add_job(conn, str(i))
    assert q.count_jobs(conn) == 7


def test_count_jobs_counts_a_view(conn):
    add_job(conn, "1")
    add_job(conn, "2", status="applied")
    assert q.count_jobs(conn, view="jobs_unseen") == 1


def test_count_jobs_counts_a_status(conn):
    ids = [add_job(conn, str(i)) for i in range(3)]
    q.set_status(conn, ids[0], "applied")
    assert q.count_jobs(conn, status="applied") == 1


def test_count_jobs_applies_a_text_predicate(conn):
    add_job(conn, "1", title="Python Engineer", description="python")
    add_job(conn, "2", title="Rust Engineer", description="rust")
    assert q.count_jobs(conn, text="python") == 1


def test_count_jobs_agrees_with_search_length(conn):
    # A count that disagrees with its reader is worse than no count.
    for i in range(5):
        add_job(conn, f"{i:02d}", title="Python Engineer", description="python")
    add_job(conn, "x", title="Rust Engineer", description="rust")
    assert q.count_jobs(conn, text="python") == len(
        q.search_jobs(conn, "python", limit=1000)
    )


def test_count_jobs_applies_filters(conn):
    add_job(conn, "1", company="Acme")
    add_job(conn, "2", company="Beta")
    assert q.count_jobs(conn, filters=q.JobFilter(companies=("Beta",))) == 1


def test_count_jobs_filters_within_a_view(conn):
    add_job(conn, "1", company="Acme")
    add_job(conn, "2", company="Beta")
    assert (
        q.count_jobs(conn, view="jobs_unseen", filters=q.JobFilter(companies=("Beta",)))
        == 1
    )


def test_count_jobs_matches_view_length(conn):
    add_job(conn, "1", company="Acme")
    add_job(conn, "2", company="Beta")
    assert q.count_jobs(conn, view="jobs_unseen") == len(q.jobs_unseen(conn))


def test_count_jobs_rejects_an_unknown_view(conn):
    with pytest.raises(q.QueryError, match="unknown view"):
        q.count_jobs(conn, view="nope")


def test_count_jobs_rejects_an_unknown_status(conn):
    with pytest.raises(q.QueryError, match="invalid status"):
        q.count_jobs(conn, status="bogus")


def test_count_jobs_rejects_view_and_status_together(conn):
    with pytest.raises(q.QueryError, match="not both"):
        q.count_jobs(conn, view="jobs_unseen", status="new")


def test_count_jobs_of_an_empty_database(conn):
    assert q.count_jobs(conn) == 0


# ---------------------------------------------------------------------------
# counting views cheaply
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "view", ["jobs_today", "jobs_unseen", "jobs_reposted", "jobs_stale"]
)
def test_count_jobs_matches_the_view_length(conn, view):
    # The fast-path counts restate each view's predicate by hand, so they must
    # be checked against the view itself or a divergence would go unnoticed.
    for i in range(6):
        add_job(conn, str(i), title=f"Job {i}", description=f"body {i}")
    assert q.count_jobs(conn, view=view) == len(
        q._read_view(conn, view, filters=None, limit=None)
    )


def test_count_jobs_of_today_matches_after_a_status_change(conn):
    ids = [add_job(conn, str(i)) for i in range(3)]
    q.set_status(conn, ids[0], "hidden")
    assert q.count_jobs(conn, view="jobs_today") == len(
        q._read_view(conn, "jobs_today", filters=None, limit=None)
    )


def test_count_jobs_of_stale_excludes_applied(conn):
    old_id = add_job(conn, "old")
    # A second, recently seen row, so the scrape anchor is not the stale row
    # itself -- a lone row can never be 14 days older than its own last_seen_at.
    add_job(conn, "fresh")
    conn.execute("UPDATE jobs SET last_seen_at = '2020-01-01T00:00:00Z' WHERE id = ?", (old_id,))
    conn.execute("UPDATE jobs SET last_seen_at = '2099-01-01T00:00:00Z' WHERE id != ?", (old_id,))
    conn.commit()
    assert q.count_jobs(conn, view="jobs_stale") == 1
    q.set_status(conn, old_id, "applied")
    assert q.count_jobs(conn, view="jobs_stale") == 0


def test_count_jobs_falls_back_when_a_filter_is_present(conn):
    # The fast path only applies to a bare view count; with a filter the result
    # must still be correct.
    add_job(conn, "1", company="Acme")
    add_job(conn, "2", company="Beta")
    assert (
        q.count_jobs(conn, view="jobs_today", filters=q.JobFilter(companies=("Beta",)))
        == 1
    )


def test_count_jobs_falls_back_when_text_is_present(conn):
    add_job(conn, "1", title="Python Engineer", description="python")
    add_job(conn, "2", title="Rust Engineer", description="rust")
    assert q.count_jobs(conn, view="jobs_today", text="python") == 1


def test_today_paging_does_not_degrade_with_depth(conn):
    # A sanity check that the index is doing its job: the tenth page must not
    # cost an order of magnitude more than the first. Thresholds are loose so
    # this is not timing-sensitive on a loaded machine.
    import time

    for i in range(500):
        add_job(conn, f"{i:04d}")
    def timed(offset):
        best = None
        for _ in range(3):
            start = time.perf_counter()
            q.jobs_today(conn, limit=50, offset=offset)
            elapsed = time.perf_counter() - start
            best = elapsed if best is None else min(best, elapsed)
        return best

    first = timed(0)
    deep = timed(450)
    assert deep < max(first * 20, 0.2), f"first={first:.4f}s deep={deep:.4f}s"
