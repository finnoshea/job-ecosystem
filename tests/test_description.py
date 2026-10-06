"""Tests for the shared description fetcher and its per-source dispatch.

The Workday machinery is exercised in detail by ``test_workday_description.py``;
these tests cover the cross-source parts: routing a payload to the right parser,
and fetching/storing rows from more than one source through one queue.
"""

from __future__ import annotations

import pytest

from jobecosystem.core.models import Job
from jobecosystem.ingest import upsert
from jobecosystem.ingest.sources import description as desc


def store(conn, source, description_url, external_id="X", **overrides):
    """Insert a description-less row of the given source."""
    fields = {
        "source": source,
        "external_id": external_id,
        "company": "Acme",
        "title": "Engineer",
        "description_url": description_url,
    }
    fields.update(overrides)
    result = upsert.upsert_job(conn, Job(**fields))
    conn.commit()
    return result.job_id


GREENHOUSE_DETAIL = {"content": "&lt;p&gt;Greenhouse body&lt;/p&gt;"}
SMARTRECRUITERS_DETAIL = {
    "jobAd": {"sections": {
        "jobDescription": {"title": "Role", "text": "<p>SmartRecruiters body</p>"},
    }}
}
WORKDAY_DETAIL = {"jobPostingInfo": {"jobDescription": "<p>Workday body</p>"}}


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_source_family_is_the_prefix():
    assert desc.source_family("workday:asml:Site") == "workday"
    assert desc.source_family("greenhouse:reddit") == "greenhouse"
    assert desc.source_family("smartrecruiters:RedBull") == "smartrecruiters"


def test_dispatches_to_the_workday_parser():
    assert desc.parse_description(WORKDAY_DETAIL, "workday:x:y") == "Workday body"


def test_dispatches_to_the_smartrecruiters_parser():
    text = desc.parse_description(SMARTRECRUITERS_DETAIL, "smartrecruiters:RedBull")
    assert "SmartRecruiters body" in text


def test_dispatches_to_the_greenhouse_parser():
    assert desc.parse_description(GREENHOUSE_DETAIL, "greenhouse:reddit") == (
        "Greenhouse body"
    )


def test_a_source_without_a_parser_is_reported():
    # Ashby needs none: its listing already carries the description.
    with pytest.raises(desc.DescriptionError, match="no description parser"):
        desc.parse_description({}, "ashby:ramp")


def test_the_wrong_payload_for_a_source_is_an_error():
    with pytest.raises(desc.DescriptionError):
        desc.parse_description(WORKDAY_DETAIL, "greenhouse:reddit")


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def test_known_url_uses_the_given_source():
    outcome = desc.fetch_description_from_known_url(
        1, "Engineer", "Acme", "https://x/j",
        source="greenhouse:reddit",
        fetch_json=lambda url: GREENHOUSE_DETAIL,
    )
    assert outcome.ok
    assert outcome.fetched.text == "Greenhouse body"


def test_fetch_end_to_end_dispatches_by_the_row_source(conn):
    job_id = store(conn, "greenhouse:reddit", "https://x/gh")
    result = desc.fetch_description(
        conn, job_id, fetch_json=lambda url: GREENHOUSE_DETAIL
    )
    assert result.written is True
    row = conn.execute(
        "SELECT description, content_hash FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    assert row["description"] == "Greenhouse body"
    assert row["content_hash"] is not None


def test_a_payload_the_parser_rejects_is_an_error_not_a_crash(conn):
    job_id = store(conn, "greenhouse:reddit", "https://x/gh")
    result = desc.fetch_description(
        conn, job_id, fetch_json=lambda url: WORKDAY_DETAIL
    )
    assert result.written is False
    assert result.error is not None
    assert conn.execute(
        "SELECT description FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()["description"] is None


# ---------------------------------------------------------------------------
# transport error reporting
# ---------------------------------------------------------------------------

class _FakeResponse:
    """A response whose body is not JSON, for exercising the error message."""

    def __init__(self, status_code, content_type=None):
        self.status_code = status_code
        self.headers = {"content-type": content_type} if content_type else {}

    def json(self):
        import json

        raise json.JSONDecodeError("Expecting value", "", 0)


def _fake_get(response):
    def get(url, **kwargs):
        return response

    return get


def test_http_error_names_the_status_and_content_type(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", _fake_get(_FakeResponse(429, "text/html")))
    with pytest.raises(desc.DescriptionError, match=r"HTTP 429 \(text/html"):
        desc._http_get_json("https://x.test/j")


def test_non_json_response_names_the_status_and_content_type(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx, "get", _fake_get(_FakeResponse(200, "text/html; charset=UTF-8"))
    )
    with pytest.raises(
        desc.DescriptionError, match=r"HTTP 200 \(text/html.*not JSON"
    ):
        desc._http_get_json("https://x.test/j")


def test_missing_content_type_is_reported_as_unknown(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "get", _fake_get(_FakeResponse(200, None)))
    with pytest.raises(desc.DescriptionError, match=r"HTTP 200 \(unknown\)"):
        desc._http_get_json("https://x.test/j")


def test_batch_handles_mixed_sources_through_one_queue(conn):
    store(conn, "greenhouse:reddit", "https://x/gh", external_id="GH")
    store(conn, "smartrecruiters:RedBull", "https://x/sr", external_id="SR")
    payloads = {"https://x/gh": GREENHOUSE_DETAIL, "https://x/sr": SMARTRECRUITERS_DETAIL}

    summary = desc.fetch_descriptions(
        conn, sleep=lambda seconds: None, fetch_json=lambda url: payloads[url]
    )

    assert summary.attempted == 2
    assert summary.written == 2
    assert summary.ok
    rows = conn.execute(
        "SELECT source, description FROM jobs ORDER BY source"
    ).fetchall()
    assert rows[0]["source"] == "greenhouse:reddit"
    assert "Greenhouse body" in rows[0]["description"]
    assert "SmartRecruiters body" in rows[1]["description"]
