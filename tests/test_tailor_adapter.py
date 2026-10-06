"""Tests for jobecosystem.tailor.adapter: prompt, parsing, validation, LLM seam."""

from __future__ import annotations

import json

import pytest

from jobecosystem.tailor import adapter
from jobecosystem.tailor import resume as R
from jobecosystem.tailor.models import TailorError


def completion(payload: dict) -> str:
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

def test_prompt_includes_the_job_and_the_base(sample_resume):
    prompt = adapter.build_prompt(sample_resume, company="Acme", title="Data Engineer",
                                  description="Build pipelines.", job_id=42)
    assert "Data Engineer" in prompt
    assert "Acme" in prompt
    assert "Build pipelines." in prompt
    assert "skill.python" in prompt          # base ids are shown
    assert "role.one" in prompt
    assert "Declarative" in prompt or "declarative" in prompt
    assert "number" in prompt                # metric rule is stated


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_plain_json():
    assert adapter.parse_overlay('{"summary": "x"}') == {"summary": "x"}


def test_parse_fenced_json():
    text = '```json\n{"summary": "x"}\n```'
    assert adapter.parse_overlay(text) == {"summary": "x"}


def test_parse_json_with_surrounding_prose():
    assert adapter.parse_overlay('here you go: {"summary": "x"} done') == {"summary": "x"}


def test_parse_without_json_raises():
    with pytest.raises(TailorError, match="no JSON object"):
        adapter.parse_overlay("no json here")


def test_parse_invalid_json_raises():
    with pytest.raises(TailorError, match="not valid JSON"):
        adapter.parse_overlay("{summary: x}")


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------

def test_propose_returns_a_validated_overlay(sample_resume):
    raw = completion({"summary": "Tailored summary.",
                      "skill_ids": ["skill.python"],
                      "role_order": ["role.one"],
                      "roles": {"role.one": {"bullets": [{"id": "b.one.1"}]}}})
    overlay = adapter.propose_overlay(sample_resume, company="Acme",
                                      title="Engineer", complete=lambda p: raw)
    assert overlay.base_hash == R.resume_hash(sample_resume)
    assert overlay.target.company == "Acme"
    assert overlay.skill_ids == ["skill.python"]


def test_propose_fills_the_target_even_if_the_model_omits_it(sample_resume):
    overlay = adapter.propose_overlay(
        sample_resume, company="Acme", title="Engineer", job_id=9,
        complete=lambda p: completion({"summary": "S."}),
    )
    assert (overlay.target.job_id, overlay.target.company, overlay.target.title) == (
        9, "Acme", "Engineer"
    )


def test_propose_rejects_a_number_change(sample_resume):
    raw = completion({"roles": {"role.one": {"bullets": [
        {"id": "b.one.1", "text": "Cut latency 99% using caches."}
    ]}}})
    with pytest.raises(TailorError, match="introduced number"):
        adapter.propose_overlay(sample_resume, company="Acme", title="E",
                                complete=lambda p: raw)


def test_propose_rejects_an_unknown_id(sample_resume):
    raw = completion({"skill_ids": ["skill.nope"]})
    with pytest.raises(TailorError, match="unknown skill"):
        adapter.propose_overlay(sample_resume, company="Acme", title="E",
                                complete=lambda p: raw)


# ---------------------------------------------------------------------------
# default completer
# ---------------------------------------------------------------------------

def test_default_completer_requires_configuration(monkeypatch):
    monkeypatch.delenv("TAILOR_COMPLETE_COMMAND", raising=False)
    with pytest.raises(TailorError, match="TAILOR_COMPLETE_COMMAND"):
        adapter.default_completer()


def test_default_completer_round_trips_stdin(monkeypatch):
    monkeypatch.setenv("TAILOR_COMPLETE_COMMAND", "cat")
    complete = adapter.default_completer()
    assert complete("hello prompt") == "hello prompt"


def test_default_completer_reports_command_failure(monkeypatch):
    monkeypatch.setenv("TAILOR_COMPLETE_COMMAND", "false")
    with pytest.raises(TailorError, match="failed"):
        adapter.default_completer()("prompt")
