"""Tests for jobecosystem.tailor.validate: the semantic guardrails."""

from __future__ import annotations

from jobecosystem.tailor import resume as R
from jobecosystem.tailor import validate as V
from jobecosystem.tailor.models import BulletEdit, Overlay, RoleEdit


def make(base, **overrides) -> Overlay:
    fields = {"base_hash": R.resume_hash(base)}
    fields.update(overrides)
    return Overlay(**fields)


# ---------------------------------------------------------------------------
# number tokens
# ---------------------------------------------------------------------------

def test_numbers_normalize_formatting():
    assert V.number_tokens("$1,000") == ["1000"]
    assert V.number_tokens("40%") == ["40"]
    assert V.number_tokens("40 percent") == ["40"]
    assert V.number_tokens("2.50") == ["2.5"]
    assert V.number_tokens("0.0") == ["0"]
    assert V.number_tokens("no numbers here") == []


# ---------------------------------------------------------------------------
# base resume
# ---------------------------------------------------------------------------

def test_a_valid_resume_has_no_errors(sample_resume):
    assert V.validate_resume(sample_resume) == []


def test_duplicate_ids_are_reported(sample_resume):
    sample_resume.skills[1].id = sample_resume.skills[0].id
    errors = V.validate_resume(sample_resume)
    assert any("duplicate id" in e for e in errors)


def test_bad_date_is_reported(sample_resume):
    sample_resume.roles[0].start = "2020"
    errors = V.validate_resume(sample_resume)
    assert any("YYYY-MM" in e for e in errors)


def test_unknown_render_section_is_reported(sample_resume):
    sample_resume.render.section_order = ["summary", "nonsense"]
    errors = V.validate_resume(sample_resume)
    assert any("nonsense" in e for e in errors)


def test_overlay_cannot_add_a_summary_when_the_base_has_none(sample_resume):
    sample_resume.basics.summary = None
    errors = V.validate_overlay(sample_resume, make(sample_resume, summary="Invented."))
    assert any("may not add" in e for e in errors)


def test_overlay_may_replace_an_existing_summary(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, summary="Rewritten."))
    assert errors == []


def test_unknown_page_break_section_is_reported(sample_resume):
    sample_resume.render.page_break_before = ["nope"]
    errors = V.validate_resume(sample_resume)
    assert any("page_break_before" in e for e in errors)


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

def test_a_valid_overlay_has_no_errors(sample_resume):
    overlay = make(sample_resume, summary="S", skill_ids=["skill.aws"],
                   role_order=["role.two"],
                   roles={"role.two": RoleEdit(bullets=[BulletEdit("b.two.1")])})
    assert V.validate_overlay(sample_resume, overlay) == []


def test_base_hash_mismatch_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, Overlay(base_hash="sha256:" + "0" * 64))
    assert any("base_hash" in e for e in errors)


def test_unknown_skill_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, skill_ids=["skill.x"]))
    assert any("unknown skill" in e for e in errors)


def test_unknown_role_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, role_order=["role.x"]))
    assert any("unknown role" in e for e in errors)


def test_unknown_bullet_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[BulletEdit("b.one.9")])}))
    assert any("unknown bullet" in e for e in errors)


def test_duplicate_bullet_selection_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[BulletEdit("b.one.1"), BulletEdit("b.one.1")])}))
    assert any("duplicate id" in e for e in errors)


def test_dropped_number_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[
            BulletEdit("b.one.1", "Reduced latency using caches.")])}))
    assert any("dropped number" in e for e in errors)


def test_introduced_number_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[
            BulletEdit("b.one.1", "Cut latency 99% with caches.")])}))
    assert any("introduced number" in e for e in errors)


def test_missing_metric_is_reported(sample_resume):
    # 40 appears as a non-number word, so the digit is gone even though the
    # sentence reads fine.
    errors = V.validate_overlay(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[
            BulletEdit("b.one.1", "Cut latency forty percent with caches.")])}))
    assert any("missing metric" in e for e in errors)


def test_a_good_rephrase_keeps_the_numbers(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[
            BulletEdit("b.one.1", "Reduced latency by 40% with a cache layer.")])}))
    assert errors == []


def test_blank_rephrase_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[BulletEdit("b.one.1", "   ")])}))
    assert any("blank" in e for e in errors)


def test_overlong_summary_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(
        sample_resume, summary="x" * (V.MAX_SUMMARY_CHARS + 1)))
    assert any("exceeds" in e for e in errors)


def test_overlong_bullet_is_reported(sample_resume):
    errors = V.validate_overlay(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[
            BulletEdit("b.one.1", "40% " + "x" * V.MAX_BULLET_CHARS)])}))
    assert any("exceeds" in e for e in errors)
