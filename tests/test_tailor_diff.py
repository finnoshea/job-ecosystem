"""Tests for jobecosystem.tailor.diff."""

from __future__ import annotations

from jobecosystem.tailor import diff as D
from jobecosystem.tailor import overlay as O
from jobecosystem.tailor import resume as R
from jobecosystem.tailor.models import BulletEdit, Overlay, RoleEdit


def make(base, **overrides) -> Overlay:
    fields = {"base_hash": R.resume_hash(base)}
    fields.update(overrides)
    return Overlay(**fields)


def tailored(base, **overrides):
    return O.apply(base, make(base, **overrides))


def test_no_changes_is_empty(sample_resume):
    assert D.diff(sample_resume, tailored(sample_resume)) == []


def test_summary_change_is_reported(sample_resume):
    changes = D.diff(sample_resume, tailored(sample_resume, summary="New."))
    assert [c.kind for c in changes] == ["summary"]
    assert changes[0].after == "New."


def test_skill_change_is_reported(sample_resume):
    changes = D.diff(sample_resume, tailored(sample_resume, skill_ids=["skill.aws"]))
    kinds = [c.kind for c in changes]
    assert "skills" in kinds


def test_role_selection_is_reported(sample_resume):
    changes = D.diff(sample_resume, tailored(sample_resume, role_order=["role.two"]))
    assert any(c.kind == "roles" for c in changes)


def test_bullet_selection_is_reported(sample_resume):
    changes = D.diff(sample_resume, tailored(sample_resume, roles={
        "role.one": RoleEdit(bullets=[BulletEdit("b.one.1")])}))
    bullet_changes = [c for c in changes if c.kind == "bullets"]
    assert bullet_changes and bullet_changes[0].after == "b.one.1"


def test_rephrase_is_reported(sample_resume):
    changes = D.diff(sample_resume, tailored(sample_resume, roles={
        "role.one": RoleEdit(bullets=[
            BulletEdit("b.one.1", "Reduced latency 40% with caches."),
            BulletEdit("b.one.2"),
        ])}))
    rephrases = [c for c in changes if c.kind == "rephrase"]
    assert len(rephrases) == 1
    assert rephrases[0].after == "Reduced latency 40% with caches."


def test_render_diff_no_changes():
    assert D.render_diff([]) == "no changes"


def test_render_diff_shows_before_and_after(sample_resume):
    text = D.render_diff(D.diff(sample_resume, tailored(sample_resume, summary="New.")))
    assert "[summary]" in text
    assert "  - Original summary." in text
    assert "  + New." in text
