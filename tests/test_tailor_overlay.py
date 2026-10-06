"""Tests for jobecosystem.tailor.overlay: the declarative merge semantics."""

from __future__ import annotations

import pytest

from jobecosystem.tailor import overlay as O
from jobecosystem.tailor import resume as R
from jobecosystem.tailor.models import (
    BulletEdit,
    Overlay,
    RoleEdit,
    TailorError,
    Target,
)


def make(base, **overrides) -> Overlay:
    fields = {"base_hash": R.resume_hash(base)}
    fields.update(overrides)
    return Overlay(**fields)


def test_an_empty_overlay_is_the_base(sample_resume):
    tailored = O.apply(sample_resume, make(sample_resume))
    assert R.resume_to_dict(tailored) == R.resume_to_dict(sample_resume)


def test_summary_replaces_the_base(sample_resume):
    tailored = O.apply(sample_resume, make(sample_resume, summary="Tailored."))
    assert tailored.basics.summary == "Tailored."
    assert sample_resume.basics.summary == "Original summary."


def test_skill_ids_select_and_order(sample_resume):
    tailored = O.apply(sample_resume, make(
        sample_resume, skill_ids=["skill.aws", "skill.python"]
    ))
    assert [s.id for s in tailored.skills] == ["skill.aws", "skill.python"]


def test_role_order_selects_and_drops(sample_resume):
    tailored = O.apply(sample_resume, make(sample_resume, role_order=["role.two"]))
    assert [r.id for r in tailored.roles] == ["role.two"]


def test_bullets_select_and_order(sample_resume):
    tailored = O.apply(sample_resume, make(
        sample_resume,
        roles={"role.one": RoleEdit(bullets=[
            BulletEdit("b.one.2"), BulletEdit("b.one.1")])},
    ))
    role = tailored.roles[0]
    assert [b.id for b in role.bullets] == ["b.one.2", "b.one.1"]


def test_a_role_absent_from_roles_keeps_its_bullets(sample_resume):
    tailored = O.apply(sample_resume, make(
        sample_resume, roles={"role.one": RoleEdit(bullets=[BulletEdit("b.one.1")])}
    ))
    role_two = next(r for r in tailored.roles if r.id == "role.two")
    assert [b.id for b in role_two.bullets] == ["b.two.1"]


def test_rephrase_overrides_only_that_bullet(sample_resume):
    tailored = O.apply(sample_resume, make(sample_resume, roles={
        "role.one": RoleEdit(bullets=[
            BulletEdit("b.one.1", "Reduced latency 40% with a cache layer."),
            BulletEdit("b.one.2"),
        ])
    }))
    role = tailored.roles[0]
    assert role.bullets[0].text == "Reduced latency 40% with a cache layer."
    assert role.bullets[1].text == "Led 5 engineers on a migration."


def test_apply_rejects_a_mismatched_base_hash(sample_resume):
    with pytest.raises(TailorError, match="base_hash"):
        O.apply(sample_resume, Overlay(base_hash="sha256:" + "0" * 64))


def test_apply_rejects_an_unknown_skill(sample_resume):
    with pytest.raises(TailorError, match="unknown skill"):
        O.apply(sample_resume, make(sample_resume, skill_ids=["skill.nope"]))


def test_apply_rejects_an_unknown_role(sample_resume):
    with pytest.raises(TailorError, match="unknown role"):
        O.apply(sample_resume, make(sample_resume, role_order=["role.nope"]))


def test_apply_rejects_a_bullet_from_another_role(sample_resume):
    with pytest.raises(TailorError, match="unknown bullet"):
        O.apply(sample_resume, make(sample_resume, roles={
            "role.one": RoleEdit(bullets=[BulletEdit("b.two.1")])
        }))


def test_overlay_round_trips_through_dict(sample_resume):
    overlay = make(sample_resume, summary="S", skill_ids=["skill.aws"],
                   role_order=["role.two"],
                   roles={"role.two": RoleEdit(bullets=[BulletEdit("b.two.1")])},
                   target=Target(job_id=7, company="Acme", title="Eng"),
                   notes="because")
    assert O.overlay_from_dict(O.overlay_to_dict(overlay)) == overlay


def test_overlay_dict_omits_unset_optionals(sample_resume):
    data = O.overlay_to_dict(make(sample_resume))
    assert "summary" not in data
    assert "skill_ids" not in data
    assert "role_order" not in data
    assert "roles" not in data
    assert data["base_hash"].startswith("sha256:")


def test_overlay_save_and_load(sample_resume, tmp_path):
    overlay = make(sample_resume, summary="S")
    path = tmp_path / "overlay.json"
    O.save(overlay, path)
    assert O.load(path) == overlay


def test_overlay_load_missing_file_raises(tmp_path):
    with pytest.raises(TailorError, match="not found"):
        O.load(tmp_path / "nope.json")
