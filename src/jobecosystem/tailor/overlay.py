"""The declarative tailoring overlay and how it merges into the base resume.

An overlay describes a *desired end state*, not a list of edits: which skills
and roles to keep and in what order, which bullets to keep per role, and optional
replacement text for a bullet or the summary. Applying it is a merge, so it is
idempotent, and the shape simply has nowhere to put a new employer, title, date,
degree, or certification -- facts can only be selected from the base, never
invented.

Merge semantics (see :func:`apply`):

* ``summary`` -- replaces the base summary when present.
* ``skill_ids`` -- the full ordered skill selection; absent means "all, base
  order". Every id must exist in the base.
* ``role_order`` -- the full ordered role selection; absent means "all, base
  order"; a role omitted here is dropped from the tailored resume.
* ``roles[role_id].bullets`` -- that role's full ordered bullet selection, with
  optional rephrase per bullet; a role absent from ``roles`` keeps its base
  bullets.
* ``base_hash`` -- must match the base it is applied to.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import (
    Bullet,
    BulletEdit,
    Overlay,
    Resume,
    Role,
    RoleEdit,
    TailorError,
    Target,
)
from .resume import (
    _as_dict,
    _as_int,
    _as_list,
    _as_str,
    _opt_str,
    _str_list,
    canonical_json,
    resume_hash,
)


# ---------------------------------------------------------------------------
# JSON <-> dataclasses
# ---------------------------------------------------------------------------

def _opt_str_list(value: Any, where: str) -> list[str] | None:
    if value is None:
        return None
    return _str_list(value, where)


def _target(value: Any) -> Target:
    if value is None:
        return Target()
    data = _as_dict(value, "target")
    return Target(
        job_id=None if data.get("job_id") is None else _as_int(data["job_id"], "target.job_id"),
        company=_opt_str(data.get("company"), "target.company"),
        title=_opt_str(data.get("title"), "target.title"),
        url=_opt_str(data.get("url"), "target.url"),
    )


def _bullet_edit(value: Any, where: str) -> BulletEdit:
    data = _as_dict(value, where)
    return BulletEdit(id=_as_str(data.get("id"), f"{where}.id"),
                      text=_opt_str(data.get("text"), f"{where}.text"))


def _role_edit(value: Any, where: str) -> RoleEdit:
    data = _as_dict(value, where)
    return RoleEdit(bullets=[
        _bullet_edit(item, f"{where}.bullets[{i}]")
        for i, item in enumerate(_as_list(data.get("bullets", []), f"{where}.bullets"))
    ])


def overlay_from_dict(value: Any) -> Overlay:
    """Build an :class:`Overlay` from parsed JSON, or raise :class:`TailorError`."""
    data = _as_dict(value, "overlay")
    roles_raw = _as_dict(data.get("roles", {}), "roles")
    roles: dict[str, RoleEdit] = {}
    for role_id, edit in roles_raw.items():
        roles[_as_str(role_id, "roles key")] = _role_edit(edit, f"roles.{role_id}")
    return Overlay(
        base_hash=_as_str(data.get("base_hash"), "base_hash"),
        schema_version=_as_int(data.get("schema_version", 1), "schema_version"),
        target=_target(data.get("target")),
        summary=_opt_str(data.get("summary"), "summary"),
        skill_ids=_opt_str_list(data.get("skill_ids"), "skill_ids"),
        role_order=_opt_str_list(data.get("role_order"), "role_order"),
        roles=roles,
        notes=_opt_str(data.get("notes"), "notes"),
    )


def overlay_to_dict(overlay: Overlay) -> dict:
    """The JSON shape of an overlay, omitting unset optionals."""
    data: dict[str, Any] = {
        "schema_version": overlay.schema_version,
        "base_hash": overlay.base_hash,
        "target": {key: value for key, value in {
            "job_id": overlay.target.job_id,
            "company": overlay.target.company,
            "title": overlay.target.title,
            "url": overlay.target.url,
        }.items() if value is not None},
    }
    if overlay.summary is not None:
        data["summary"] = overlay.summary
    if overlay.skill_ids is not None:
        data["skill_ids"] = overlay.skill_ids
    if overlay.role_order is not None:
        data["role_order"] = overlay.role_order
    if overlay.roles:
        data["roles"] = {
            role_id: {
                "bullets": [
                    {key: value for key, value in {"id": b.id, "text": b.text}.items()
                     if value is not None}
                    for b in edit.bullets
                ]
            }
            for role_id, edit in overlay.roles.items()
        }
    if overlay.notes is not None:
        data["notes"] = overlay.notes
    return data


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------

def load(path: str | os.PathLike[str]) -> Overlay:
    """Load an overlay JSON file."""
    overlay_file = Path(path).expanduser()
    try:
        text = overlay_file.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise TailorError(f"overlay not found: {overlay_file}") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise TailorError(f"{overlay_file}: not valid JSON: {error}") from error
    return overlay_from_dict(data)


def save(overlay: Overlay, path: str | os.PathLike[str]) -> Path:
    """Write an overlay canonically; returns the path written."""
    overlay_file = Path(path).expanduser()
    overlay_file.parent.mkdir(parents=True, exist_ok=True)
    overlay_file.write_text(canonical_json(overlay_to_dict(overlay)), encoding="utf-8")
    return overlay_file


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def _require(mapping: dict, key: str, kind: str) -> Any:
    if key not in mapping:
        raise TailorError(f"unknown {kind} {key!r} referenced by the overlay")
    return mapping[key]


def _resolve_role(role: Role, edit: RoleEdit | None) -> Role:
    if edit is None:
        return role                      # absent from roles => keep base bullets
    by_id = {bullet.id: bullet for bullet in role.bullets}
    bullets: list[Bullet] = []
    for bullet_edit in edit.bullets:
        base_bullet = _require(by_id, bullet_edit.id, f"bullet of {role.id}")
        text = base_bullet.text if bullet_edit.text is None else bullet_edit.text
        bullets.append(replace(base_bullet, text=text))
    return replace(role, bullets=bullets)


def apply(base: Resume, overlay: Overlay) -> Resume:
    """Merge an overlay into its base, returning the tailored :class:`Resume`.

    Raises :class:`TailorError` when the hash does not match or the overlay
    references something the base does not have. Semantic rules (number
    preservation, bounds) belong to :mod:`jobecosystem.tailor.validate`, which
    callers should run first.
    """
    if overlay.base_hash != resume_hash(base):
        raise TailorError(
            "overlay base_hash does not match the base resume; regenerate it"
        )

    summary = base.basics.summary if overlay.summary is None else overlay.summary
    basics = replace(base.basics, summary=summary)

    if overlay.skill_ids is None:
        skills = list(base.skills)
    else:
        by_id = {skill.id: skill for skill in base.skills}
        skills = [_require(by_id, skill_id, "skill") for skill_id in overlay.skill_ids]

    by_role = {role.id: role for role in base.roles}
    order = [role.id for role in base.roles] if overlay.role_order is None else overlay.role_order
    roles = [
        _resolve_role(_require(by_role, role_id, "role"), overlay.roles.get(role_id))
        for role_id in order
    ]

    return replace(base, basics=basics, skills=skills, roles=roles)
