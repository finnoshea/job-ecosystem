"""Structural diff between the base resume and a tailored one.

The approval gate needs to show what actually changed, at the level of roles,
bullets, and rephrasings -- not a unified text diff of a rendered document,
which would be dominated by layout. Both resumes carry stable ids, so match by
id and report order/selection and text changes separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Resume


@dataclass(slots=True)
class Change:
    """One reviewable difference. ``before``/``after`` are display strings."""

    kind: str
    subject: str
    before: str
    after: str


def _labels(ids: list[str], items: dict[str, str]) -> str:
    return ", ".join(items.get(item, item) for item in ids)


def diff(base: Resume, tailored: Resume) -> list[Change]:
    """Compare two resumes, returning changes worth a human's attention."""
    changes: list[Change] = []

    if base.basics.summary != tailored.basics.summary:
        changes.append(Change(
            kind="summary", subject="summary",
            before=(base.basics.summary or "")[:400],
            after=(tailored.basics.summary or "")[:400],
        ))

    base_skills = [skill.id for skill in base.skills]
    tailored_skills = [skill.id for skill in tailored.skills]
    if base_skills != tailored_skills:
        names = {skill.id: skill.name for skill in base.skills}
        changes.append(Change(
            kind="skills", subject="skills",
            before=_labels(base_skills, names),
            after=_labels(tailored_skills, names),
        ))

    base_roles = [role.id for role in base.roles]
    tailored_roles = [role.id for role in tailored.roles]
    if base_roles != tailored_roles:
        titles = {role.id: f"{role.title} @ {role.company}" for role in base.roles}
        changes.append(Change(
            kind="roles", subject="role order/selection",
            before=_labels(base_roles, titles),
            after=_labels(tailored_roles, titles),
        ))

    base_by_id = {role.id: role for role in base.roles}
    for role in tailored.roles:
        original = base_by_id.get(role.id)
        if original is None:                       # pragma: no cover - apply forbids it
            continue
        original_bullets = [bullet.id for bullet in original.bullets]
        tailored_bullets = [bullet.id for bullet in role.bullets]
        if original_bullets != tailored_bullets:
            changes.append(Change(
                kind="bullets", subject=f"{role.title} @ {role.company}",
                before=", ".join(original_bullets),
                after=", ".join(tailored_bullets),
            ))
        original_text = {bullet.id: bullet.text for bullet in original.bullets}
        for bullet in role.bullets:
            if bullet.id in original_text and original_text[bullet.id] != bullet.text:
                changes.append(Change(
                    kind="rephrase", subject=f"{role.id}.{bullet.id}",
                    before=original_text[bullet.id],
                    after=bullet.text,
                ))

    return changes


def render_diff(changes: list[Change]) -> str:
    """Human-readable summary for the approval screen."""
    if not changes:
        return "no changes"
    lines: list[str] = []
    for change in changes:
        lines.append(f"[{change.kind}] {change.subject}")
        lines.append(f"  - {change.before}")
        lines.append(f"  + {change.after}")
    return "\n".join(lines)
