"""The enforcement layer: what an LLM edit is allowed to do.

Shape (types, required fields) is enforced while parsing (``resume.py`` /
``overlay.py``). This module adds the semantic rules that keep a declarative
overlay truthful:

* an overlay is pinned to a specific base by ``base_hash``;
* every id it references exists (skills, roles, and bullets of the right role);
* the only id-less content it may produce is rephrased bullet text and a
  summary -- it has no field in which to add an employer, title, date, degree,
  or certification, so those facts are immutable by construction;
* a rephrased bullet must preserve every number from the original (and may not
  introduce a number that was not there), which is what stops the model quietly
  changing a metric.

Errors are returned as a list of human-readable strings; the ``assert_*``
helpers raise :class:`TailorError` with all of them joined.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from .models import Overlay, Resume, TailorError
from .resume import resume_hash

#: Content sections the renderer knows about.
SECTIONS = ("summary", "skills", "roles", "projects", "education",
            "certifications", "awards", "publications")

#: Soft caps, to stop a model from producing a wall of text.
MAX_SUMMARY_CHARS = 1200
MAX_BULLET_CHARS = 500
MAX_NOTES_CHARS = 2000

#: A date is ``YYYY-MM``; ``end``/``date`` may be absent, not arbitrary prose.
_DATE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

#: A number: digits with optional thousands separators and a decimal part.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def number_tokens(text: str) -> list[str]:
    """Canonical numeric tokens in ``text`` (commas removed, zeros trimmed).

    ``"$1,000"`` and ``"1000"`` both yield ``"1000"``; ``"40%"`` and ``"40
    percent"`` both yield ``"40"``. This is the unit the metric-preservation
    check compares, so formatting differences are not flagged as changes.
    """
    tokens: list[str] = []
    for match in _NUMBER.finditer(text):
        raw = match.group().replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:            # pragma: no cover - regex guards this
            continue
        tokens.append(format(value.normalize(), "f"))
    return tokens


def _unique(ids: list[str], where: str, errors: list[str]) -> None:
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        errors.append(f"{where}: duplicate id(s): {', '.join(duplicates)}")


def _check_date(value: str | None, where: str, errors: list[str]) -> None:
    if value is not None and not _DATE.match(value):
        errors.append(f"{where}: {value!r} is not YYYY-MM")


# ---------------------------------------------------------------------------
# base resume
# ---------------------------------------------------------------------------

def validate_resume(resume: Resume) -> list[str]:
    """Semantic checks on the base resume. Returns a list of errors."""
    errors: list[str] = []

    if resume.schema_version < 1:
        errors.append(f"schema_version: {resume.schema_version} is not supported")

    _unique([s.id for s in resume.skills], "skills", errors)
    _unique([r.id for r in resume.roles], "roles", errors)
    _unique([p.id for p in resume.projects], "projects", errors)
    _unique([e.id for e in resume.education], "education", errors)
    _unique([c.id for c in resume.certifications], "certifications", errors)
    _unique([a.id for a in resume.awards], "awards", errors)
    for role in resume.roles:
        _unique([b.id for b in role.bullets], f"role {role.id}.bullets", errors)
        _check_date(role.start, f"role {role.id}.start", errors)
        _check_date(role.end, f"role {role.id}.end", errors)
    for edu in resume.education:
        _check_date(edu.start, f"education {edu.id}.start", errors)
        _check_date(edu.end, f"education {edu.id}.end", errors)
    for cert in resume.certifications:
        _check_date(cert.date, f"certification {cert.id}.date", errors)
    for award in resume.awards:
        _check_date(award.date, f"award {award.id}.date", errors)

    unknown = [name for name in resume.render.section_order if name not in SECTIONS]
    if unknown:
        errors.append(f"render.section_order: unknown section(s): {', '.join(unknown)}")
    unknown_breaks = [
        name for name in resume.render.page_break_before if name not in SECTIONS
    ]
    if unknown_breaks:
        errors.append(
            "render.page_break_before: unknown section(s):"
            f" {', '.join(unknown_breaks)}"
        )
    if resume.render.max_roles is not None and resume.render.max_roles < 1:
        errors.append("render.max_roles: must be at least 1 when set")
    if resume.render.bullets_per_role is not None and resume.render.bullets_per_role < 1:
        errors.append("render.bullets_per_role: must be at least 1 when set")

    return errors


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

def _check_rephrase(original: str, rewritten: str, metrics: list[str],
                    where: str, errors: list[str]) -> None:
    original_numbers = Counter(number_tokens(original))
    new_numbers = Counter(number_tokens(rewritten))

    missing = original_numbers - new_numbers
    if missing:
        errors.append(
            f"{where}: rephrase dropped number(s): {', '.join(sorted(missing))}"
        )
    allowed = set(original_numbers) | {
        number for metric in metrics for number in number_tokens(metric)
    }
    introduced = sorted(number for number in new_numbers if number not in allowed)
    if introduced:
        errors.append(
            f"{where}: rephrase introduced number(s): {', '.join(introduced)}"
        )
    for metric in metrics:
        absent = [number for number in number_tokens(metric) if number not in new_numbers]
        if absent:
            errors.append(
                f"{where}: rephrase is missing metric number(s): {', '.join(absent)}"
            )


def validate_overlay(base: Resume, overlay: Overlay) -> list[str]:
    """Semantic checks on an overlay against its base. Returns a list of errors."""
    errors: list[str] = []

    if overlay.schema_version < 1:
        errors.append(f"schema_version: {overlay.schema_version} is not supported")
    if overlay.base_hash != resume_hash(base):
        errors.append(
            "base_hash: overlay was written against a different base resume"
        )

    if overlay.summary is not None:
        # A field that is blank in the base is closed: the overlay may not fill
        # it in. Otherwise a model given no summary to rewrite would simply
        # invent one, and nothing else here would catch ungrounded prose.
        if base.basics.summary is None:
            errors.append(
                "summary: the base resume has no summary, so the overlay may"
                " not add one"
            )
        if not overlay.summary.strip():
            errors.append("summary: must not be blank when present")
        if len(overlay.summary) > MAX_SUMMARY_CHARS:
            errors.append(
                f"summary: {len(overlay.summary)} chars exceeds {MAX_SUMMARY_CHARS}"
            )

    skills = {skill.id: skill for skill in base.skills}
    if overlay.skill_ids is not None:
        _unique(overlay.skill_ids, "skill_ids", errors)
        for skill_id in overlay.skill_ids:
            if skill_id not in skills:
                errors.append(f"skill_ids: unknown skill {skill_id!r}")

    roles = {role.id: role for role in base.roles}
    if overlay.role_order is not None:
        _unique(overlay.role_order, "role_order", errors)
        for role_id in overlay.role_order:
            if role_id not in roles:
                errors.append(f"role_order: unknown role {role_id!r}")

    for role_id, edit in overlay.roles.items():
        role = roles.get(role_id)
        if role is None:
            errors.append(f"roles: unknown role {role_id!r}")
            continue
        bullet_ids = [bullet_edit.id for bullet_edit in edit.bullets]
        _unique(bullet_ids, f"roles.{role_id}.bullets", errors)
        by_id = {bullet.id: bullet for bullet in role.bullets}
        for bullet_edit in edit.bullets:
            base_bullet = by_id.get(bullet_edit.id)
            if base_bullet is None:
                errors.append(
                    f"roles.{role_id}.bullets: unknown bullet {bullet_edit.id!r}"
                )
                continue
            if bullet_edit.text is None:
                continue
            where = f"roles.{role_id}.bullets.{bullet_edit.id}"
            if not bullet_edit.text.strip():
                errors.append(f"{where}: rephrase must not be blank")
                continue
            if len(bullet_edit.text) > MAX_BULLET_CHARS:
                errors.append(
                    f"{where}: {len(bullet_edit.text)} chars exceeds {MAX_BULLET_CHARS}"
                )
            _check_rephrase(base_bullet.text, bullet_edit.text,
                            base_bullet.metrics, where, errors)

    if overlay.notes is not None and len(overlay.notes) > MAX_NOTES_CHARS:
        errors.append(f"notes: {len(overlay.notes)} chars exceeds {MAX_NOTES_CHARS}")

    return errors


def assert_valid_resume(resume: Resume) -> None:
    errors = validate_resume(resume)
    if errors:
        raise TailorError("invalid resume:\n  " + "\n  ".join(errors))


def assert_valid_overlay(base: Resume, overlay: Overlay) -> None:
    errors = validate_overlay(base, overlay)
    if errors:
        raise TailorError("invalid overlay:\n  " + "\n  ".join(errors))
