"""Loading, serializing, and hashing the base resume.

The base resume is a single JSON file at the repo root (``resume.base.json``),
resolved like the scraper config files: an explicit path, then
``TAILOR_RESUME_FILE``, then the repo root. It is the source of truth for facts;
a tailoring overlay may only reference it, never extend it.

Serialization is canonical -- stable key order and two-space indent -- so a git
diff shows only what actually changed, and the content hash used to pin
overlays is stable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .models import (
    Award,
    Basics,
    Bullet,
    Certification,
    Education,
    Link,
    Project,
    RenderSpec,
    Resume,
    Role,
    Skill,
    TailorError,
)

#: src/jobecosystem/tailor/resume.py -> tailor -> jobecosystem -> src -> root
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESUME_FILE = _REPO_ROOT / "resume.base.json"


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------

def resolve_resume_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Explicit path > ``TAILOR_RESUME_FILE`` > ``resume.base.json`` at the root."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("TAILOR_RESUME_FILE")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_RESUME_FILE


# ---------------------------------------------------------------------------
# coercion helpers
# ---------------------------------------------------------------------------

def _as_dict(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise TailorError(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _as_list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise TailorError(f"{where}: expected a list, got {type(value).__name__}")
    return value


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TailorError(f"{where}: expected a non-empty string")
    return value.strip()


def _opt_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TailorError(f"{where}: expected a string or null")
    return value.strip() or None


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TailorError(f"{where}: expected an integer")
    return value


def _opt_int(value: Any, where: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, where)


def _str_list(value: Any, where: str) -> list[str]:
    return [_as_str(item, f"{where}[{index}]") for index, item in enumerate(
        _as_list(value, where)
    )]


# ---------------------------------------------------------------------------
# JSON -> dataclasses
# ---------------------------------------------------------------------------

def _link(value: Any, where: str) -> Link:
    data = _as_dict(value, where)
    return Link(label=_as_str(data.get("label"), f"{where}.label"),
                url=_as_str(data.get("url"), f"{where}.url"))


def _bullet(value: Any, where: str) -> Bullet:
    data = _as_dict(value, where)
    return Bullet(
        id=_as_str(data.get("id"), f"{where}.id"),
        text=_as_str(data.get("text"), f"{where}.text"),
        tags=_str_list(data.get("tags", []), f"{where}.tags"),
        metrics=_str_list(data.get("metrics", []), f"{where}.metrics"),
        technologies=_str_list(data.get("technologies", []), f"{where}.technologies"),
        priority=_as_int(data.get("priority", 1), f"{where}.priority"),
    )


def _role(value: Any, where: str) -> Role:
    data = _as_dict(value, where)
    return Role(
        id=_as_str(data.get("id"), f"{where}.id"),
        company=_as_str(data.get("company"), f"{where}.company"),
        title=_as_str(data.get("title"), f"{where}.title"),
        employment_type=_opt_str(data.get("employment_type"), f"{where}.employment_type"),
        location=_opt_str(data.get("location"), f"{where}.location"),
        start=_opt_str(data.get("start"), f"{where}.start"),
        end=_opt_str(data.get("end"), f"{where}.end"),
        summary=_opt_str(data.get("summary"), f"{where}.summary"),
        bullets=[_bullet(item, f"{where}.bullets[{i}]")
                 for i, item in enumerate(_as_list(data.get("bullets", []), f"{where}.bullets"))],
    )


def _skill(value: Any, where: str) -> Skill:
    data = _as_dict(value, where)
    return Skill(id=_as_str(data.get("id"), f"{where}.id"),
                 name=_as_str(data.get("name"), f"{where}.name"),
                 category=_opt_str(data.get("category"), f"{where}.category"),
                 keywords=_str_list(data.get("keywords", []), f"{where}.keywords"))


def _project(value: Any, where: str) -> Project:
    data = _as_dict(value, where)
    return Project(
        id=_as_str(data.get("id"), f"{where}.id"),
        name=_as_str(data.get("name"), f"{where}.name"),
        url=_opt_str(data.get("url"), f"{where}.url"),
        description=_opt_str(data.get("description"), f"{where}.description"),
        bullets=[_bullet(item, f"{where}.bullets[{i}]")
                 for i, item in enumerate(_as_list(data.get("bullets", []), f"{where}.bullets"))],
    )


def _education(value: Any, where: str) -> Education:
    data = _as_dict(value, where)
    return Education(
        id=_as_str(data.get("id"), f"{where}.id"),
        institution=_as_str(data.get("institution"), f"{where}.institution"),
        degree=_opt_str(data.get("degree"), f"{where}.degree"),
        field_of_study=_opt_str(data.get("field"), f"{where}.field"),
        location=_opt_str(data.get("location"), f"{where}.location"),
        start=_opt_str(data.get("start"), f"{where}.start"),
        end=_opt_str(data.get("end"), f"{where}.end"),
        details=_str_list(data.get("details", []), f"{where}.details"),
    )


def _certification(value: Any, where: str) -> Certification:
    data = _as_dict(value, where)
    return Certification(id=_as_str(data.get("id"), f"{where}.id"),
                         name=_as_str(data.get("name"), f"{where}.name"),
                         issuer=_opt_str(data.get("issuer"), f"{where}.issuer"),
                         date=_opt_str(data.get("date"), f"{where}.date"),
                         url=_opt_str(data.get("url"), f"{where}.url"))


def _award(value: Any, where: str) -> Award:
    data = _as_dict(value, where)
    return Award(id=_as_str(data.get("id"), f"{where}.id"),
                 name=_as_str(data.get("name"), f"{where}.name"),
                 issuer=_opt_str(data.get("issuer"), f"{where}.issuer"),
                 date=_opt_str(data.get("date"), f"{where}.date"))


def _basics(value: Any) -> Basics:
    data = _as_dict(value, "basics")
    return Basics(
        name=_as_str(data.get("name"), "basics.name"),
        headline=_opt_str(data.get("headline"), "basics.headline"),
        email=_opt_str(data.get("email"), "basics.email"),
        phone=_opt_str(data.get("phone"), "basics.phone"),
        location=_opt_str(data.get("location"), "basics.location"),
        links=[_link(item, f"basics.links[{i}]")
               for i, item in enumerate(_as_list(data.get("links", []), "basics.links"))],
        summary=_opt_str(data.get("summary"), "basics.summary"),
    )


def _render(value: Any) -> RenderSpec:
    if value is None:
        return RenderSpec()
    data = _as_dict(value, "render")
    default = RenderSpec()
    return RenderSpec(
        template=_opt_str(data.get("template"), "render.template") or default.template,
        section_order=_str_list(data.get("section_order", default.section_order),
                                "render.section_order"),
        max_roles=_opt_int(data.get("max_roles"), "render.max_roles"),
        bullets_per_role=_opt_int(data.get("bullets_per_role"), "render.bullets_per_role"),
    )


def resume_from_dict(value: Any) -> Resume:
    """Build a :class:`Resume` from parsed JSON, or raise :class:`TailorError`."""
    data = _as_dict(value, "resume")
    return Resume(
        basics=_basics(data.get("basics")),
        schema_version=_as_int(data.get("schema_version", 1), "schema_version"),
        skills=[_skill(x, f"skills[{i}]")
                for i, x in enumerate(_as_list(data.get("skills", []), "skills"))],
        roles=[_role(x, f"roles[{i}]")
               for i, x in enumerate(_as_list(data.get("roles", []), "roles"))],
        projects=[_project(x, f"projects[{i}]")
                  for i, x in enumerate(_as_list(data.get("projects", []), "projects"))],
        education=[_education(x, f"education[{i}]")
                   for i, x in enumerate(_as_list(data.get("education", []), "education"))],
        certifications=[_certification(x, f"certifications[{i}]") for i, x in enumerate(
            _as_list(data.get("certifications", []), "certifications"))],
        awards=[_award(x, f"awards[{i}]")
                for i, x in enumerate(_as_list(data.get("awards", []), "awards"))],
        render=_render(data.get("render")),
    )


# ---------------------------------------------------------------------------
# dataclasses -> JSON
# ---------------------------------------------------------------------------

def _drop_none(data: dict) -> dict:
    return {key: value for key, value in data.items() if value is not None}


def _bullet_dict(bullet: Bullet) -> dict:
    return {"id": bullet.id, "text": bullet.text, "tags": bullet.tags,
            "metrics": bullet.metrics, "technologies": bullet.technologies,
            "priority": bullet.priority}


def resume_to_dict(resume: Resume) -> dict:
    """The JSON shape of a resume, with ``None`` optionals omitted."""
    return {
        "schema_version": resume.schema_version,
        "basics": {
            **_drop_none({
                "name": resume.basics.name,
                "headline": resume.basics.headline,
                "email": resume.basics.email,
                "phone": resume.basics.phone,
                "location": resume.basics.location,
                "summary": resume.basics.summary,
            }),
            "links": [{"label": link.label, "url": link.url}
                      for link in resume.basics.links],
        },
        "skills": [_drop_none({"id": s.id, "name": s.name, "category": s.category,
                               "keywords": s.keywords}) for s in resume.skills],
        "roles": [
            {**_drop_none({
                "id": role.id, "company": role.company, "title": role.title,
                "employment_type": role.employment_type, "location": role.location,
                "start": role.start, "end": role.end, "summary": role.summary,
            }),
             "bullets": [_bullet_dict(b) for b in role.bullets]}
            for role in resume.roles
        ],
        "projects": [
            {**_drop_none({"id": p.id, "name": p.name, "url": p.url,
                           "description": p.description}),
             "bullets": [_bullet_dict(b) for b in p.bullets]}
            for p in resume.projects
        ],
        "education": [
            _drop_none({"id": e.id, "institution": e.institution, "degree": e.degree,
                        "field": e.field_of_study, "location": e.location,
                        "start": e.start, "end": e.end, "details": e.details})
            for e in resume.education
        ],
        "certifications": [
            _drop_none({"id": c.id, "name": c.name, "issuer": c.issuer,
                        "date": c.date, "url": c.url})
            for c in resume.certifications
        ],
        "awards": [
            _drop_none({"id": a.id, "name": a.name, "issuer": a.issuer, "date": a.date})
            for a in resume.awards
        ],
        "render": _drop_none({
            "template": resume.render.template,
            "section_order": resume.render.section_order,
            "max_roles": resume.render.max_roles,
            "bullets_per_role": resume.render.bullets_per_role,
        }),
    }


# ---------------------------------------------------------------------------
# load / save / hash
# ---------------------------------------------------------------------------

def load(path: str | os.PathLike[str] | None = None) -> Resume:
    """Load and parse the base resume from disk."""
    resume_file = resolve_resume_path(path)
    try:
        text = resume_file.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise TailorError(f"resume not found: {resume_file}") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise TailorError(f"{resume_file}: not valid JSON: {error}") from error
    return resume_from_dict(data)


def canonical_json(value: Any) -> str:
    """Stable JSON text: sorted keys, two-space indent, trailing newline."""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def save(resume: Resume, path: str | os.PathLike[str] | None = None) -> Path:
    """Write the resume canonically; returns the path written."""
    resume_file = resolve_resume_path(path)
    resume_file.write_text(canonical_json(resume_to_dict(resume)), encoding="utf-8")
    return resume_file


def resume_hash(resume: Resume) -> str:
    """Content hash used to pin an overlay to the base it was written against."""
    digest = hashlib.sha256(canonical_json(resume_to_dict(resume)).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def query_text(resume: Resume) -> str:
    """Text used to embed the resume as a search *query*.

    Headline, summary, skills, and every role/bullet, joined plainly. This is
    the whole resume on purpose: Voyage handles long input, and a narrower
    digest would silently drop signal.
    """
    lines: list[str] = []
    basics = resume.basics
    for value in (basics.headline, basics.location, basics.summary):
        if value:
            lines.append(value)
    for skill in resume.skills:
        pieces = [skill.name, *skill.keywords]
        if skill.category:
            pieces.append(skill.category)
        lines.append(", ".join(pieces))
    for role in resume.roles:
        lines.append(f"{role.title} at {role.company}")
        if role.summary:
            lines.append(role.summary)
        lines.extend(bullet.text for bullet in role.bullets)
    for project in resume.projects:
        lines.append(project.name)
        if project.description:
            lines.append(project.description)
        lines.extend(bullet.text for bullet in project.bullets)
    return "\n".join(lines)
