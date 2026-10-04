"""Typed carriers for the resume and its declarative tailoring overlay.

Plain dataclasses with no I/O and no validation, mirroring ``core/models.py``.
Conversion to and from JSON lives in :mod:`jobecosystem.tailor.resume` and
:mod:`jobecosystem.tailor.overlay`; the rules live in
:mod:`jobecosystem.tailor.validate`. Keeping the carriers dumb means an LLM edit
cannot change behaviour by smuggling in an unexpected type -- it either loads as
the declared shape or it is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class TailorError(Exception):
    """A resume/overlay could not be loaded, validated, or applied."""


# ---------------------------------------------------------------------------
# base resume
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Link:
    label: str
    url: str


@dataclass(slots=True)
class Bullet:
    id: str
    text: str
    tags: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    priority: int = 1


@dataclass(slots=True)
class Role:
    id: str
    company: str
    title: str
    employment_type: str | None = None
    location: str | None = None
    start: str | None = None
    end: str | None = None
    summary: str | None = None
    bullets: list[Bullet] = field(default_factory=list)


@dataclass(slots=True)
class Skill:
    id: str
    name: str
    category: str | None = None
    keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Project:
    id: str
    name: str
    url: str | None = None
    description: str | None = None
    bullets: list[Bullet] = field(default_factory=list)


@dataclass(slots=True)
class Education:
    id: str
    institution: str
    degree: str | None = None
    # JSON key is "field"; renamed here so it cannot shadow dataclasses.field
    # inside the class body.
    field_of_study: str | None = None
    location: str | None = None
    start: str | None = None
    end: str | None = None
    details: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Certification:
    id: str
    name: str
    issuer: str | None = None
    date: str | None = None
    number: str | None = None
    url: str | None = None


@dataclass(slots=True)
class Award:
    id: str
    name: str
    issuer: str | None = None
    date: str | None = None


@dataclass(slots=True)
class Basics:
    name: str
    headline: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[Link] = field(default_factory=list)
    summary: str | None = None


@dataclass(slots=True)
class RenderSpec:
    """Presentation hints only -- never content."""

    template: str = "default"
    section_order: list[str] = field(default_factory=lambda: [
        "summary", "roles", "education", "certifications", "projects",
        "awards", "skills", "publications",
    ])
    #: Sections that begin on a fresh page.
    page_break_before: list[str] = field(default_factory=lambda: ["publications"])
    max_roles: int | None = None
    bullets_per_role: int | None = None


@dataclass(slots=True)
class Resume:
    basics: Basics
    schema_version: int = 1
    skills: list[Skill] = field(default_factory=list)
    roles: list[Role] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)
    education: list[Education] = field(default_factory=list)
    certifications: list[Certification] = field(default_factory=list)
    awards: list[Award] = field(default_factory=list)
    #: Free-form citation lines. Authoring and formatting are the writer's job;
    #: the renderer only escapes and applies **bold** / *italic*, one line each.
    publications: list[str] = field(default_factory=list)
    render: RenderSpec = field(default_factory=RenderSpec)


# ---------------------------------------------------------------------------
# declarative overlay
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Target:
    job_id: int | None = None
    company: str | None = None
    title: str | None = None
    url: str | None = None


@dataclass(slots=True)
class BulletEdit:
    id: str
    text: str | None = None


@dataclass(slots=True)
class RoleEdit:
    bullets: list[BulletEdit] = field(default_factory=list)


@dataclass(slots=True)
class Overlay:
    """The LLM's edit surface: a desired end state, never new facts."""

    base_hash: str
    target: Target = field(default_factory=Target)
    schema_version: int = 1
    summary: str | None = None
    skill_ids: list[str] | None = None
    role_order: list[str] | None = None
    roles: dict[str, RoleEdit] = field(default_factory=dict)
    notes: str | None = None
