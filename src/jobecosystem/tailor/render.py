"""Render the resume to PDF with reportlab.

Pure Python: reportlab has no system-library dependencies, unlike the
HTML-to-PDF tools, so this runs anywhere the venv does. The layout is entirely
:mod:`jobecosystem.tailor.layout` (page size, margins, fonts, sizes, colors,
spacing), so there is exactly one surface to control the look.

``build_flowables`` is public so the ordered content can be inspected in tests
without parsing a PDF.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from .layout import Layout, load_layout
from .models import Resume, TailorError

#: Inline emphasis allowed in text; only ``**bold**``.
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _rich(text: str | None) -> str:
    """Escape text for reportlab's mini-markup, then apply ``**bold**``."""
    return _BOLD.sub(r"<b>\1</b>", escape(text or ""))


def _dates(start: str | None, end: str | None) -> str:
    if start and end:
        return f"{start} – {end}"
    if start:
        return f"{start} – Present"
    return end or ""


def _sep(parts: list[str | None]) -> str:
    return " · ".join(part for part in parts if part)


def _reportlab():
    """Import the reportlab pieces the renderer needs, or explain the extra."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        from reportlab.lib.pagesizes import A4, LEGAL, LETTER
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            HRFlowable,
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as error:  # pragma: no cover - environment problem
        raise TailorError(
            "reportlab is not installed; install the pdf extra:"
            " pip install -e '.[pdf]'"
        ) from error
    return {
        "colors": colors, "TA_CENTER": TA_CENTER, "TA_RIGHT": TA_RIGHT,
        "LETTER": LETTER, "LEGAL": LEGAL, "A4": A4, "inch": inch,
        "ParagraphStyle": ParagraphStyle, "HRFlowable": HRFlowable,
        "KeepTogether": KeepTogether, "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate, "Spacer": Spacer,
        "Table": Table, "TableStyle": TableStyle,
    }


def build_flowables(resume: Resume, layout: Layout | None = None) -> list[Any]:
    """The ordered reportlab flowables for a resume (no file written)."""
    rl = _reportlab()
    layout = layout or load_layout()
    Paragraph, Spacer, Table, TableStyle = (
        rl["Paragraph"], rl["Spacer"], rl["Table"], rl["TableStyle"]
    )
    colors, inch = rl["colors"], rl["inch"]

    page = {"LETTER": rl["LETTER"], "LEGAL": rl["LEGAL"], "A4": rl["A4"]}.get(
        layout.page.upper()
    )
    if page is None:
        raise TailorError(f"layout.page: unknown page size {layout.page!r}")
    content_width = page[0] - (layout.margin_left + layout.margin_right) * inch

    text_color = colors.HexColor(layout.text_color)
    muted_color = colors.HexColor(layout.muted_color)
    rule_color = colors.HexColor(layout.rule_color)

    def style(name: str, *, font: str, size: float, color, **extra):
        return rl["ParagraphStyle"](
            name, fontName=font, fontSize=size, leading=size * layout.leading,
            textColor=color, **extra,
        )

    styles = {
        "name": style("name", font=layout.font_bold, size=layout.name_size,
                      color=text_color, alignment=rl["TA_CENTER"], spaceAfter=1),
        "headline": style("headline", font=layout.font, size=layout.headline_size,
                          color=muted_color, alignment=rl["TA_CENTER"], spaceAfter=1),
        "contact": style("contact", font=layout.font, size=layout.contact_size,
                         color=muted_color, alignment=rl["TA_CENTER"], spaceAfter=2),
        "section": style("section", font=layout.font_bold, size=layout.section_size,
                         color=text_color, spaceAfter=2),
        "body": style("body", font=layout.font, size=layout.body_size,
                      color=text_color, spaceAfter=2),
        "role": style("role", font=layout.font_bold, size=layout.body_size,
                      color=text_color),
        "dates": style("dates", font=layout.font, size=layout.meta_size,
                       color=muted_color, alignment=rl["TA_RIGHT"]),
        "meta": style("meta", font=layout.font, size=layout.meta_size,
                      color=muted_color, spaceAfter=2),
        "bullet": style("bullet", font=layout.font, size=layout.body_size,
                        color=text_color, leftIndent=layout.bullet_indent,
                        bulletIndent=0, spaceAfter=1),
    }

    def p(markup: str, key: str):
        return Paragraph(markup, styles[key])

    def header_row(left_markup: str, right_text: str):
        table = Table(
            [[Paragraph(left_markup, styles["role"]),
              Paragraph(_rich(right_text) if right_text else "", styles["dates"])]],
            colWidths=[content_width - 92, 92],
        )
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return table

    def bullet(text: str):
        return Paragraph(_rich(text), styles["bullet"], bulletText="\u2022")

    def section(title: str, body: list):
        return [
            p(title.upper(), "section"),
            rl["HRFlowable"](width=content_width, thickness=0.6,
                             color=rule_color, spaceBefore=1, spaceAfter=4),
            *body,
        ]

    def header():
        basics = resume.basics
        out = [p(_rich(basics.name), "name")]
        if basics.headline:
            out.append(p(_rich(basics.headline), "headline"))
        contact = _sep([
            basics.location, basics.email, basics.phone,
            *[f"{link.label}: {link.url}" for link in basics.links],
        ])
        if contact:
            out.append(p(_rich(contact), "contact"))
        return out

    def summary():
        if not resume.basics.summary:
            return []
        return section("Summary", [p(_rich(resume.basics.summary), "body")])

    def skills():
        if not resume.skills:
            return []
        groups: dict[str | None, list[str]] = {}
        for skill in resume.skills:
            groups.setdefault(skill.category, []).append(skill.name)
        body = []
        for category, names in groups.items():
            joined = escape(", ".join(names))
            body.append(p(f"<b>{escape(category)}</b> {joined}", "body")
                        if category else p(joined, "body"))
        return section("Skills", body)

    def roles():
        items = resume.roles
        if resume.render.max_roles is not None:
            items = items[: resume.render.max_roles]
        if not items:
            return []
        body = []
        for role in items:
            left = f"<b>{escape(role.title)}</b>"
            if role.company:
                left += f" — {escape(role.company)}"
            block = [header_row(left, _dates(role.start, role.end))]
            meta = _sep([role.location, role.employment_type])
            if meta:
                block.append(p(_rich(meta), "meta"))
            if role.summary:
                block.append(p(_rich(role.summary), "body"))
            bullets = role.bullets
            if resume.render.bullets_per_role is not None:
                bullets = bullets[: resume.render.bullets_per_role]
            block.extend(bullet(item.text) for item in bullets)
            block.append(Spacer(1, layout.item_gap))
            body.append(rl["KeepTogether"](block))
        return section("Experience", body)

    def projects():
        if not resume.projects:
            return []
        body = []
        for project in resume.projects:
            block = [header_row(f"<b>{escape(project.name)}</b>", project.url or "")]
            if project.description:
                block.append(p(_rich(project.description), "body"))
            block.extend(bullet(item.text) for item in project.bullets)
            block.append(Spacer(1, layout.item_gap))
            body.append(rl["KeepTogether"](block))
        return section("Projects", body)

    def education():
        if not resume.education:
            return []
        body = []
        for item in resume.education:
            degree = " ".join(part for part in [item.degree, item.field_of_study] if part)
            left = f"<b>{escape(item.institution)}</b>"
            if degree:
                left += f" — {escape(degree)}"
            block = [header_row(left, _dates(item.start, item.end))]
            block.extend(bullet(detail) for detail in item.details)
            block.append(Spacer(1, layout.item_gap))
            body.append(rl["KeepTogether"](block))
        return section("Education", body)

    def certifications():
        if not resume.certifications:
            return []
        return section("Certifications", [
            p(_rich(_sep([cert.name, cert.issuer, cert.date])), "body")
            for cert in resume.certifications
        ])

    def awards():
        if not resume.awards:
            return []
        return section("Awards", [
            p(_rich(_sep([award.name, award.issuer, award.date])), "body")
            for award in resume.awards
        ])

    builders = {
        "summary": summary, "skills": skills, "roles": roles,
        "projects": projects, "education": education,
        "certifications": certifications, "awards": awards,
    }

    flow: list[Any] = header()
    order = resume.render.section_order or list(builders)
    for name in order:
        builder = builders.get(name)
        if builder is None:
            continue
        chunk = builder()
        if chunk:
            flow.append(Spacer(1, layout.section_gap))
            flow.extend(chunk)
    return flow


def render(resume: Resume, path: str | Path, *, layout: Layout | None = None) -> Path:
    """Write the resume to ``path`` as a PDF; returns the path written."""
    rl = _reportlab()
    layout = layout or load_layout()
    page = {"LETTER": rl["LETTER"], "LEGAL": rl["LEGAL"], "A4": rl["A4"]}.get(
        layout.page.upper()
    )
    if page is None:
        raise TailorError(f"layout.page: unknown page size {layout.page!r}")

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    inch = rl["inch"]
    document = rl["SimpleDocTemplate"](
        str(out), pagesize=page,
        leftMargin=layout.margin_left * inch, rightMargin=layout.margin_right * inch,
        topMargin=layout.margin_top * inch, bottomMargin=layout.margin_bottom * inch,
        title=resume.basics.name, author=resume.basics.name,
    )
    document.build(build_flowables(resume, layout))
    return out
