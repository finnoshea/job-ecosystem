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

#: Inline emphasis allowed in free text: ``**bold**`` and ``*italic*``.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"\*([^*]+?)\*")

#: SimpleDocTemplate wraps content in a Frame with 6pt padding on each side and
#: exposes no kwarg to change it. Widths must subtract it, or a full-width table
#: is wider than the frame and reportlab centers it -- shifting the row left of
#: the paragraphs, which reads as "everything after the title is indented".
_FRAME_PADDING = 6.0


def _rich(text: str | None) -> str:
    """Escape text for reportlab's mini-markup, then apply bold/italic.

    Bold is substituted first so its ``**`` is consumed before the italic
    pattern sees a single ``*``.
    """
    escaped = _BOLD.sub(r"<b>\1</b>", escape(text or ""))
    return _ITALIC.sub(r"<i>\1</i>", escaped)


def _dates(start: str | None, end: str | None) -> str:
    if start and end:
        return f"{start} – {end}"
    if start:
        return f"{start} – Present"
    return end or ""


def _sep(parts: list[str | None]) -> str:
    return " · ".join(part for part in parts if part)


def fit_one_line(text: str, max_width: float, font: str, size: float,
                 width_of, *, min_size: float = 6.5):
    """Fit ``text`` onto a single line no wider than ``max_width``.

    Returns ``(font_size, display_text, width)``. The font is shrunk toward
    ``min_size`` first; only if even that overflows is the text truncated with
    an ellipsis. ``width_of(text, font, size) -> float`` is injected so the
    function is testable without reportlab.
    """
    if width_of(text, font, size) <= max_width:
        return size, text, width_of(text, font, size)

    current = size
    while current > min_size and width_of(text, font, current) > max_width:
        current = max(min_size, current - 0.25)
    width = width_of(text, font, current)
    if width <= max_width:
        return current, text, width

    ellipsis = "\u2026"
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if width_of(text[:mid] + ellipsis, font, current) <= max_width:
            low = mid
        else:
            high = mid - 1
    shown = text[:low] + ellipsis
    return current, shown, width_of(shown, font, current)


def _reportlab():
    """Import the reportlab pieces the renderer needs, or explain the extra."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.lib.pagesizes import A4, LEGAL, LETTER
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            HRFlowable,
            KeepTogether,
            PageBreak,
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
        "colors": colors, "TA_CENTER": TA_CENTER, "stringWidth": stringWidth,
        "LETTER": LETTER, "LEGAL": LEGAL, "A4": A4, "inch": inch,
        "ParagraphStyle": ParagraphStyle, "HRFlowable": HRFlowable,
        "KeepTogether": KeepTogether, "Paragraph": Paragraph,
        "PageBreak": PageBreak, "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer, "Table": Table, "TableStyle": TableStyle,
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
    content_width = (
        page[0] - (layout.margin_left + layout.margin_right) * inch
        - 2 * _FRAME_PADDING
    )

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
        "meta": style("meta", font=layout.font, size=layout.meta_size,
                      color=muted_color, spaceAfter=2),
        "bullet": style("bullet", font=layout.font, size=layout.body_size,
                        color=text_color, spaceAfter=2),
        "publication": style("publication", font=layout.font, size=layout.body_size,
                             color=text_color, leftIndent=layout.bullet_indent,
                             firstLineIndent=-layout.bullet_indent, spaceAfter=3),
    }

    def p(markup: str, key: str):
        return Paragraph(markup, styles[key])

    def two_col_row(left_markup: str, right_text: str, *, left_style: str = "role"):
        # The right cell is a plain string, so reportlab draws it as one line:
        # a Paragraph would wrap a long URL. The column is sized to the fitted
        # text (shrunk, then truncated only if necessary), which keeps it on the
        # same line as the left cell and flush with the right margin.
        right_text = right_text or ""
        max_right = content_width * 0.62
        if right_text:
            right_size, shown, right_width = fit_one_line(
                right_text, max_right, layout.font, layout.meta_size,
                rl["stringWidth"],
            )
        else:
            right_size, shown, right_width = layout.meta_size, "", 0.0
        table = Table(
            [[Paragraph(left_markup, styles[left_style]), shown]],
            colWidths=[content_width - right_width, right_width],
        )
        table.hAlign = "LEFT"   # must not be centered away from the left margin
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("FONTNAME", (1, 0), (1, 0), layout.font),
            ("FONTSIZE", (1, 0), (1, 0), right_size),
            ("TEXTCOLOR", (1, 0), (1, 0), muted_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return table

    def header_row(left_markup: str, right_text: str):
        return two_col_row(left_markup, right_text, left_style="role")

    def bullet(text: str):
        # Flush left, with the marker inline: a real bullet glyph would be drawn
        # at the frame edge and the text indented past it, which is the indent
        # this removes. The line now starts at the same x as the role title.
        return Paragraph(f"\u2022 {_rich(text)}", styles["bullet"])

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
        # Contact details in email, phone, location order; links each on their
        # own line below, so a long URL cannot crowd the line you read first.
        contact = _sep([basics.email, basics.phone, basics.location])
        if contact:
            out.append(p(_rich(contact), "contact"))
        for link in basics.links:
            out.append(p(_rich(f"{link.label}: {link.url}"), "contact"))
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
            # Location right-justified on its own row, opposite the employment
            # type (usually blank), so it lines up with the dates above it.
            if role.location or role.employment_type:
                block.append(two_col_row(
                    _rich(role.employment_type or ""), role.location or "",
                    left_style="meta",
                ))
            if role.summary:
                block.append(p(_rich(role.summary), "body"))
            bullets = role.bullets
            if resume.render.bullets_per_role is not None:
                bullets = bullets[: resume.render.bullets_per_role]
            if bullets:
                block.append(Spacer(1, 4))       # gap after the summary
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
            p(_rich(_sep([
                cert.name, cert.issuer, cert.date,
                f"No. {cert.number}" if cert.number else None,
            ])), "body")
            for cert in resume.certifications
        ])

    def awards():
        if not resume.awards:
            return []
        return section("Awards", [
            p(_rich(_sep([award.name, award.issuer, award.date])), "body")
            for award in resume.awards
        ])

    def publications():
        if not resume.publications:
            return []
        return section("Selected Publications", [
            p(_rich(line), "publication") for line in resume.publications
        ])

    builders = {
        "summary": summary, "skills": skills, "roles": roles,
        "projects": projects, "education": education,
        "certifications": certifications, "awards": awards,
        "publications": publications,
    }

    flow: list[Any] = header()
    order = resume.render.section_order or list(builders)
    for name in order:
        builder = builders.get(name)
        if builder is None:
            continue
        chunk = builder()
        if not chunk:
            continue
        if name in resume.render.page_break_before:
            flow.append(rl["PageBreak"]())
        else:
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
