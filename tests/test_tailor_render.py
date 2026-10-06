"""Tests for jobecosystem.tailor.render (reportlab PDF)."""

from __future__ import annotations

import pytest
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Paragraph, Table

from jobecosystem.tailor import layout as L
from jobecosystem.tailor import render as RD
from jobecosystem.tailor.models import TailorError


def paragraphs(flowables):
    """Every Paragraph in a flowable tree, descending into tables/groups."""
    found = []
    for flowable in flowables:
        if isinstance(flowable, Paragraph):
            found.append(flowable)
        elif isinstance(flowable, Table):
            for row in flowable._cellvalues:
                for cell in row:
                    if isinstance(cell, Paragraph):
                        found.append(cell)
        else:
            content = getattr(flowable, "_content", None)
            if content:
                found.extend(paragraphs(content))
    return found


def tables(flowables):
    """Every Table in a flowable tree, descending into KeepTogether groups."""
    found = []
    for flowable in flowables:
        if isinstance(flowable, Table):
            found.append(flowable)
        else:
            content = getattr(flowable, "_content", None)
            if content:
                found.extend(tables(content))
    return found


def texts(resume) -> list[str]:
    return [p.text for p in paragraphs(RD.build_flowables(resume))]


def test_flowables_include_the_header(sample_resume):
    rendered = texts(sample_resume)
    assert any("Test Person" in t for t in rendered)
    assert any("Backend Engineer" in t for t in rendered)


def test_header_orders_contact_and_puts_links_on_one_line(sample_resume):
    rendered = texts(sample_resume)
    assert "test@example.com · +1-512-555-0000 · Remote" in rendered

    link_lines = [t for t in rendered if "GitHub:" in t]
    assert len(link_lines) == 1
    assert "LinkedIn: https://linkedin.com/in/tp" in link_lines[0]

    contact = next(t for t in rendered if t.startswith("test@example.com"))
    assert "GitHub" not in contact and "LinkedIn" not in contact


def test_section_order_is_respected(sample_resume):
    sample_resume.render.section_order = ["roles", "summary", "skills"]
    rendered = texts(sample_resume)
    assert rendered.index("EXPERIENCE") < rendered.index("SUMMARY")
    assert rendered.index("SUMMARY") < rendered.index("SKILLS")


def test_a_section_left_out_is_not_rendered(sample_resume):
    sample_resume.render.section_order = ["summary"]
    rendered = texts(sample_resume)
    assert "EXPERIENCE" not in rendered
    assert "SKILLS" not in rendered


def test_text_is_escaped_for_markup(sample_resume):
    sample_resume.roles[0].bullets[0].text = "Danger <script>alert(1)</script> 40%"
    rendered = texts(sample_resume)
    assert any("&lt;script&gt;" in t for t in rendered)
    assert not any("<script>" in t for t in rendered)


def test_bold_markup_is_applied(sample_resume):
    sample_resume.roles[0].bullets[0].text = "Use **Python** for 40% of it."
    assert any("<b>Python</b>" in t for t in texts(sample_resume))


def test_max_roles_caps_the_list(sample_resume):
    sample_resume.render.max_roles = 1
    rendered = texts(sample_resume)
    assert any("One Corp" in t for t in rendered)
    assert not any("Two Inc" in t for t in rendered)


def test_bullets_per_role_caps_the_list(sample_resume):
    sample_resume.render.bullets_per_role = 1
    rendered = texts(sample_resume)
    assert any("Cut latency 40%" in t for t in rendered)
    assert not any("Led 5 engineers" in t for t in rendered)


def test_publications_render_last_and_on_a_new_page(sample_resume):
    from reportlab.platypus import PageBreak

    flow = RD.build_flowables(sample_resume)
    headings = [f.text for f in flow
                if isinstance(f, Paragraph) and f.style.name == "section"]
    assert headings[-1] == "SELECTED PUBLICATIONS"
    assert "SKILLS" in headings

    break_at = next(i for i, f in enumerate(flow) if isinstance(f, PageBreak))
    heading_at = next(i for i, f in enumerate(flow)
                      if isinstance(f, Paragraph)
                      and f.text == "SELECTED PUBLICATIONS")
    assert break_at < heading_at


def test_project_url_stays_on_one_line(sample_resume):
    table = next(
        t for t in tables(RD.build_flowables(sample_resume))
        if any(cell == "https://github.com/tp/job-ecosystem"
               for row in t._cellvalues for cell in row)
    )
    left, right = table._cellvalues[0]
    assert isinstance(right, str)                 # plain string => no wrapping
    assert right == "https://github.com/tp/job-ecosystem"
    assert "Job Ecosystem" in left.text


def test_fit_one_line_keeps_text_that_already_fits():
    size, shown, width = RD.fit_one_line("short", 200, "Helvetica", 9.5, stringWidth)
    assert (shown, size) == ("short", 9.5)
    assert width <= 200


def test_fit_one_line_shrinks_before_truncating():
    text = "https://github.com/someone/a-fairly-long-repository-name"
    size, shown, width = RD.fit_one_line(text, 200, "Helvetica", 9.5, stringWidth)
    assert shown == text and size < 9.5
    assert width <= 200


def test_fit_one_line_truncates_when_the_floor_is_still_too_wide():
    size, shown, width = RD.fit_one_line("x" * 400, 80, "Helvetica", 9.5,
                                         stringWidth, min_size=9.5)
    assert shown.endswith("\u2026")
    assert width <= 80


def test_role_bullets_have_three_points_of_space(sample_resume):
    bullets = [p for p in paragraphs(RD.build_flowables(sample_resume))
               if p.style.name == "bullet"]
    assert bullets
    assert all(p.style.spaceAfter == 3 for p in bullets)


def test_role_location_is_a_right_aligned_cell(sample_resume):
    row = next(t for t in tables(RD.build_flowables(sample_resume))
               if any(cell == "Seattle, WA" for r in t._cellvalues for cell in r))
    assert row._cellvalues[0][1] == "Seattle, WA"


def test_role_bullets_are_not_indented(sample_resume):
    bullets = [p for p in paragraphs(RD.build_flowables(sample_resume))
               if p.style.name == "bullet"]
    assert bullets
    assert all(p.style.leftIndent == 0 for p in bullets)
    assert all(p.text.startswith("\u2022 ") for p in bullets)


def test_certifications_render_the_number(sample_resume):
    assert any("No. CERT-123" in t for t in texts(sample_resume))


def test_publications_render_one_formatted_line_each(sample_resume):
    sample_resume.publications = [
        "**A Title** — A. Author, *Journal* 7, 035057 (2026).",
        "Second **Line** with *italics*.",
    ]
    rendered = texts(sample_resume)
    assert any("<b>A Title</b>" in t and "<i>Journal</i>" in t for t in rendered)
    assert any("<b>Line</b>" in t for t in rendered)


def test_no_page_break_without_publications(sample_resume):
    from reportlab.platypus import PageBreak

    sample_resume.publications = []
    flow = RD.build_flowables(sample_resume)
    assert not any(isinstance(f, PageBreak) for f in flow)


def test_render_writes_a_pdf(sample_resume, tmp_path):
    out = RD.render(sample_resume, tmp_path / "sub" / "resume.pdf")
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 1000


def test_render_accepts_a_layout(sample_resume, tmp_path):
    out = RD.render(sample_resume, tmp_path / "a4.pdf", layout=L.Layout(page="A4"))
    assert out.read_bytes().startswith(b"%PDF")


def test_render_rejects_an_unknown_page(sample_resume, tmp_path):
    bad = L.Layout()
    bad.page = "TABLOID"
    with pytest.raises(TailorError, match="page size"):
        RD.render(sample_resume, tmp_path / "x.pdf", layout=bad)
