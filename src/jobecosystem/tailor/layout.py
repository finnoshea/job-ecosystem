"""The layout specification for the PDF renderer.

This is the single place to control how the resume looks: page size and
margins, fonts and sizes, colors, and spacing. The defaults here produce a clean
single-column resume; override any field programmatically or point
``TAILOR_LAYOUT_FILE`` at a JSON object with a subset of these keys.

Keeping it a plain dataclass (rather than CSS) is deliberate: the renderer is
reportlab, layout is Python data, and there is exactly one surface to edit.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .models import TailorError


@dataclass(slots=True)
class Layout:
    """Page and type settings. Lengths are inches (converted by the renderer)."""

    page: str = "LETTER"            # LETTER, LEGAL, or A4
    margin_top: float = 0.6
    margin_bottom: float = 0.6
    margin_left: float = 0.7
    margin_right: float = 0.7

    font: str = "Helvetica"
    font_bold: str = "Helvetica-Bold"

    name_size: float = 20
    headline_size: float = 11
    contact_size: float = 9.5
    section_size: float = 11
    body_size: float = 10.5
    meta_size: float = 9.5
    leading: float = 1.35           # multiple of font size

    text_color: str = "#1a1a1a"
    muted_color: str = "#555555"
    rule_color: str = "#c9c9c9"

    section_gap: float = 12         # space above each section, in points
    item_gap: float = 7             # space between roles/entries, in points
    bullet_indent: float = 12


DEFAULT_LAYOUT = Layout()


def resolve_layout_path(path: str | os.PathLike[str] | None = None) -> Path | None:
    """Explicit path > ``TAILOR_LAYOUT_FILE`` > None (use the defaults)."""
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("TAILOR_LAYOUT_FILE")
    return Path(configured).expanduser() if configured else None


def load_layout(path: str | os.PathLike[str] | None = None) -> Layout:
    """The default layout, with any JSON overrides applied.

    Unknown keys are rejected: a typo like ``margins_left`` should fail loudly
    rather than silently do nothing.
    """
    resolved = resolve_layout_path(path)
    if resolved is None:
        return Layout()
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise TailorError(f"layout not found: {resolved}") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise TailorError(f"{resolved}: not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise TailorError("layout: expected a JSON object")

    known = {f.name for f in fields(Layout)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise TailorError(f"layout: unknown key(s): {', '.join(unknown)}")

    layout = Layout()
    for key, value in data.items():
        setattr(layout, key, value)
    return layout


def layout_to_dict(layout: Layout) -> dict:
    """A serializable view of a layout, for editing or diffing."""
    return asdict(layout)
