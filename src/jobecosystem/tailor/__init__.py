"""Structured resume, declarative tailoring overlay, and PDF rendering.

See :mod:`jobecosystem.tailor.resume` (the base facts),
:mod:`jobecosystem.tailor.overlay` (the LLM's edit surface),
:mod:`jobecosystem.tailor.validate` (the rules), and
:mod:`jobecosystem.tailor.render` (layout).
"""

from .models import TailorError

__all__ = ["TailorError"]
