"""Turning a job into a proposed tailoring overlay.

This is the LLM boundary and the place where "don't do stupid things" is
enforced: the model never sees a task of "write my resume". It is asked to
produce the declarative overlay -- a selection/order/rephrase of ids that
already exist in the base -- and whatever it returns is parsed, pinned to the
current base, and validated before it is used. A bad edit cannot render.

The completion callable is injected (``complete(prompt) -> str``) so the adapter
is testable and provider-agnostic. The default reads a shell command from
``TAILOR_COMPLETE_COMMAND`` (prompt on stdin, completion on stdout), which works
with any local LLM CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from .models import Overlay, Resume, TailorError
from .overlay import overlay_from_dict
from .resume import canonical_json, resume_hash, resume_to_dict
from .validate import assert_valid_overlay

#: The overlay shape the model must produce, shown verbatim in the prompt.
_OVERLAY_SKELETON = {
    "summary": "optional replacement summary, or omit",
    "skill_ids": ["existing-skill-id"],
    "role_order": ["existing-role-id"],
    "roles": {
        "existing-role-id": {
            "bullets": [
                {"id": "existing-bullet-id"},
                {"id": "existing-bullet-id", "text": "optional rephrase"},
            ]
        }
    },
    "notes": "why these choices",
}

_RULES = """\
Rules (violating any of them invalidates the result):
- Reference only ids that appear in the base resume. You may not invent roles,
  employers, titles, dates, degrees, certifications, skills, or bullet ids.
- Every number in a bullet you rephrase must appear unchanged. Do NOT add,
  remove, or alter any number, percentage, dollar amount, or date.
- Your job is to select, order, and re-word the EXISTING content so it mirrors
  the job posting's language. Prefer bullets that overlap the posting.
- Only include a field when the base resume already has it. In particular, do
  not add a summary if the base has none -- a blank field stays blank.
- "role_order" is the complete ordered list of roles to show; omit a role to
  drop it. "roles[id].bullets" is that role's complete ordered bullet list;
  omit a role from "roles" to keep all of its bullets.
- Return one JSON object and nothing else. No prose, no code fences.
"""


def build_prompt(base: Resume, *, company: str, title: str,
                 description: str | None, job_id: int | None = None) -> str:
    """Assemble the prompt: the job, the base resume, and the overlay contract."""
    job_bits = [f"Title: {title}", f"Company: {company}"]
    if job_id is not None:
        job_bits.insert(0, f"Job id: {job_id}")
    job = "\n".join(job_bits)
    body = (description or "").strip()
    return (
        "You are tailoring a resume to a specific job posting. Produce a "
        "declarative overlay that selects, orders, and rephrases existing "
        "content -- never new facts.\n\n"
        f"{_RULES}\n"
        "The overlay object has this shape:\n"
        f"{canonical_json(_OVERLAY_SKELETON)}\n"
        f"=== JOB POSTING ===\n{job}\n\n{body}\n\n"
        "=== BASE RESUME (the only source of facts) ===\n"
        f"{canonical_json(resume_to_dict(base))}\n"
        "=== END ===\n"
        "Return only the overlay JSON object."
    )


def parse_overlay(text: str) -> dict[str, Any]:
    """Extract the overlay object from a completion, tolerating code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise TailorError("completion contained no JSON object")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as error:
        raise TailorError(f"completion was not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise TailorError("completion JSON was not an object")
    return value


def propose_overlay(
    base: Resume,
    *,
    company: str,
    title: str,
    description: str | None = None,
    job_id: int | None = None,
    complete,
) -> Overlay:
    """Ask the model for an overlay and return it only if it validates.

    ``base_hash``, ``schema_version``, and ``target`` are filled in here rather
    than trusted from the model, so those bookkeeping fields cannot be wrong.
    """
    prompt = build_prompt(base, company=company, title=title,
                          description=description, job_id=job_id)
    raw = complete(prompt)

    data = parse_overlay(raw)
    data["schema_version"] = 1
    data["base_hash"] = resume_hash(base)
    target = data.get("target")
    if not isinstance(target, dict):
        target = {}
    target.setdefault("job_id", job_id)
    target.setdefault("company", company)
    target.setdefault("title", title)
    data["target"] = target

    overlay = overlay_from_dict(data)
    assert_valid_overlay(base, overlay)
    return overlay


def default_completer():
    """A completer that shells out to ``$TAILOR_COMPLETE_COMMAND``.

    The command receives the prompt on stdin and must write the completion to
    stdout -- the same contract as most local LLM CLIs. Raises when unset so the
    CLI can explain how to configure one.
    """
    command = os.environ.get("TAILOR_COMPLETE_COMMAND")
    if not command:
        raise TailorError(
            "set TAILOR_COMPLETE_COMMAND to an LLM command that reads the prompt"
            " on stdin and writes JSON on stdout"
        )

    def complete(prompt: str) -> str:
        result = subprocess.run(
            command, shell=True, input=prompt, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise TailorError(
                f"TAILOR_COMPLETE_COMMAND failed ({result.returncode}):"
                f" {result.stderr.strip()}"
            )
        return result.stdout

    return complete
