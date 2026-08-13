"""Fixed rules for building a new question when no reference match is found.

Unlike a user-editable template, these rules are hard-coded: the new
question's ID is derived deterministically from a combination of the
source question's section, topic, and answer type. The question's own
section and text are kept unchanged.

To change how new questions are built, edit this module directly.
"""
import re
from typing import Optional, Tuple

from models import Question

_CODE_LENGTH = 4


def _slugify(value: Optional[str], fallback: str) -> str:
    """Turn a free-text value into a short uppercase code for IDs."""
    if not value or not value.strip():
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value)
    return (cleaned[:_CODE_LENGTH] or fallback).upper()


def infer_answer_type(options: str) -> str:
    """Infer a coarse answer type from the raw options text.

    Used as a fallback when the source CSV doesn't have a dedicated
    answer-type column.
    """
    options = (options or "").strip()
    if not options:
        return "open"

    parts = [p.strip().lower() for p in options.split(";") if p.strip()]
    if len(parts) == 2 and set(parts) <= {"yes", "no"}:
        return "boolean"
    if len(parts) > 1:
        return "choice"
    if parts and parts[0] in {"number", "numeric"}:
        return "numeric"
    return "open"


def build_new_question(source: Question, sequence: int) -> Tuple[str, str, str]:
    """Build the (id, section, text) for a new question using fixed rules.

    The ID combines short codes for section and answer type plus a running
    sequence number, e.g. ``NEW-DEMO-RADI-001``. Section and text are
    carried over from the source question as-is.
    """
    section_code = _slugify(source.section, fallback="SEC")
    answer_type = source.answer_type or infer_answer_type(source.options)
    answer_code = _slugify(answer_type, fallback="OPEN")

    new_id = f"NEW-{section_code}-{answer_code}-{sequence:03d}"
    new_section = source.section
    new_text = source.question
    return new_id, new_section, new_text
