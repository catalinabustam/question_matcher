"""Save and restore an in-progress matching session as a JSON file.

`st.session_state` only survives the current browser tab/connection — a
page refresh starts a brand new Streamlit session, losing every decision
made so far. This module lets the user download a small JSON snapshot and
re-upload it later to pick up where they left off.

The file only stores what can't be cheaply recomputed: each question's
decision (status, matched ARC rows by position, new-question fields, field
overrides) and where the user left off. The source CSV and column mapping
must be re-supplied when resuming, exactly like starting fresh — that's what
lets `source_questions` and `reference_questions` come out identical to how
they were built when the file was saved, so decisions can be re-attached to
them positionally (source) and by `row_index` (matched reference rows).
"""
import json
from typing import Any

from models import MatchDecision, Question, StandaloneQuestion

PROGRESS_VERSION = 1


def build_progress_dict(
    decisions: list[MatchDecision],
    current_idx: int,
    reference_row_count: int,
    source_filename: str = "",
    standalone_questions: list[StandaloneQuestion] | None = None,
) -> dict[str, Any]:
    """Everything needed to resume later, ready to `json.dumps`.

    `reference_row_count` and `source_filename` are stored purely as sanity
    checks at resume time (see `restore_decisions` callers in `app.py`) —
    neither affects restoration itself.
    """
    return {
        "version": PROGRESS_VERSION,
        "current_idx": current_idx,
        "reference_row_count": reference_row_count,
        "source_filename": source_filename,
        "decisions": [d.to_dict() for d in decisions],
        "standalone_questions": [
            question.to_dict() for question in (standalone_questions or [])
        ],
    }


def peek_source_filename(raw: bytes) -> str:
    """Best-effort read of `source_filename` from a progress file, without
    the strict version check `load_progress_dict` does.

    Used to warn the user as soon as they upload a progress file whose
    source CSV doesn't match the one they've already uploaded — before
    they've even clicked "Resume from saved progress". Returns "" for
    anything unreadable (invalid JSON, old file with no such field, ...),
    since this is only ever used for an early hint, not for validation.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return data.get("source_filename", "") if isinstance(data, dict) else ""


def load_progress_dict(raw: bytes) -> dict[str, Any]:
    """Parse a progress file previously produced by `build_progress_dict`."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Not a valid progress file (invalid JSON).") from exc
    if data.get("version") != PROGRESS_VERSION:
        raise ValueError(
            "This progress file was made by an incompatible app version."
        )
    return data


def restore_decisions(
    progress: dict[str, Any],
    source_questions: list[Question],
    reference_questions: list[Question],
) -> tuple[list[MatchDecision], int, list[StandaloneQuestion]]:
    """Rebuild `decisions`, `current_idx`, and standalone questions from progress.

    `source_questions` / `reference_questions` must be built the same way
    they were when the file was saved (same source CSV + column mapping,
    same reference index) — decisions are re-attached to `source_questions`
    positionally, and each matched question is re-attached to
    `reference_questions` by its `row_index`.
    """
    saved_decisions = progress["decisions"]
    if len(saved_decisions) != len(source_questions):
        raise ValueError(
            f"This progress file has {len(saved_decisions)} question(s), but "
            f"the uploaded CSV has {len(source_questions)}. Make sure you're "
            "uploading the same source CSV used to save this progress."
        )

    reference_by_row_index = {q.row_index: q for q in reference_questions}
    decisions = [
        MatchDecision.from_dict(data, source, reference_by_row_index)
        for data, source in zip(saved_decisions, source_questions)
    ]

    standalone_questions = [
        StandaloneQuestion.from_dict(data)
        for data in progress.get("standalone_questions", [])
    ]

    current_idx = progress.get("current_idx", 0)
    current_idx = max(0, min(current_idx, len(decisions) - 1)) if decisions else 0
    return decisions, current_idx, standalone_questions
