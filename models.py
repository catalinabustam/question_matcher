"""Domain models for questionnaire comparison."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MatchStatus(str, Enum):
    """Status of the decision made about a source question."""

    PENDING = "pending"
    MATCHED = "matched"
    CREATED = "created"
    MATCHED_CREATED = "matched and created"
    IGNORED = "ignored"  # user chose not to include this question in the export


@dataclass(frozen=True)
class Question:
    """A question coming from a CSV (source or reference)."""

    # Required core fields
    row_index: int
    question: str
    variable: str

    # Optional generic fields
    section: Optional[str] = None
    definition: str | None = None
    options: str | None = None
    translated_section: str | None = None
    translated_question: str | None = None
    translated_definition: str | None = None
    translated_options: str | None = None
    topic: str | None = None
    # Optional REDCap Data Dictionary fields
    form_name: str | None = None
    field_type: str | None = None
    field_note: str | None = None
    validation: str | None = None
    validation_min: str | None = None
    validation_max: str | None = None
    identifier: str | None = None
    branching_logic: str | None = None
    required_field: str | None = None
    custom_alignment: str | None = None
    field_annotation: str | None = None
    matrix_group: str | None = None
    matrix_ranking: str | None = None
    question_number: str | None = None


# Plain string fields on `MatchDecision` that round-trip through
# `to_dict`/`from_dict` unchanged (i.e. everything except `source`, `status`,
# `matched`/`matches`, and `field_overrides`, which all need their own
# handling — see those methods).
_DECISION_STR_FIELDS = (
    "new_section",
    "new_text",
    "new_id",
    "new_variable_name_source",
    "new_text_source",
    "new_form_name",
    "new_field_type",
    "new_options",
    "new_field_note",
    "new_validation_type",
    "new_validation_min",
    "new_validation_max",
    "new_identifier",
    "new_branching_logic",
    "new_required_field",
    "new_custom_alignment",
    "new_field_annotation",
    "new_matrix_group_name",
    "new_matrix_ranking",
    "new_question_number",
    "edited_translated_question",
    "edited_translated_definition",
)


@dataclass
class MatchCandidate:
    """A match candidate with its similarity score (0.0 - 1.0)."""

    question: Question
    score: float


@dataclass
class MatchDecision:
    """The decision made by the user for a source question."""

    source: Question
    status: MatchStatus = MatchStatus.PENDING
    matched: Question | None = None
    matches: list[Question] = field(default_factory=list)
    new_section: str = ""
    new_text: str = ""
    new_id: str = ""
    new_variable_name_source: str = ""
    new_text_source: str = ""
    # Data dictionary fields for newly created questions
    new_form_name: str = ""
    new_field_type: str = ""
    new_options: str = ""
    new_field_note: str = ""
    new_validation_type: str = ""
    new_validation_min: str = ""
    new_validation_max: str = ""
    new_identifier: str = ""
    new_branching_logic: str = ""
    new_required_field: str = ""
    new_custom_alignment: str = ""
    new_field_annotation: str = ""
    new_matrix_group_name: str = ""
    new_matrix_ranking: str = ""
    new_question_number: str = ""
    edited_translated_question: str = ""
    edited_translated_definition: str = ""
    # Per-field source for a MATCHED decision ("arc" or "source"), keyed by
    # "question" / "options" / "field_type" / "validation". Only meaningful
    # when status is MATCHED and exactly one question is matched.
    field_overrides: dict[str, str] = field(default_factory=dict)

    @property
    def matched_questions(self) -> list[Question]:
        if self.matches:
            return self.matches
        if self.matched:
            return [self.matched]
        return []

    def to_dict(self) -> dict:
        """Serialize this decision for `progress_io.build_progress_dict`.

        `source` isn't included — it's re-derived from the re-uploaded
        source CSV when resuming, and matched questions are stored by
        `row_index` (not the full `Question`) so they can be re-attached to
        whatever reference catalog is loaded at resume time.
        """
        data = {name: getattr(self, name) for name in _DECISION_STR_FIELDS}
        data["status"] = self.status.value
        data["matched_row_indices"] = [q.row_index for q in self.matched_questions]
        data["field_overrides"] = dict(self.field_overrides)
        return data

    @classmethod
    def from_dict(
        cls, data: dict, source: Question, reference_by_row_index: dict[int, Question]
    ) -> "MatchDecision":
        """Rebuild a decision saved by `to_dict`.

        `reference_by_row_index` resolves `matched_row_indices` back into
        `Question` objects — any row_index no longer present (e.g. the ARC
        index was rebuilt in between) is silently dropped rather than
        raising, since a stale match is still recoverable by hand.
        """
        decision = cls(source=source)
        decision.status = MatchStatus(data.get("status", MatchStatus.PENDING.value))
        decision.matches = [
            reference_by_row_index[row_index]
            for row_index in data.get("matched_row_indices", [])
            if row_index in reference_by_row_index
        ]
        decision.matched = decision.matches[0] if decision.matches else None
        for name in _DECISION_STR_FIELDS:
            setattr(decision, name, data.get(name, ""))
        decision.field_overrides = dict(data.get("field_overrides", {}))
        return decision
