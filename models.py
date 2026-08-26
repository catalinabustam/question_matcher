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
