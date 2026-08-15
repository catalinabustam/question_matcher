"""Domain models for questionnaire comparison."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MatchStatus(str, Enum):
    """Status of the decision made about a source question."""
    PENDING = "pending"
    MATCHED = "matched"
    CREATED = "created"
    IGNORED = "ignored"  # user chose not to include this question in the export


@dataclass(frozen=True)
class Question:
    """A question coming from a CSV (source or reference)."""
    row_index: int
    question: str
    question_id: str
    section: Optional[str] = None
    definition: Optional[str] = None
    options: Optional[str] = None
    translated_question: Optional[str] = None
    translated_definition: Optional[str] = None
    topic: Optional[str] = None
    answer_type: Optional[str] = None


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
    matched: Optional[Question] = None
    new_section: str = ""
    new_text: str = ""
    new_id: str = ""
    edited_translated_question: str = ""
    edited_translated_definition: str = ""
