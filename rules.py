"""Fixed rules for building a new question when no reference match is found.

Unlike a user-editable template, these rules are hard-coded: the new
question's variable name follows the ISARIC ARC naming convention
(https://isaric-arc.readthedocs.io/en/latest/sources/variable-naming.html),
derived deterministically from the source question's section, text, and
answer type. The question's own section and text are kept unchanged.

To change how new questions are built, edit this module directly.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

from datadictionary import _KEPT_FIELD_TYPES
from models import Question
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------------------------------- #
# ARC-style variable naming: `[domain]_[topic]_[detail]`
# See: https://isaric-arc.readthedocs.io/en/latest/sources/variable-naming.html
# --------------------------------------------------------------------------- #

# Short domain codes for the ARC sections seen most often. Unmapped sections
# fall back to a slug of their own name (see `_domain_code`) — this table
# only needs to cover the common/ambiguous cases where a naive slug would be
# misleading (e.g. "Follow Up" -> "fllow" reads worse than "follow").
_SECTION_DOMAIN_CODES: dict[str, str] = {
    "demographics": "demog",
    "inclusion criteria": "inclu",
    "comorbidities": "comor",
    "presentation": "pres",
    "vital signs": "vital",
    "laboratory results": "labs",
    "medication": "medi",
    "interventions": "inter",
    "vaccination": "vacci",
    "exposure history": "expo",
    "associated symptoms": "adsym",
    "readmission": "readm",
    "outcome": "outco",
    "testing": "test",
    "follow up": "follow",
    "symptoms": "sympt",
}


# Detail suffix inferred from the field's answer type, per the "Common
# Suffixes" table in the ARC naming convention.
_TYPE_DETAIL_SUFFIX: dict[str, str] = {
    "yesno": "yn",
    "truefalse": "yn",
    "date_dmy": "date",
    "datetime_dmy": "date",
    "number": "num",
    "integer": "num",
}

_DOMAIN_CODE_LENGTH = 5
_TOPIC_MAX_WORDS = 1
_TOPIC_MAX_LEN = 20


def _domain_code(section: str | None) -> str:
    """Map a section to the domain prefix used by its ARC variables."""
    key = (section or "").strip().lower()
    if key in _SECTION_DOMAIN_CODES:
        return _SECTION_DOMAIN_CODES[key]
    slug = re.sub(r"[^a-z]", "", key)
    return slug[:_DOMAIN_CODE_LENGTH] or "misc"


def _topic_code(question: str) -> list[str]:
    # Clean non-alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', question.lower())
    
    # Initialize TF-IDF Vectorizer with English stop words removal
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    
    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    # Rank words by highest TF-IDF score
    word_score_pairs = list(zip(feature_names, scores))
    sorted_pairs = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)

    second_topic, score = sorted_pairs[1]
    
    return second_topic.replace(" ", "")[:_TOPIC_MAX_LEN]


def build_variable_name(
    section: str | None,
    question: str,
    existing_ids: set[str] | None = None
) -> str:
    """Build an ARC-style `domain_topic[_detail]` variable name.

    Follows the ISARIC ARC naming convention: lowercase, underscore-
    separated `[domain]_[topic]_[detail]`, where `domain` is a short
    thematic code for the section, `topic` is the clinical concept (from
    the question text), and `detail` is an optional suffix (`_yn`, `_date`,
    `_num`, `_oth`, ...) inferred from the answer type or field note.

    If the resulting name collides with `existing_ids` (already-created
    variables, and/or the ARC catalog's own `Variable` column), a numbered
    suffix is appended — mirroring ARC's own convention for repeated
    instances (e.g. `vacci_covid19_date1`, `vacci_covid19_date2`).
    """
    base = f"{_domain_code(section)}_{_topic_code(question)}"

    existing_ids = existing_ids or set()
    if base not in existing_ids:
        return base

    sequence = 2
    while f"{base}{sequence}" in existing_ids:
        sequence += 1
    return f"{base}{sequence}"


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


def build_new_question(
    source: Question,
    sequence: int,
    form_name: str = "",
    field_type: str = "",
    choices: str = "",
    field_note: str = "",
    validation_min: str = "",
    validation_max: str = "",
    required_field: str = "",
    existing_ids: set[str] | None = None,
    section: str | None = None
) -> dict[str, Any]:
    
    del sequence  # kept for backward-compatible call signature; see docstring
    answer_type = source.answer_type or infer_answer_type(source.options or "")
    new_section = section if section is not None else source.section
    new_id = build_variable_name(
        section=new_section,
        question=source.translated_question or source.question,
        existing_ids=existing_ids
    )

    # Default field_type to "text" if not provided, but validate against REDCap allowed types
    if not field_type:
        field_type = "text"
    elif field_type not in _KEPT_FIELD_TYPES:
        # If provided field_type is not in allowed types, default to text
        field_type = "text"

    # Use source options if choices not provided
    if not choices:
        choices = source.options or ""

    # Use translated question if available, otherwise fall back to original
    new_text = source.translated_question or source.question

    return {
        "new_id": new_id,
        "new_form_name": form_name,
        "new_section": new_section or "",
        "new_field_type": field_type,
        "new_text": new_text,
        "new_choices": choices,
        "new_field_note": field_note,
        "new_validation_type": "",
        "new_validation_min": validation_min,
        "new_validation_max": validation_max,
        "new_identifier": "",
        "new_branching_logic": "",
        "new_required_field": required_field,
        "new_custom_alignment": "",
        "new_field_annotation": "",
    }
