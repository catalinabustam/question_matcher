"""CSV loading and export (Repository pattern)."""

from typing import ClassVar

import pandas as pd

from datadictionary import (
    _source_field_value,
    build_variable_rename_map,
    rename_variable_references,
)
from models import MatchDecision, MatchStatus, Question

_OVERRIDABLE_FIELDS = ("question", "options", "field_type", "validation")


def _resolved_field(decision: MatchDecision, matched: Question, key: str) -> str:
    """The value that will end up in the export/data dictionary for one of
    the four overridable fields, given `decision.field_overrides`."""
    if decision.field_overrides.get(key) == "source":
        return _source_field_value(decision, key)
    arc_value = {
        "question": matched.question,
        "options": matched.options,
        "field_type": matched.field_type,
        "validation": matched.validation,
    }[key]
    return arc_value or ""


class QuestionCsvRepository:
    """Converts a DataFrame into `Question` objects and exports decisions to CSV."""

    # REDCap Data Dictionary column names
    REDCAP_COLUMNS: ClassVar[list[str]] = [
        "Variable / Field Name",
        "Form Name",
        "Section Header",
        "Field Type",
        "Field Label",
        "Choices, Calculations, OR Slider Labels",
        "Field Note",
        "Text Validation Type OR Show Slider Number",
        "Text Validation Min",
        "Text Validation Max",
        "Identifier?",
        "Branching Logic (Show field only if...)",
        "Required Field?",
        "Custom Alignment",
        "Question Number (surveys only)",
        "Matrix Group Name",
        "Matrix Ranking?",
        "Field Annotation",
    ]

    @staticmethod
    def load(df: pd.DataFrame, column_mapping: dict[str, str]) -> list[Question]:
        records = df.to_dict(orient="records")

        return [
            Question(
                row_index=idx,
                **{
                    field: str(row[col]).strip()
                    for field, col in column_mapping.items()
                    if col in row and pd.notna(row[col])
                },
            )
            for idx, row in enumerate(records)
        ]

    @staticmethod
    def export(decisions: list[MatchDecision]) -> pd.DataFrame:
        rows = []
        # Any source question that got matched to a different ARC variable,
        # or created under an auto-generated ARC-style name, may still be
        # referenced by its *original* name in another question's branching
        # logic / calculation formula. Renaming those references is a single
        # pass over all decisions, done once up front.
        rename_map = build_variable_rename_map(decisions)

        for d in decisions:
            # Questions the user explicitly marked "ignore" are left out of
            # the export entirely — they never get a row in the final CSV.
            if d.status == MatchStatus.IGNORED:
                continue

            # A MATCHED_CREATED decision produces TWO independent rows: one
            # for the matched reference(s), one for the newly created
            # question. They're intentionally kept separate (rather than
            # merged into a single row) so each can be reviewed/exported on
            # its own.
            if d.status in (MatchStatus.MATCHED, MatchStatus.MATCHED_CREATED):
                for matched in d.matched_questions:
                    resolved = {
                        key: _resolved_field(d, matched, key)
                        for key in _OVERRIDABLE_FIELDS
                    }
                    rows.append(
                        {
                            "original_section": d.source.section,
                            "original_topic": d.source.topic or "",
                            "original_answer_type": d.source.field_type or "",
                            "original_question": d.source.question,
                            "translated_question": d.source.translated_question or "",
                            "edited_translated_question": d.edited_translated_question
                            or "",
                            "original_options": d.source.options,
                            "status": MatchStatus.MATCHED.value,
                            "matched_reference_id": matched.variable,
                            "matched_reference_question": matched.question,
                            "final_matched_question": resolved["question"],
                            "final_matched_options": resolved["options"],
                            "final_matched_field_type": resolved["field_type"],
                            "final_matched_validation": resolved["validation"],
                            "question_source": d.field_overrides.get("question", "arc"),
                            "options_source": d.field_overrides.get("options", "arc"),
                            "type_source": d.field_overrides.get("field_type", "arc"),
                            "validation_source": d.field_overrides.get(
                                "validation", "arc"
                            ),
                            "new_form_name": "",
                            "new_question_id": "",
                            "new_question_section": "",
                            "new_field_type": "",
                            "new_question_text": "",
                            "new_choices": "",
                            "new_field_note": "",
                            "new_validation_type": "",
                            "new_validation_min": "",
                            "new_validation_max": "",
                            "new_identifier": "",
                            "new_branching_logic": "",
                            "new_required_field": "",
                            "new_custom_alignment": "",
                            "new_field_annotation": "",
                            "final_question": resolved["question"],
                        }
                    )

            if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED):
                rows.append(
                    {
                        "original_section": d.source.section,
                        "original_topic": d.source.topic or "",
                        "original_answer_type": d.source.field_type or "",
                        "original_question": d.source.question,
                        "translated_question": d.source.translated_question or "",
                        "edited_translated_question": d.edited_translated_question
                        or "",
                        "original_options": d.source.options,
                        "status": MatchStatus.CREATED.value,
                        "matched_reference_id": "",
                        "matched_reference_question": "",
                        "final_matched_question": "",
                        "final_matched_options": "",
                        "final_matched_field_type": "",
                        "final_matched_validation": "",
                        "question_source": "",
                        "options_source": "",
                        "type_source": "",
                        "validation_source": "",
                        # Data dictionary fields for newly created questions
                        "new_question_id": d.new_id,
                        "new_form_name": d.new_form_name,
                        "new_question_section": d.new_section,
                        "new_field_type": d.new_field_type,
                        "new_question_text": d.new_text,
                        # `new_choices` only actually contains `[variable]`
                        # references for calc-type rows; renaming is a no-op
                        # for plain "code, label" choice text.
                        "new_choices": rename_variable_references(
                            d.new_options, rename_map
                        ),
                        "new_field_note": d.new_field_note,
                        "new_validation_type": d.new_validation_type,
                        "new_validation_min": d.new_validation_min,
                        "new_validation_max": d.new_validation_max,
                        "new_identifier": d.new_identifier,
                        "new_branching_logic": rename_variable_references(
                            d.new_branching_logic, rename_map
                        ),
                        "new_required_field": d.new_required_field,
                        "new_custom_alignment": d.new_custom_alignment,
                        "new_field_annotation": rename_variable_references(
                            d.new_field_annotation, rename_map
                        ),
                        "new_matrix_group_name": d.new_matrix_group_name,
                        "new_matrix_ranking": d.new_matrix_ranking,
                        "new_question_number": d.new_question_number,
                        "final_question": d.new_text,
                    }
                )

        return pd.DataFrame(rows)
