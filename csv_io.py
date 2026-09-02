"""CSV loading and export (Repository pattern)."""

from typing import ClassVar

import pandas as pd

from datadictionary import (
    _source_field_value,
    build_variable_rename_map,
    rename_variable_references,
)
from models import MatchDecision, MatchStatus, Question, StandaloneQuestion, iter_ordered_items

_OVERRIDABLE_FIELDS = (
    "form_name",
    "question",
    "options",
    "field_type",
    "section",
    "validation",
    "branching_logic",
    "field_note",
    "identifier",
    "required_field",
    "custom_alignment",
    "question_number",
    "matrix_group",
    "matrix_ranking",
    "field_annotation",
)


def _resolved_field(decision: MatchDecision, matched: Question, key: str) -> str:
    """The value that will end up in the export/data dictionary for one of
    the four overridable fields, given `decision.field_overrides`."""
    if decision.field_overrides.get(key) == "source":
        return _source_field_value(decision, key)
    arc_value = {
        "form_name": matched.form_name,
        "question": matched.question,
        "options": matched.options,
        "field_type": matched.field_type,
        "section": matched.section,
        "validation": matched.validation,
        "branching_logic": matched.branching_logic,
        "field_note": matched.field_note,
        "identifier": matched.identifier,
        "required_field": matched.required_field,
        "custom_alignment": matched.custom_alignment,
        "question_number": matched.question_number,
        "matrix_group": matched.matrix_group,
        "matrix_ranking": matched.matrix_ranking,
        "field_annotation": matched.field_annotation,
    }[key]
    return arc_value or ""


def _standalone_export_row(
    question: StandaloneQuestion, rename_map: dict[str, str]
) -> dict:
    """One export-table row for a standalone question."""
    variable_name = question.new_id or question.st_id
    return {
        "source_question_index": question.st_id,
        "original_form_name": "",
        "original_section": "",
        "original_topic": "",
        "original_answer_type": "",
        "original_question": "",
        "translated_question": "",
        "edited_translated_question": "",
        "original_options": "",
        "status": "standalone",
        "matched_reference_id": "",
        "matched_reference_question": "",
        "final_form_name": question.new_form_name,
        "final_matched_question": "",
        "final_matched_options": "",
        "final_matched_field_type": "",
        "final_matched_validation": "",
        "question_source": "",
        "options_source": "",
        "type_source": "",
        "validation_source": "",
        "section_source": "",
        "form_name_source": "",
        "new_form_name": question.new_form_name,
        "new_question_id": variable_name,
        "new_question_section": question.new_section,
        "new_field_type": question.new_field_type,
        "new_question_text": question.new_text,
        "new_choices": rename_variable_references(question.new_options, rename_map),
        "new_field_note": question.new_field_note,
        "new_validation_type": question.new_validation_type,
        "new_validation_min": question.new_validation_min,
        "new_validation_max": question.new_validation_max,
        "new_identifier": question.new_identifier,
        "new_branching_logic": rename_variable_references(
            question.new_branching_logic, rename_map
        ),
        "new_required_field": question.new_required_field,
        "new_custom_alignment": question.new_custom_alignment,
        "new_field_annotation": rename_variable_references(
            question.new_field_annotation, rename_map
        ),
        "new_matrix_group_name": question.new_matrix_group_name,
        "new_matrix_ranking": question.new_matrix_ranking,
        "new_question_number": question.new_question_number,
        "final_question": question.new_text,
    }


def _decision_export_rows(
    decision: MatchDecision, rename_map: dict[str, str]
) -> list[dict]:
    rows: list[dict] = []
    if decision.status in (MatchStatus.MATCHED, MatchStatus.MATCHED_CREATED):
        for matched in decision.matched_questions:
            resolved = {
                key: _resolved_field(decision, matched, key)
                for key in _OVERRIDABLE_FIELDS
            }
            rows.append(
                {
                    "source_question_index": decision.source.row_index + 1,
                    "original_form_name": decision.source.form_name or "",
                    "original_section": decision.source.section,
                    "original_topic": decision.source.topic or "",
                    "original_answer_type": decision.source.field_type or "",
                    "original_question": decision.source.question,
                    "translated_question": decision.source.translated_question or "",
                    "edited_translated_question": decision.edited_translated_question
                    or "",
                    "original_options": decision.source.options,
                    "status": MatchStatus.MATCHED.value,
                    "matched_reference_id": matched.variable,
                    "matched_reference_question": matched.question,
                    "final_form_name": resolved["form_name"],
                    "final_matched_question": resolved["question"],
                    "final_matched_options": resolved["options"],
                    "final_matched_field_type": resolved["field_type"],
                    "final_matched_validation": resolved["validation"],
                    "question_source": decision.field_overrides.get("question", "arc"),
                    "options_source": decision.field_overrides.get("options", "arc"),
                    "type_source": decision.field_overrides.get("field_type", "arc"),
                    "section_source": decision.field_overrides.get("section", "arc"),
                    "form_name_source": decision.field_overrides.get(
                        "form_name", "arc"
                    ),
                    "validation_source": decision.field_overrides.get(
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

    if decision.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED):
        rows.append(
            {
                "source_question_index": decision.source.row_index + 1,
                "original_form_name": decision.source.form_name or "",
                "original_section": decision.source.section,
                "original_topic": decision.source.topic or "",
                "original_answer_type": decision.source.field_type or "",
                "original_question": decision.source.question,
                "translated_question": decision.source.translated_question or "",
                "edited_translated_question": decision.edited_translated_question
                or "",
                "original_options": decision.source.options,
                "status": MatchStatus.CREATED.value,
                "matched_reference_id": "",
                "matched_reference_question": "",
                "final_form_name": decision.new_form_name,
                "final_matched_question": "",
                "final_matched_options": "",
                "final_matched_field_type": "",
                "final_matched_validation": "",
                "question_source": "",
                "options_source": "",
                "type_source": "",
                "validation_source": "",
                "section_source": "",
                "form_name_source": "",
                "new_form_name": decision.new_form_name,
                "new_question_id": decision.new_id,
                "new_question_section": decision.new_section,
                "new_field_type": decision.new_field_type,
                "new_question_text": decision.new_text,
                "new_choices": rename_variable_references(
                    decision.new_options, rename_map
                ),
                "new_field_note": decision.new_field_note,
                "new_validation_type": decision.new_validation_type,
                "new_validation_min": decision.new_validation_min,
                "new_validation_max": decision.new_validation_max,
                "new_identifier": decision.new_identifier,
                "new_branching_logic": rename_variable_references(
                    decision.new_branching_logic, rename_map
                ),
                "new_required_field": decision.new_required_field,
                "new_custom_alignment": decision.new_custom_alignment,
                "new_field_annotation": rename_variable_references(
                    decision.new_field_annotation, rename_map
                ),
                "new_matrix_group_name": decision.new_matrix_group_name,
                "new_matrix_ranking": decision.new_matrix_ranking,
                "new_question_number": decision.new_question_number,
                "final_question": decision.new_text,
            }
        )
    return rows


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
    def export(
        decisions: list[MatchDecision],
        standalone_questions: list[StandaloneQuestion] | None = None,
    ) -> pd.DataFrame:
        rows = []
        standalone_questions = standalone_questions or []
        # Any source question that got matched to a different ARC variable,
        # or created under an auto-generated ARC-style name, may still be
        # referenced by its *original* name in another question's branching
        # logic / calculation formula. Renaming those references is a single
        # pass over all decisions, done once up front.
        rename_map = build_variable_rename_map(decisions)

        for kind, item in iter_ordered_items(decisions, standalone_questions):
            if kind == "standalone":
                rows.append(_standalone_export_row(item, rename_map))
                continue

            decision = item
            # Questions the user explicitly marked "ignore" are left out of
            # the export entirely — they never get a row in the final CSV.
            if decision.status == MatchStatus.IGNORED:
                continue
            rows.extend(_decision_export_rows(decision, rename_map))

        return pd.DataFrame(rows)
