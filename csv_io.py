"""CSV loading and export (Repository pattern)."""
from typing import List, Optional

import pandas as pd

from models import MatchDecision, MatchStatus, Question


class QuestionCsvRepository:
    """Converts a DataFrame into `Question` objects and exports decisions to CSV."""

    @staticmethod
    def load(df: pd.DataFrame, question_col: str,  definition_col: str, section_col: Optional[str],
              options_col: Optional[str], id_col: str,
              topic_col: Optional[str] = None,
              answer_type_col: Optional[str] = None) -> List[Question]:
        questions = []
        for idx, row in df.iterrows():
            questions.append(Question(
                row_index=idx,
                section=str(row[section_col]).strip() if section_col else "",
                question=str(row[question_col]).strip(),
                definition=str(row[definition_col]).strip() if definition_col else None,
                options=str(row[options_col]).strip() if options_col else "",
                question_id=str(row[id_col]).strip() if id_col else None,
                topic=str(row[topic_col]).strip() if topic_col else None,
                answer_type=str(row[answer_type_col]).strip() if answer_type_col else None,
            ))
        return questions

    @staticmethod
    def export(decisions: List[MatchDecision]) -> pd.DataFrame:
        rows = []
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
                    rows.append({
                        "original_section": d.source.section,
                        "original_topic": d.source.topic or "",
                        "original_answer_type": d.source.answer_type or "",
                        "original_question": d.source.question,
                        "translated_question": d.source.translated_question or "",
                        "edited_translated_question": d.edited_translated_question or "",
                        "original_options": d.source.options,
                        "status": MatchStatus.MATCHED.value,
                        "matched_reference_id": matched.question_id,
                        "matched_reference_question": matched.question,
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
                        "final_question": matched.question,
                    })

            if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED):
                rows.append({
                    "original_section": d.source.section,
                    "original_topic": d.source.topic or "",
                    "original_answer_type": d.source.answer_type or "",
                    "original_question": d.source.question,
                    "translated_question": d.source.translated_question or "",
                    "edited_translated_question": d.edited_translated_question or "",
                    "original_options": d.source.options,
                    "status": MatchStatus.CREATED.value,
                    "matched_reference_id": "",
                    "matched_reference_question": "",
                    # Data dictionary fields for newly created questions
                    "new_question_id": d.new_id,
                    "new_form_name": d.new_form_name,
                    "new_question_section": d.new_section,
                    "new_field_type": d.new_field_type,
                    "new_question_text": d.new_text,
                    "new_choices": d.new_choices,
                    "new_field_note": d.new_field_note,
                    "new_validation_type": d.new_validation_type,
                    "new_validation_min": d.new_validation_min,
                    "new_validation_max": d.new_validation_max,
                    "new_identifier": d.new_identifier,
                    "new_branching_logic": d.new_branching_logic,
                    "new_required_field": d.new_required_field,
                    "new_custom_alignment": d.new_custom_alignment,
                    "new_field_annotation": d.new_field_annotation,
                    "final_question": d.new_text,
                })

        return pd.DataFrame(rows)
