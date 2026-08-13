"""CSV loading and export (Repository pattern)."""
from typing import List, Optional

import pandas as pd

from models import MatchDecision, MatchStatus, Question


class QuestionCsvRepository:
    """Converts a DataFrame into `Question` objects and exports decisions to CSV."""

    @staticmethod
    def load(df: pd.DataFrame, question_col: str,  definition_col: str, section_col: Optional[str],
              options_col: Optional[str], id_col: Optional[str],
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

            final_question = ""
            if d.status == MatchStatus.MATCHED and d.matched:
                final_question = d.matched.question
            elif d.status == MatchStatus.CREATED:
                final_question = d.new_text

            rows.append({
                "original_section": d.source.section,
                "original_topic": d.source.topic or "",
                "original_answer_type": d.source.answer_type or "",
                "original_question": d.source.question,
                "translated_question": d.source.translated_question or "",
                "edited_translated_question": d.edited_translated_question or "",
                "original_options": d.source.options,
                "status": d.status.value,
                "matched_reference_id": d.matched.question_id if d.matched else "",
                "matched_reference_question": d.matched.question if d.matched else "",
                "new_question_id": d.new_id,
                "new_question_section": d.new_section,
                "new_question_text": d.new_text,
                "final_question": final_question,
            })
        return pd.DataFrame(rows)
