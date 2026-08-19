from models import MatchDecision, MatchStatus, Question
from csv_io import QuestionCsvRepository


def _question(row_index: int, question: str, question_id: str) -> Question:
    return Question(
        row_index=row_index,
        question=question,
        question_id=question_id,
        section="Section A",
        definition="Definition text",
        answer_type="text",
    )


def test_export_expands_each_selected_match_into_its_own_row():
    source = _question(0, "What symptoms did the patient have?", "source_1")
    match_a = _question(10, "Symptoms", "arc_1")
    match_b = _question(11, "Symptoms present", "arc_2")

    decision = MatchDecision(
        source=source,
        status=MatchStatus.MATCHED,
        matched=match_a,
        matches=[match_a, match_b],
    )

    export_df = QuestionCsvRepository.export([decision])

    assert len(export_df) == 2
    assert export_df["matched_reference_id"].tolist() == ["arc_1", "arc_2"]
    assert export_df["matched_reference_question"].tolist() == ["Symptoms", "Symptoms present"]
    assert export_df["final_question"].tolist() == ["Symptoms", "Symptoms present"]
