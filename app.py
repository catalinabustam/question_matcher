"""Streamlit application to match questions from a CSV against a reference catalog.

Flow:
1. The user uploads the CSV to process and the reference CSV, and maps their columns.
2. The app walks through the questions one by one, suggesting the most similar
   ones from the reference CSV. The user picks a match, ignores the question,
   or creates a new question following the fixed rules in `rules.py`.
3. A final CSV is exported with the original question, the matched one (if
   any), and/or the newly built question. Ignored questions are left out.
"""
import os

import pandas as pd
import streamlit as st

from bm25 import load_bm25_retriever
from csv_io import QuestionCsvRepository
from datadictionary import build_data_dictionary, _KEPT_FIELD_TYPES
from matching_service import QuestionMatchingService
from models import MatchDecision, MatchStatus
from rules import build_new_question
from translate import DeepLTranslator, translate_questions
from dotenv import load_dotenv
from vector_db import EMBEDDING_MODEL, INDEX_DATA_DIR, build_documents, build_ids, load_chromadb_collections

AUTO_DETECT = "Auto-detect"
DEEPL_LANGUAGES = ["ES", "EN-US", "EN-GB", "PT-BR", "PT-PT", "FR", "DE", "IT", "CA"]

st.set_page_config(page_title="Question Matcher", layout="wide")

NONE_OPTION = "— None —"
CREATE_NEW_LABEL = "➕ No match: create a new question"
IGNORE_LABEL = "🚫 Ignore this question (do not include in export)"

# Pagination over the candidate list: show this many at first, grow by this
# many per "Load more" click, up to this hard cap.
CANDIDATES_MAX = 50

load_dotenv(override=True)


# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading the reference catalog and search index...")
def _load_index():
    """Load the ARC index built ahead of time by `build_index.py`.

    The app never builds the ChromaDB collections or the BM25 index itself
    anymore — that's a separate, explicit step (`python build_index.py`)
    run only when the ARC catalog needs to be (re)downloaded and (re)indexed.
    This just reads back the already-built artifacts:

    - `index_data/arc_raw.csv` / `arc_expanded.csv` — the reference tables.
    - the ChromaDB collections persisted under `chromadb_data/`.
    - the BM25 index persisted under `arc_index_bm25/`.

    `documents`/`ids` aren't persisted separately; they're rebuilt
    deterministically from `df_expanded` (see `vector_db.build_documents`/
    `build_ids`), which guarantees they always match what the on-disk
    collections/BM25 index were actually built from.

    Raises a clear, actionable error if the index hasn't been built yet.
    """
    if not (INDEX_DATA_DIR / "arc_expanded.csv").exists():
        raise RuntimeError(
            "No reference index found. Build it first by running: python build_index.py"
        )

    reference_df = pd.read_csv(INDEX_DATA_DIR / "arc_raw.csv", dtype=str).fillna("")
    df_expanded = pd.read_csv(INDEX_DATA_DIR / "arc_expanded.csv", dtype=str).fillna("")

    documents = build_documents(df_expanded)
    ids = build_ids(df_expanded)

    collection_questions, collection_ques_def = load_chromadb_collections(EMBEDDING_MODEL)
    bm25_retriever, stemmer = load_bm25_retriever()

    return (reference_df, df_expanded, collection_questions, collection_ques_def,
            documents, ids, bm25_retriever, stemmer)


def _read_csv(uploaded_file, separator: str) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=separator, dtype=str).fillna("")


def _column_selector(df: pd.DataFrame, label: str, key: str, optional: bool = False,
                      preferred: tuple = ()):
    columns = list(df.columns)
    options = [NONE_OPTION] + columns if optional else columns
    default_index = 0
    for name in preferred:
        matches = [c for c in columns if c.strip().lower() == name.lower()]
        if matches:
            default_index = options.index(matches[0])
            break
    choice = st.selectbox(label, options, index=default_index, key=key)
    return None if choice == NONE_OPTION else choice


def _candidate_label(candidate) -> str:
    return (f"{candidate.question.question}  ·  section: {candidate.question.section or '—'}"
            f"  ·  score: {candidate.score:.0%}")


# --------------------------------------------------------------------------- #
# Session state handling & Callbacks
# --------------------------------------------------------------------------- #

def _on_status_change(idx: int, trigger: str, num_candidates: int):
    """Callback triggered when Ignore or Create New checkbox state changes."""
    ignore_key = f"ignore_{idx}"
    create_new_key = f"create_new_{idx}"

    if trigger == "ignore" and st.session_state.get(ignore_key):
        st.session_state[create_new_key] = False
        
        # Clear candidate selections
        for i in range(num_candidates):
            st.session_state[f"candidate_{idx}_{i}"] = False

        # Automatically save decision as IGNORED and move to the next question
        _save_decision(idx, selected_labels=[], candidates=[], new_section="", new_text="", ignore=True)
        
        total = len(st.session_state.source_questions)
        if st.session_state.current_idx < total - 1:
            st.session_state.current_idx += 1


    elif trigger == "create_new" and st.session_state.get(create_new_key):
        st.session_state[ignore_key] = False
        
        # Clear candidate selections
        for i in range(num_candidates):
            st.session_state[f"candidate_{idx}_{i}"] = False


def _on_candidate_change(idx: int):
    """Callback triggered when any candidate checkbox state changes."""
    ignore_key = f"ignore_{idx}"
    create_new_key = f"create_new_{idx}"
    
    # If any candidate gets checked, clear Ignore and Create New
    st.session_state[ignore_key] = False
    st.session_state[create_new_key] = False


def _init_session(source_qs, matcher: QuestionMatchingService, reference_df: pd.DataFrame,
                   arc_catalog_df: pd.DataFrame):
    st.session_state.source_questions = source_qs
    st.session_state.decisions = [MatchDecision(source=q) for q in source_qs]
    st.session_state.matcher = matcher
    st.session_state.current_idx = 0
    st.session_state.flow_started = True
    
    st.session_state.reference_df = reference_df
  
    st.session_state.arc_catalog_df = arc_catalog_df


def _reset_session():
    for key in ("source_questions", "decisions", "matcher", "current_idx", "flow_started",
                "reference_df", "arc_catalog_df", "arc_filter_columns", "allowed_row_indices"):
        st.session_state.pop(key, None)


def _save_decision(idx: int, selected_labels: list[str], candidates, new_section: str,
                  new_text: str, create_new: bool = False, ignore: bool = False):
    decision = st.session_state.decisions[idx]
    if create_new:
        sequence = sum(1 for d in st.session_state.decisions
                        if d.status == MatchStatus.CREATED) + 1
        new_id, default_section, default_text = build_new_question(decision.source, sequence)
        decision.status = MatchStatus.CREATED
        decision.matched = None
        decision.matches = []
        decision.new_id = new_id
        decision.new_section = new_section or default_section
        decision.new_text = new_text or default_text
    elif ignore:
        decision.status = MatchStatus.IGNORED
        decision.matched = None
        decision.matches = []
        decision.new_id = decision.new_section = decision.new_text = ""
    else:
        matched_questions = [
            c.question for c in candidates if _candidate_label(c) in selected_labels
        ]
        decision.status = MatchStatus.MATCHED
        decision.matched = matched_questions[0] if matched_questions else None
        decision.matches = matched_questions
        decision.new_id = decision.new_section = decision.new_text = ""


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #

def _render_sidebar_upload():
    st.sidebar.header("1. Upload files")
    separator = st.sidebar.selectbox("CSV separator", [",", ";", "\t"], index=0)
    source_file = st.sidebar.file_uploader("CSV to process", type="csv")
    return separator, source_file


def _render_translation_form():
    st.sidebar.header("2. Translation (DeepL)")
    use_translation = st.sidebar.checkbox(
        "Translate source CSV questions before comparing", value=False)

    source_lang = None
    api_key=''
    if use_translation:
        api_key = os.getenv("DEEPL_API_KEY", "")
        source_choice = st.sidebar.selectbox(
            "Source language", [AUTO_DETECT] + DEEPL_LANGUAGES, index=0)
        source_lang = None if source_choice == AUTO_DETECT else source_choice

    return use_translation, api_key, source_lang


def _render_candidate_filter(reference_df: pd.DataFrame) -> dict:
    """Sidebar UI: pick ARC columns, then values within each, to restrict
    which reference rows are eligible to be suggested as match candidates.

    Returns a dict of {column: [selected values]} for every column with at
    least one value chosen. An empty dict means "no filter active".
    """
    st.sidebar.header("3. Filter candidates (optional)")
    st.sidebar.caption("Restrict which ARC rows can be suggested as matches.")

    columns = list(reference_df.columns)
    selected_columns = st.sidebar.multiselect(
        "Filter by column(s)", columns, key="arc_filter_columns")

    filters = {}
    for col in selected_columns:
        options = sorted(v for v in reference_df[col].dropna().astype(str).unique() if v.strip())
        chosen = st.sidebar.multiselect(f"'{col}' values", options, key=f"arc_filter_values_{col}")
        if chosen:
            filters[col] = chosen

    if selected_columns and st.sidebar.button("Clear filters"):
        for col in selected_columns:
            st.session_state.pop(f"arc_filter_values_{col}", None)
        st.session_state.pop("arc_filter_columns", None)
        st.rerun()

    return filters


def _allowed_row_indices(reference_df: pd.DataFrame, filters: dict):
    """Turn {column: [values]} into a set of matching row_index values, or
    None if no filter is active (i.e. don't restrict candidates at all).
    Columns are AND-combined; values within a column are OR-combined.
    """
    if not filters:
        return None
    mask = pd.Series(True, index=reference_df.index)
    for col, values in filters.items():
        mask &= reference_df[col].astype(str).isin(values)
    matched = set(reference_df.index[mask])
    st.sidebar.caption(f"{len(matched)} / {len(reference_df)} reference rows match the filter.")
    return matched


def _render_mapping(df: pd.DataFrame, prefix: str):
    st.markdown(f"**Columns — {prefix}**")
    form_col = _column_selector(df, "Form column", f"{prefix}_form", optional=True,
                                     preferred=("Form","Form",))
    section_col = _column_selector(df, "Section column", f"{prefix}_section", optional=True,
                                        preferred=("Section","Section",))
    question_col = _column_selector(df, "Question column", f"{prefix}_question",
                                     preferred=("Question","Question",))
    definition_col = _column_selector(df, "Definition column", f"{prefix}_definition", optional=True,
                                         preferred=("Definition","Definition",))
    answer_type_col = _column_selector(df, "Answer type column", f"{prefix}_answer_type", optional=True,
                                        preferred=("Type", "Type"))
    options_col = _column_selector(df, "Answer options column", f"{prefix}_options", optional=True,
                                    preferred=("Answer Options","Answer Options",))
    body_system_col = _column_selector(df, "Body System column", f"{prefix}_body_system", optional=True,
                                        preferred=("Body System", "Body System",))
    id_col = _column_selector(df, "ID column", f"{prefix}_id", optional=True,
                               preferred=("Variable", "ID", "Id"))
    return form_col,question_col, definition_col, section_col, options_col, answer_type_col, id_col, body_system_col


def _render_progress():
    decisions = st.session_state.decisions
    total = len(decisions)
    matched = sum(1 for d in decisions if d.status == MatchStatus.MATCHED)
    created = sum(1 for d in decisions if d.status == MatchStatus.CREATED)
    ignored = sum(1 for d in decisions if d.status == MatchStatus.IGNORED)
    pending = total - matched - created - ignored

    cols = st.columns(5)
    cols[0].metric("Total", total)
    cols[1].metric("Matched", matched)
    cols[2].metric("New", created)
    cols[3].metric("Ignored", ignored)
    cols[4].metric("Pending", pending)
    st.progress((matched + created + ignored) / total if total else 0)


def _render_question_flow():
    idx = st.session_state.current_idx
    total = len(st.session_state.source_questions)
    source = st.session_state.source_questions[idx]
    decision = st.session_state.decisions[idx]

    st.markdown(f"**Question {idx + 1} of {total}**")
    with st.container(border=True):
        st.markdown(f"**Original question:** {source.question}"
                    f"  ·  **Original definition:** {source.definition or '—'}")
        st.markdown(f"**Section:** {source.section or '—'}" 
                    f"  ·  **Answer type:** {source.answer_type or '—'}") 
        if source.options:
            st.markdown(f"**Options:** {source.options}")

        default_translation_question = (getattr(decision, "edited_translated_question", "")
                                or source.translated_question or source.question)
        default_translation_definition = (getattr(decision, "edited_translated_definition", "")
                                or source.translated_definition or source.definition)
        translated_question_input = st.text_area(
            "Translated question (editable)", value=default_translation_question,
            key=f"translated_edit_{idx}",
            height=68,
            help="Edit the text if the automatic translation isn't quite right, "
                 "then click 'Recalculate similarity' to refresh the suggested matches.",    
        )
        translated_definition_input = st.text_area(
                    "Translated definition (editable)", value=default_translation_definition,
                    key=f"translated_def_edit_{idx}",
                    height=68,
                    help="Edit the text if the automatic translation isn't quite right, "
                         "then click 'Recalculate similarity' to refresh the suggested matches.",
                )
        if st.button("🔄 Recalculate similarity", key=f"recalc_{idx}"):
            decision.edited_translated_question = translated_question_input
            decision.edited_translated_definition = translated_definition_input
            st.rerun()

    override_question = getattr(decision, "edited_translated_question", "") or None
    override_definition = getattr(decision, "edited_translated_definition", "") or None

    # Fetch up to CANDIDATES_MAX candidates once (already sorted best-first).
    candidates = st.session_state.matcher.find_candidates(
        source, top_n=CANDIDATES_MAX,
        override_question=override_question,
        override_definition=override_definition,
        allowed_row_indices=st.session_state.get("allowed_row_indices"),
    )

    num_candidates = len(candidates)
    visible_labels = [_candidate_label(c) for c in candidates]

    if not candidates:
        st.warning("No reference questions match the current filter. "
                   "Adjust the filter in the sidebar, ignore this question, or create a new one.")
    
    default_selected = set()
    if decision.status == MatchStatus.MATCHED:
        default_selected = {
            _candidate_label(c) for c in candidates
            if any(matched.row_index == c.question.row_index for matched in decision.matched_questions)
        }
    elif decision.status == MatchStatus.PENDING and candidates:
        default_selected = {visible_labels[0]}

    # Initialize keys in st.session_state before rendering widgets
    ignore_key = f"ignore_{idx}"
    if ignore_key not in st.session_state:
        st.session_state[ignore_key] = decision.status == MatchStatus.IGNORED

    create_new_key = f"create_new_{idx}"
    if create_new_key not in st.session_state:
        st.session_state[create_new_key] = decision.status == MatchStatus.CREATED

    ignore = st.checkbox(
        IGNORE_LABEL,
        key=ignore_key,
        on_change=_on_status_change,
        args=(idx, "ignore", num_candidates),
    )

    selected_match_labels = []
    with st.container(height=300):
        for i, candidate in enumerate(candidates):
            label = _candidate_label(candidate)
            key = f"candidate_{idx}_{i}"

            if key not in st.session_state:
                st.session_state[key] = label in default_selected

            checkbox_col, action_col = st.columns([6, 2])
            checked = checkbox_col.checkbox(
                label,
                key=key,
                on_change=_on_candidate_change,
                args=(idx,),
            )
            if checked:
                selected_match_labels.append(label)

            with action_col:
                with st.popover("View full ARC row"):
                    arc_row = st.session_state.reference_df.loc[candidate.question.row_index]
                    st.dataframe(arc_row.astype(str).rename("Value"), use_container_width=True)
       
    create_new = st.checkbox(
        CREATE_NEW_LABEL,
        key=create_new_key,
        on_change=_on_status_change,
        args=(idx, "create_new", num_candidates),
    )

    new_section = new_text = ""
    if create_new:
        preview_seq = sum(1 for d in st.session_state.decisions
                           if d.status == MatchStatus.CREATED) + 1
        preview_id, default_section, default_text = build_new_question(source, preview_seq)
        new_section = st.text_input("New question section",
                                     value=decision.new_section or default_section, key=f"sec_{idx}")
        new_text = st.text_area("New question text",
                                 value=decision.new_text or default_text, key=f"text_{idx}")
    elif ignore:
        st.caption("This question will be excluded from the final export.")

    can_save = bool(selected_match_labels) or create_new or ignore
    if not can_save:
        st.caption("Select one or more candidates, choose a new question, or ignore this row.")

    nav_cols = st.columns([1, 1, 1, 5])
    if nav_cols[0].button("⬅ Previous", disabled=idx == 0):
        st.session_state.current_idx = max(0, idx - 1)
        st.rerun()
    if nav_cols[1].button("Save and continue ➡", type="primary", disabled=not can_save):
        _save_decision(idx, selected_match_labels, candidates, new_section, new_text,
                       create_new=create_new, ignore=ignore)
        if idx < total - 1:
            st.session_state.current_idx = idx + 1
        st.rerun()
    if nav_cols[2].button("Next ➡", disabled=idx == total - 1):
        st.session_state.current_idx = min(total - 1, idx + 1)
        st.rerun()


def _matched_arc_rows() -> pd.DataFrame:
    """Original ARC catalog rows for every MATCHED decision.

    Ignored and newly created questions have no corresponding ARC catalog
    row, so they're excluded here — only matches can produce a data
    dictionary entry (Type, Answer Options, Validation, etc. all come from
    the matched ARC row, not from the source question).
    """
    reference_df = st.session_state.reference_df
    arc_catalog_df = st.session_state.arc_catalog_df
    matched_row_question_ids = []
    for decision in st.session_state.decisions:
        if decision.status != MatchStatus.MATCHED:
            continue
        for matched in decision.matched_questions:
            if matched.question_id:
                matched_row_question_ids.append(matched.question_id)

    return arc_catalog_df[arc_catalog_df["Variable"].isin(matched_row_question_ids)]


def _render_export():
    st.divider()
    st.subheader("4. Export result")
    st.caption("Questions marked \"ignore\" are excluded from this export.")
    df = QuestionCsvRepository.export(st.session_state.decisions)
    st.dataframe(df, use_container_width=True, height=250)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Download result CSV", csv_bytes,
                        file_name="matched_questions.csv", mime="text/csv")

    st.markdown("**Data dictionary (REDCap format)**")
    st.caption("Built only from matched questions, using their original ARC catalog "
               "row — ignored and newly created questions are excluded.")
    data_dictionary_df = build_data_dictionary(_matched_arc_rows(), st.session_state.arc_catalog_df)
    dictionary_bytes = data_dictionary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ Download data dictionary CSV", dictionary_bytes,
                        file_name="datadictionary.csv", mime="text/csv",
                        disabled=data_dictionary_df.empty)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    st.title("Question Matcher against a Reference Catalog")

    separator, source_file = _render_sidebar_upload()
    use_translation, api_key, source_lang = _render_translation_form()

    if st.sidebar.button("Reset"):
        _reset_session()
        st.rerun()

    if not st.session_state.get("flow_started"):
        if source_file:
            try:
                (reference_df, df_expanded, collection_questions, collection_ques_def,
                 documents, ids, bm25_retriever, stemmer) = _load_index()
            except RuntimeError as exc:
                st.error(str(exc))
                return

            source_df = _read_csv(source_file, separator)

            st.markdown("### Column mapping")
            col_a, col_b = st.columns(2)
            with col_a:
                (s_form_col,s_question_col, s_definition_col, s_section_col, s_options_col, s_answer_type_col, s_id_col, s_body_system_col) = _render_mapping(source_df, "source")
            with col_b:
                (r_form_col,r_question_col, r_definition_col, r_section_col, r_options_col, r_answer_type_col, r_id_col, r_body_system_col) = _render_mapping(reference_df, "ARC")

            if st.button("Start comparison", type="primary", disabled=not (s_question_col and r_question_col)):
                source_qs = QuestionCsvRepository.load(
                    source_df, question_col=s_question_col, definition_col=s_definition_col, section_col=s_section_col,
                    options_col=s_options_col, id_col=s_id_col, answer_type_col=s_answer_type_col)
                # IMPORTANT: loaded from `df_expanded`, not `reference_df`. The
                # ChromaDB collections and the BM25 index above were built over
                # the expanded catalog (one row per user-list item), so the
                # reference Question at position i must come from that same
                # dataframe for `reference[i]` to correspond to doc id `ids[i]`.
                reference_qs = QuestionCsvRepository.load(
                    df_expanded, question_col=r_question_col, definition_col=r_definition_col, section_col=r_section_col,
                    options_col=r_options_col, id_col=r_id_col, answer_type_col=r_answer_type_col)
                if use_translation:
                    try:
                        translator = DeepLTranslator(api_key)
                        with st.spinner("Translating questions with DeepL..."):
                            source_qs = translate_questions(translator, questions=source_qs, source_lang=source_lang)
                    except Exception as exc:
                        st.error(f"Error translating with DeepL: {exc}")
                        return

                matcher = QuestionMatchingService(
                    reference=reference_qs,
                    collection_questions=collection_questions,
                    collection_ques_def=collection_ques_def,
                    documents=documents,
                    ids=ids,
                    bm25_retriever=bm25_retriever,
                    stemmer=stemmer,
                    arc_pd=reference_df
                )

                _init_session(source_qs, matcher, df_expanded, reference_df)
                st.rerun()
        else:
            st.info("Upload the CSV to process and the reference CSV in the sidebar to get started.")
        return

    _render_progress()

    filters = _render_candidate_filter(st.session_state.reference_df)
    st.session_state.allowed_row_indices = _allowed_row_indices(st.session_state.reference_df, filters)

    _render_question_flow()
    _render_export()


if __name__ == "__main__":
    main()