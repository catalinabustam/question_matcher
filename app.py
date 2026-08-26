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
from collections import defaultdict

import pandas as pd
import streamlit as st

from bm25 import load_bm25_retriever
from csv_io import QuestionCsvRepository
from datadictionary import (
    _source_field_value,
    available_field_types,
    build_data_dictionary,
)
from matching_service import QuestionMatchingService
from models import MatchDecision, MatchStatus
from redcap_validation import validate_record
from rules import build_new_question, build_variable_name
from translate import DeepLTranslator, OllamaTranslator, translate_questions
from translations import available_languages, load_translation
from dotenv import load_dotenv
from vector_db import (
    EMBEDDING_MODEL,
    INDEX_DATA_DIR,
    build_documents,
    build_ids,
    load_chromadb_collections,
)

AUTO_DETECT = "Auto-detect"
DEEPL_LANGUAGES = ["ES", "EN-US", "EN-GB", "PT-BR", "PT-PT", "FR", "DE", "IT", "CA"]
ENGLISH_OPTION = "English (original)"

st.set_page_config(page_title="Question Matcher", layout="wide")

NONE_OPTION = "— None —"
CREATE_NEW_LABEL = "➕ Create a new question"
IGNORE_LABEL = "🚫 Ignore this question (do not include in export)"

# Pagination over the candidate list: show this many at first, grow by this
# many per "Load more" click, up to this hard cap.
CANDIDATES_MAX = 100

load_dotenv(override=True)


# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading the reference catalog and search index...")
def _load_index():
    """Load the ARC index built ahead of time by `build_index.py`.
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


@st.cache_data(show_spinner=False)
def _load_translation_cached(language: str) -> pd.DataFrame:
    """Cached wrapper around `translations.load_translation` — the CSV
    itself never changes within a run, so re-reading it from disk on every
    export-section rerun would be wasted work.
    """
    return load_translation(language)


def _read_csv(uploaded_file, separator: str) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=separator, dtype=str).fillna("")


def _column_selector(df: pd.DataFrame, label: str, key: str, optional: bool = False,
                      preferred: tuple = (), disabled: bool = False):
    columns = list(df.columns)
    options = [NONE_OPTION] + columns if optional else columns
    default_index = 0
    for name in preferred:
        matches = [c for c in columns if c.strip().lower() == name.lower()]
        if matches:
            default_index = options.index(matches[0])
            break
    choice = st.selectbox(label, options, index=default_index, key=key, disabled=disabled)
    return None if choice == NONE_OPTION else choice


def _candidate_label(candidate, exact_match: bool = False) -> str:
    badge = "  ·  🟢 Same ARC variable name" if exact_match else ""
    return (f"{candidate.question.question}  ·  section: {candidate.question.section or '—'}"
            f"  ·  score: {candidate.score:.0%}{badge}")


def _is_exact_variable_match(candidate, source) -> bool:
    """Whether `candidate` has the exact same variable name as `source`.

    A strong signal the two represent the same field — much stronger than
    the text-similarity score — so it's used to pin the candidate first in
    the list and pre-select it (see `_render_question_flow`).
    """
    return bool(source.variable) and candidate.question.variable == source.variable


def _existing_match_question_number(candidate, exclude_idx: int) -> int | None:
    """Return the source question number that already uses this ARC row."""
    for question_idx, decision in enumerate(st.session_state.get("decisions", [])):
        if question_idx == exclude_idx or decision.status not in (
            MatchStatus.MATCHED,
            MatchStatus.MATCHED_CREATED,
        ):
            continue
        if any(
            matched.variable == candidate.question.variable
            for matched in decision.matched_questions
        ):
            return question_idx + 1
    return None


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



def _on_candidate_change(idx: int, candidate_key: str, candidate):
    """Callback triggered when any candidate checkbox state changes."""
    ignore_key = f"ignore_{idx}"

    # If any candidate gets checked, clear Ignore and Create New
    st.session_state[ignore_key] = False

    if not st.session_state.get(candidate_key):
        return

    existing_question_number = _existing_match_question_number(candidate, idx)
    if existing_question_number is None:
        return

    st.session_state[candidate_key] = False
    st.session_state[f"duplicate_match_alert_{idx}"] = (
        f"This ARC question already exists in source question "
        f"{existing_question_number}. Ignore this question or create a new one."
    )


def _init_session(source_qs, matcher: QuestionMatchingService, reference_df: pd.DataFrame,
                   arc_catalog_df: pd.DataFrame, scope_filters: dict):
    st.session_state.source_questions = source_qs
    st.session_state.decisions = [MatchDecision(source=q) for q in source_qs]
    st.session_state.matcher = matcher
    st.session_state.current_idx = 0
    st.session_state.flow_started = True
    
    st.session_state.reference_df = reference_df
    st.session_state.scope_filters = scope_filters
  
    st.session_state.arc_catalog_df = arc_catalog_df


def _reset_session():
    for key in (
        "source_questions",
        "decisions",
        "matcher",
        "current_idx",
        "flow_started",
        "mixed_match_default",
        "reference_df",
        "arc_catalog_df",
        "scope_filters",
        "arc_filter_columns",
        "arc_scope_values_Form",
        "arc_scope_values_Section",
        "allowed_row_indices",
    ):
        st.session_state.pop(key, None)


def _existing_variable_ids(exclude: MatchDecision | None = None) -> set[str]:
    """Every variable name already in use — the ARC catalog plus any
    already-created new questions — so `build_new_question` can guarantee
    its ARC-style variable name doesn't collide with either.

    `exclude` is the decision currently being (re)previewed, if any: its own
    previously-assigned `new_id` must not count as "existing", or every
    rerun would see it as a collision with itself and keep incrementing.
    """
    arc_catalog_df = st.session_state.get("arc_catalog_df")
    arc_ids = set(arc_catalog_df["Variable"].dropna().astype(str)) if arc_catalog_df is not None else set()
    created_ids = {
        d.new_id for d in st.session_state.get("decisions", [])
        if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED)
        and d.new_id and d is not exclude
    }
    return arc_ids | created_ids

def _save_decision(
    idx: int, selected_labels: list[str], candidates, new_section: str,
    new_text: str, create_new: bool = False, ignore: bool = False,
    new_field_name: str = "", new_form_name: str = "", new_field_type: str = "",
    new_variable_name_source: str = "",
    new_text_source: str = "",
    new_options: str = "", new_field_note: str = "", new_validation_type: str = "",
    new_validation_min: str = "", new_validation_max: str = "",
    new_identifier: str = "", new_branching_logic: str = "",
    new_required_field: str = "", new_custom_alignment: str = "",
    new_field_annotation: str = "", new_matrix_group_name: str = "",
    new_matrix_ranking: str = "", new_question_number: str = "",
    field_overrides: dict[str, str] | None = None,
):
    decision = st.session_state.decisions[idx]
    if create_new:
        matched_questions = [
            c.question
            for c in candidates
            if _candidate_label(
                c, exact_match=_is_exact_variable_match(c, decision.source)
            )
            in selected_labels
        ]
        sequence = (
            sum(
                1
                for d in st.session_state.decisions
                if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED)
            )
            + 1
        )
        preview = build_new_question(
            decision.source,
            sequence,
            section=new_section,
            existing_ids=_existing_variable_ids(exclude=decision),
        )
        decision.status = (
            MatchStatus.MATCHED_CREATED if selected_labels else MatchStatus.CREATED
        )
        decision.matched = matched_questions[0] if matched_questions else None
        decision.matches = matched_questions
        decision.new_id = new_field_name or preview["new_id"]
        decision.new_variable_name_source = new_variable_name_source
        decision.new_form_name = new_form_name or preview["new_form_name"]
        decision.new_section = new_section or preview["new_section"]
        decision.new_field_type = new_field_type or preview["new_field_type"]
        decision.new_text = new_text or preview["new_text"]
        decision.new_text_source = new_text_source
        decision.new_options = new_options or preview["new_options"]
        decision.new_field_note = new_field_note or preview["new_field_note"]
        decision.new_validation_type = new_validation_type or preview.get(
            "new_validation_type", ""
        )
        decision.new_validation_min = (
            new_validation_min or preview["new_validation_min"]
        )
        decision.new_validation_max = (
            new_validation_max or preview["new_validation_max"]
        )
        decision.new_identifier = new_identifier or preview.get("new_identifier", "")
        decision.new_branching_logic = new_branching_logic or preview.get(
            "new_branching_logic", ""
        )
        decision.new_required_field = (
            new_required_field or preview["new_required_field"]
        )
        decision.new_custom_alignment = new_custom_alignment or preview.get(
            "new_custom_alignment", ""
        )
        decision.new_field_annotation = new_field_annotation or preview.get(
            "new_field_annotation", ""
        )
        decision.new_matrix_group_name = new_matrix_group_name or preview.get(
            "new_matrix_group", ""
        )
        decision.new_matrix_ranking = new_matrix_ranking or preview.get(
            "new_matrix_ranking", ""
        )
        decision.new_question_number = new_question_number or preview.get(
            "new_question_number", ""
        )
        decision.field_overrides = {}
    elif ignore:
        decision.status = MatchStatus.IGNORED
        decision.matched = None
        decision.matches = []
        decision.new_id = decision.new_section = decision.new_text = ""
        decision.new_variable_name_source = ""
        decision.new_text_source = ""
        decision.new_form_name = decision.new_field_type = decision.new_options = ""
        decision.new_field_note = decision.new_validation_type = (
            decision.new_validation_min
        ) = ""
        decision.new_validation_max = decision.new_identifier = (
            decision.new_branching_logic
        ) = ""
        decision.new_required_field = decision.new_custom_alignment = (
            decision.new_field_annotation
        ) = ""
        decision.new_matrix_group_name = decision.new_matrix_ranking = (
            decision.new_question_number
        ) = ""
        decision.field_overrides = {}
    else:
        matched_questions = [
            c.question
            for c in candidates
            if _candidate_label(
                c, exact_match=_is_exact_variable_match(c, decision.source)
            )
            in selected_labels
        ]
        decision.status = MatchStatus.MATCHED
        decision.matched = matched_questions[0] if matched_questions else None
        decision.matches = matched_questions
        decision.new_id = decision.new_section = decision.new_text = ""
        decision.new_variable_name_source = ""
        decision.new_text_source = ""
        decision.new_form_name = decision.new_field_type = decision.new_options = ""
        decision.new_field_note = decision.new_validation_type = (
            decision.new_validation_min
        ) = ""
        decision.new_validation_max = decision.new_identifier = (
            decision.new_branching_logic
        ) = ""
        decision.new_required_field = decision.new_custom_alignment = (
            decision.new_field_annotation
        ) = ""
        decision.new_matrix_group_name = decision.new_matrix_ranking = (
            decision.new_question_number
        ) = ""
        decision.field_overrides = (
            field_overrides or {} if len(matched_questions) == 1 else {}
        )


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #


def _render_sidebar_upload():
    st.sidebar.header("1. Upload files")
    separator = st.sidebar.selectbox("CSV separator", [",", ";", "\t"], index=0)
    source_file = st.sidebar.file_uploader("CSV to process", type="csv")
    return separator, source_file


def _render_translation_form():
    st.sidebar.header("2. Translation")
    use_translation = st.sidebar.checkbox(
        "Translate source CSV questions before comparing", value=False
    )

    translator_type = "DeepL"
    api_key = ""
    ollama_model = None
    ollama_base_url = "http://localhost:11434"
    source_lang = None

    if use_translation:
        translator_type = st.sidebar.selectbox(
            "Translation provider", ["DeepL", "Ollama"], index=0
        )

        if translator_type == "DeepL":
            api_key = os.getenv("DEEPL_API_KEY", "")
            source_choice = st.sidebar.selectbox(
                "Source language", [AUTO_DETECT] + DEEPL_LANGUAGES, index=0
            )
            source_lang = None if source_choice == AUTO_DETECT else source_choice
        else:  # Ollama
            ollama_base_url = st.sidebar.text_input(
                "Ollama base URL", value="http://localhost:11434"
            )
            # Fetch available models
            try:
                available_models = OllamaTranslator.get_available_models(ollama_base_url)
            except Exception:
                available_models = []

            if available_models:
                ollama_model = st.sidebar.selectbox(
                    "Ollama model", available_models, index=0
                )
            else:
                st.sidebar.warning("No Ollama models found. Make sure Ollama is running.")
                ollama_model = st.sidebar.text_input("Model name (manual)", value="llama3.2")

            source_choice = st.sidebar.selectbox(
                "Source language", [AUTO_DETECT] + DEEPL_LANGUAGES, index=0
            )
            source_lang = None if source_choice == AUTO_DETECT else source_choice

    return use_translation, translator_type, api_key, ollama_model, ollama_base_url, source_lang


def _render_search_scope(reference_df: pd.DataFrame) -> dict:
    """Sidebar UI for the Form and Section search scope selected at startup."""
    st.sidebar.header("3. Search scope (optional)")
    st.sidebar.caption("Choose ARC forms or sections before matching starts.")

    filters = {}
    for col in ("Form", "Section"):
        if col not in reference_df.columns:
            continue
        options = sorted(v for v in reference_df[col].astype(str).unique() if v.strip())
        chosen = st.sidebar.multiselect(
            f"{col}", options, key=f"arc_scope_values_{col}"
        )
        if chosen:
            filters[col] = chosen

    if filters and st.sidebar.button("Clear filters"):
        for col in ("Form", "Section"):
            st.session_state.pop(f"arc_scope_values_{col}", None)
        st.rerun()

    return filters


def _render_candidate_filter(reference_df: pd.DataFrame,
                             scope_filters: dict | None = None) -> dict:
    """Sidebar UI for narrowing candidates while matching is in progress."""
    st.sidebar.header("4. Filter candidates (optional)")
    st.sidebar.caption("Restrict which ARC rows can be suggested as matches.")

    available_df = reference_df
    for col, values in (scope_filters or {}).items():
        available_df = available_df[available_df[col].astype(str).isin(values)]

    selected_columns = st.sidebar.multiselect(
        "Filter by column(s)", list(available_df.columns), key="arc_filter_columns"
    )

    filters = {}
    for col in selected_columns:
        options = sorted(
            v for v in available_df[col].dropna().astype(str).unique() if v.strip()
        )
        chosen = st.sidebar.multiselect(
            f"'{col}' values", options, key=f"arc_filter_values_{col}"
        )
        if chosen:
            filters[col] = chosen

    if selected_columns and st.sidebar.button("Clear candidate filters"):
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
    st.sidebar.caption(
        f"{len(matched)} / {len(reference_df)} reference rows match the filter."
    )
    return matched


def _render_mapping(df: pd.DataFrame, prefix: str, is_source: bool = True):
    """Render column mapping UI for source (REDCap data dictionary) or reference (ARC catalog)."""
    st.markdown(f"**Columns — {prefix}**")

    # REDCap Data Dictionary columns (for source)
    redcap_columns = [
        ("Variable / Field Name", "variable", "Variable"),
        ("Form Name", "form_name", "form"),
        ("Section Header", "section", "Section"),
        ("Field Type", "field_type", "Type"),
        ("Field Label", "question", "Question"),
        ("Definition (optional)", "definition", "Definition"),
        ("Choices, Calculations, OR Slider Labels", "options", "Answer Options"),
        ("Field Note", "field_note", ""),
        ("Text Validation Type OR Show Slider Number", "validation", "Validation"),
        ("Text Validation Min", "validation_min", "Minimum"),
        ("Text Validation Max", "validation_max", "Maximum"),
        ("Identifier?", "identifier", "Identifier"),
        ("Branching Logic (Show field only if...)", "branching_logic", "Skip Logic"),
        ("Required Field?", "required_field", ""),
        ("Custom Alignment", "custom_alignment", ""),
        ("Question Number (surveys only)", "question_number", ""),
        ("Matrix Group Name", "matrix_group", ""),
        ("Matrix Ranking?", "matrix_ranking", ""),
        ("Field Annotation", "field_annotation", ""),
    ]

    # Render selectors for each column
    results = {}
    for label, key, preferred in redcap_columns:
        # For required columns (id, question), don't allow None
        optional = key not in ("question")
        results[key] = _column_selector(
            df,
            f"{label}",
            f"{prefix}_{key}",
            optional=optional,
            preferred=(preferred, label),
            disabled=not is_source,
        )

    return results


def _render_progress():
    decisions = st.session_state.decisions
    total = len(decisions)
    matched = sum(
        1
        for d in decisions
        if d.status in (MatchStatus.MATCHED, MatchStatus.MATCHED_CREATED)
    )
    created = sum(
        1
        for d in decisions
        if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED)
    )
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
        st.markdown(
            f"**Original question:** {source.question}"
            f"  ·  **Original definition:** {source.definition or '—'}"
        )
        st.markdown(
            f"**Section:** {source.section or '—'}"
            f"  ·  **Answer type:** {source.field_type or '—'}"
        )
        if source.options:
            st.markdown(f"**Options:** {source.options}")

        default_translation_question = (
            getattr(decision, "edited_translated_question", "")
            or source.translated_question
            or source.question
        )
        default_translation_definition = (
            getattr(decision, "edited_translated_definition", "")
            or source.translated_definition
            or source.definition
        )
        translated_question_input = st.text_area(
            "Translated question (editable)",
            value=default_translation_question,
            key=f"translated_edit_{idx}",
            height=68,
            help="Edit the text if the automatic translation isn't quite right, "
            "then click 'Recalculate similarity' to refresh the suggested matches.",
        )
        translated_definition_input = st.text_area(
            "Translated definition (editable)",
            value=default_translation_definition,
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
        source,
        top_n=CANDIDATES_MAX,
        override_question=override_question,
        override_definition=override_definition,
        allowed_row_indices=st.session_state.get("allowed_row_indices"),
    )

    # If one of the candidates has the exact same variable name as the
    # source question, surface it first — it's almost certainly the right
    # match, and matching by identical ARC-style ID is a much stronger
    # signal than the text-similarity score.
    exact_match_pos = next(
        (i for i, c in enumerate(candidates) if _is_exact_variable_match(c, source)),
        None,
    )
    if exact_match_pos is not None and exact_match_pos != 0:
        candidates.insert(0, candidates.pop(exact_match_pos))

    num_candidates = len(candidates)
    visible_labels = [
        _candidate_label(c, exact_match=_is_exact_variable_match(c, source))
        for c in candidates
    ]

    if not candidates:
        st.warning(
            "No reference questions match the current filter. "
            "Adjust the filter in the sidebar, ignore this question, or create a new one."
        )
    duplicate_match_alert = st.session_state.pop(
        f"duplicate_match_alert_{idx}", None
    )
    if duplicate_match_alert:
        st.error(duplicate_match_alert)

    default_selected = set()
    if decision.status in (MatchStatus.MATCHED, MatchStatus.MATCHED_CREATED):
        default_selected = {
            _candidate_label(c, exact_match=_is_exact_variable_match(c, source))
            for c in candidates
            if any(
                matched.row_index == c.question.row_index
                for matched in decision.matched_questions
            )
        }
    elif decision.status == MatchStatus.PENDING and candidates:
        default_selected = {visible_labels[0]}

    # Initialize keys in st.session_state before rendering widgets
    ignore_key = f"ignore_{idx}"
    if ignore_key not in st.session_state:
        st.session_state[ignore_key] = decision.status == MatchStatus.IGNORED

    create_new_key = f"create_new_{idx}"
    if create_new_key not in st.session_state:
        st.session_state[create_new_key] = decision.status in (
            MatchStatus.CREATED,
            MatchStatus.MATCHED_CREATED,
        )

    ignore = st.checkbox(
        IGNORE_LABEL,
        key=ignore_key,
        on_change=_on_status_change,
        args=(idx, "ignore", num_candidates),
    )

    selected_match_labels = []
    with st.container(height=300):
        for i, candidate in enumerate(candidates):
            label = _candidate_label(
                candidate, exact_match=_is_exact_variable_match(candidate, source)
            )
            key = f"candidate_{idx}_{i}"

            if key not in st.session_state:
                st.session_state[key] = label in default_selected

            checkbox_col, action_col = st.columns([6, 2])
            checked = checkbox_col.checkbox(
                label,
                key=key,
                on_change=_on_candidate_change,
                args=(idx, key, candidate),
            )
            if checked:
                selected_match_labels.append(label)

            with action_col, st.popover("View full ARC row"):
                arc_row = st.session_state.reference_df.loc[
                    candidate.question.row_index
                ]
                st.dataframe(
                    arc_row.astype(str).rename("Value"), use_container_width=True
                )

    st.caption(
        f"✅ {len(selected_match_labels)} of {num_candidates} candidate(s) selected for matching."
    )

    if create_new_key not in st.session_state:
        st.session_state[create_new_key] = decision.status in (
            MatchStatus.CREATED,
            MatchStatus.MATCHED_CREATED,
        )
    create_new = st.checkbox(
        CREATE_NEW_LABEL,
        key=create_new_key,
        on_change=_on_status_change,
        args=(idx, "create_new", num_candidates),
    )

    # Initialize all new question fields

    new_field_name = new_form_name = new_section = new_field_type = new_text = ""
    new_options = new_field_note = new_validation_type = ""
    new_validation_min = new_validation_max = ""
    new_identifier = new_branching_logic = new_required_field = ""
    new_custom_alignment = new_field_annotation = ""
    new_matrix_group_name = new_matrix_ranking = new_question_number = ""
    variable_name_source = ""
    new_text_source = ""
    variable_name_conflict = False
    create_new_errors: list[str] = []

    if create_new:
        preview_seq = (
            sum(
                1
                for d in st.session_state.decisions
                if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED)
            )
            + 1
        )

        # Get available forms from ARC catalog

        available_forms = sorted(
            st.session_state.reference_df["Form"].dropna().astype(str).unique().tolist()
        )
        available_forms_lower = [form.lower() for form in available_forms]

        if (
            source.form_name is not None
            and source.form_name.lower() not in available_forms_lower
        ):
            available_forms = [source.form_name] + available_forms

        # Also include any custom form name from the saved decision
        if (
            decision.new_form_name
            and decision.new_form_name.lower() not in available_forms_lower
        ):
            available_forms = [decision.new_form_name] + available_forms
            available_forms_lower = [form.lower() for form in available_forms]

        default_selected_form = decision.new_form_name or source.form_name or None

        available_sections = sorted(
            [
                x
                for x in st.session_state.reference_df["Section"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if x
            ]
        )

        available_sections_lower = [section.lower() for section in available_sections]

        if (
            source.section != ""
            and source.section.lower() not in available_sections_lower
        ):
            available_sections = [source.section] + available_sections
            available_sections_lower = [section.lower() for section in available_sections]

        # Also include any custom section name from the saved decision
        if (
            decision.new_section
            and decision.new_section.lower() not in available_sections_lower
        ):
            available_sections = [decision.new_section] + available_sections
            available_sections_lower = [section.lower() for section in available_sections]

        default_selected_section = decision.new_section or source.section or None

        # Defaults for the free-text fields below (question text, options,
        # validation, ...) — section-independent, so safe to compute before
        # the Section Header widget runs.
        preview = build_new_question(
            source, preview_seq, existing_ids=_existing_variable_ids(exclude=decision)
        )

        field_type_options = available_field_types(st.session_state.reference_df)

        # Streamlit handles the text input inline when accept_new_options=True
        new_section = st.selectbox(
            "Section Header *",
            options=[""] + available_sections,
            index=available_sections.index(default_selected_section) + 1
            if default_selected_section
            else 0,
            accept_new_options=True,
            key=f"sec_choice_{idx}",
            help="Choose an ARC section or type to add a new one.",
        )

        st.markdown("**New Question Details (Data Dictionary Fields)**")

        # Row 1: Form Name, Field Type
        col1, col2 = st.columns(2)
        with col1:
            new_form_name = st.selectbox(
                "Form Name *",
                options=[""] + available_forms,
                index=available_forms.index(default_selected_form) + 1
                if default_selected_form
                else 0,
                key=f"form_{idx}",
                help="Select the form this question belongs to",
            )
        with col2:
            preferred_field_type = decision.new_field_type or preview["new_field_type"]
            default_field_type = (
                preferred_field_type
                if preferred_field_type in field_type_options
                else ("text" if "text" in field_type_options else field_type_options[0])
            )
            new_field_type = st.selectbox(
                "Field Type *",
                options=field_type_options,
                index=field_type_options.index(default_field_type),
                key=f"ftype_{idx}",
                help="REDCap field type — options reflect types actually used in the ARC catalog",
            )

        # Row 2: Selected Section, Variable / Field Name
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Section: {new_section or '—'}")
        with col2:
            variable_name_source_key = f"vid_src_{idx}"
            if source.variable:
                if variable_name_source_key not in st.session_state:
                    st.session_state[variable_name_source_key] = (
                        decision.new_variable_name_source
                        or "Use source variable name"
                    )
                variable_name_source = st.radio(
                    "Variable name source",
                    ["Use source variable name", "Auto-generate (ARC convention)"],
                    key=variable_name_source_key,
                    horizontal=True,
                    help="Choose whether the suggested name below starts from the "
                    "source CSV's own ID or from ARC's naming convention. "
                    "Either way, you can still edit it freely.",
                )
            else:
                variable_name_source = "Auto-generate (ARC convention)"

            auto_generated_id = build_variable_name(
                section=new_section,
                question=source.translated_question or source.question,
                existing_ids=_existing_variable_ids(exclude=decision),
            )
            suggested_id = (
                source.variable
                if variable_name_source == "Use source variable name"
                and source.variable
                else auto_generated_id
            )

            field_name_key = f"vid_{idx}"
            field_name_context_key = f"vid_context_{idx}"
            context = (new_section, variable_name_source)
            if (
                field_name_key not in st.session_state
                or field_name_context_key not in st.session_state
            ):
                st.session_state[field_name_key] = decision.new_id or suggested_id
                st.session_state[field_name_context_key] = context
            elif st.session_state[field_name_context_key] != context:
                st.session_state[field_name_key] = suggested_id
                st.session_state[field_name_context_key] = context

            new_field_name = st.text_input(
                "Variable / Field Name *",
                key=field_name_key,
                help="ARC-style variable name (domain_topic_detail), auto-suggested — edit if needed",
            )
            variable_name_conflict = bool(
                new_field_name
            ) and new_field_name in _existing_variable_ids(exclude=decision)
            if variable_name_conflict:
                st.error(
                    f"Variable name '{new_field_name}' is already in use "
                    "(ARC catalog or another created question). Choose a different name."
                )

        # Row 3: Field Label (Question text)
        has_translation = bool(
            source.translated_question
            and source.translated_question != source.question
        )
        if has_translation:
            label_source_key = f"label_src_{idx}"
            if label_source_key not in st.session_state:
                st.session_state[label_source_key] = (
                    "Use original question"
                    if decision.new_text_source == "original"
                    else "Use translated question"
                )
            label_source_choice = st.radio(
                "Field Label source",
                ["Use translated question", "Use original question"],
                key=label_source_key,
                horizontal=True,
                help="Choose whether the Field Label below starts from the "
                "translated question text or the original source text. "
                "Either way, you can still edit it freely.",
            )
        else:
            label_source_choice = "Use original question"
        new_text_source = (
            "translated" if label_source_choice == "Use translated question" else "original"
        )

        suggested_text = (
            source.translated_question
            if new_text_source == "translated" and source.translated_question
            else source.question
        )

        text_key = f"text_{idx}"
        text_context_key = f"text_context_{idx}"
        if text_key not in st.session_state or text_context_key not in st.session_state:
            st.session_state[text_key] = decision.new_text or suggested_text
            st.session_state[text_context_key] = new_text_source
        elif st.session_state[text_context_key] != new_text_source:
            st.session_state[text_key] = suggested_text
            st.session_state[text_context_key] = new_text_source

        new_text = st.text_area(
            "Field Label *",
            key=text_key,
            height=80,
            help="The question text shown to users",
        )

        # Row 4: Choices, Calculations, OR Slider Labels
        new_options = st.text_area(
            "Choices, Calculations, OR Slider Labels",
            value=decision.new_options or preview["new_options"],
            key=f"choices_{idx}",
            height=80,
            help="For radio/dropdown/checkbox: pipe-separated 'code, label' pairs. For slider: min,max,step",
        )

        # Row 5: Field Note
        new_field_note = st.text_area(
            "Field Note",
            value=decision.new_field_note or preview["new_field_note"],
            key=f"fnote_{idx}",
            height=60,
            help="Optional note shown below the field",
        )

        # Row 6: Text Validation Type, Text Validation Min, Text Validation Max
        col1, col2, col3 = st.columns(3)
        with col1:
            new_validation_type = st.text_input(
                "Text Validation Type OR Show Slider Number",
                value=decision.new_validation_type
                or preview.get("new_validation_type", ""),
                key=f"vtype_{idx}",
                help="Validation type (e.g., integer, number, date_ymd, email, etc.)",
            )
        with col2:
            new_validation_min = st.text_input(
                "Text Validation Min",
                value=decision.new_validation_min or preview["new_validation_min"],
                key=f"vmin_{idx}",
                help="Minimum value for validation",
            )
        with col3:
            new_validation_max = st.text_input(
                "Text Validation Max",
                value=decision.new_validation_max or preview["new_validation_max"],
                key=f"vmax_{idx}",
                help="Maximum value for validation",
            )

        # Row 7: Required Field?
        new_required_field = st.selectbox(
            "Required Field?",
            options=["", "yes", "no"],
            index=["", "yes", "no"].index(decision.new_required_field)
            if decision.new_required_field in ["yes", "no"]
            else 0,
            key=f"req_{idx}",
            help="Whether this field is required",
        )

        with st.expander("Additional REDCap fields (optional)"):
            adv1, adv2 = st.columns(2)
            with adv1:
                default_identifier = decision.new_identifier or preview.get(
                    "new_identifier", ""
                )
                new_identifier = st.selectbox(
                    "Identifier?",
                    options=["", "y"],
                    index=1 if default_identifier == "y" else 0,
                    key=f"identifier_{idx}",
                    help="Mark 'y' if this field contains identifying data",
                )
                default_custom_alignment = decision.new_custom_alignment or preview.get(
                    "new_custom_alignment", ""
                )
                alignment_options = ["", "LH", "RH", "LV", "RV"]
                new_custom_alignment = st.selectbox(
                    "Custom Alignment",
                    options=alignment_options,
                    index=alignment_options.index(default_custom_alignment)
                    if default_custom_alignment in alignment_options
                    else 0,
                    key=f"calign_{idx}",
                )
                new_matrix_group_name = st.text_input(
                    "Matrix Group Name",
                    value=decision.new_matrix_group_name
                    or preview.get("new_matrix_group", ""),
                    key=f"matrixgrp_{idx}",
                )
                default_matrix_ranking = decision.new_matrix_ranking or preview.get(
                    "new_matrix_ranking", ""
                )
                new_matrix_ranking = st.selectbox(
                    "Matrix Ranking?",
                    options=["", "y"],
                    index=1 if default_matrix_ranking == "y" else 0,
                    key=f"matrixrank_{idx}",
                )
            with adv2:
                new_branching_logic = st.text_input(
                    "Branching Logic (Show field only if...)",
                    value=decision.new_branching_logic
                    or preview.get("new_branching_logic", ""),
                    key=f"branching_{idx}",
                    help="REDCap logic syntax, e.g. [some_var]='1'",
                )
                new_field_annotation = st.text_input(
                    "Field Annotation",
                    value=decision.new_field_annotation
                    or preview.get("new_field_annotation", ""),
                    key=f"fannot_{idx}",
                )
                new_question_number = st.text_input(
                    "Question Number (surveys only)",
                    value=decision.new_question_number
                    or preview.get("new_question_number", ""),
                    key=f"qnum_{idx}",
                )

        create_new_errors, create_new_warnings = validate_record(
            {
                "variable": new_field_name,
                "form_name": new_form_name,
                "section": new_section,
                "field_type": new_field_type,
                "label": new_text,
                "choices": new_options,
                "validation_type": new_validation_type,
                "validation_min": new_validation_min,
                "validation_max": new_validation_max,
                "branching_logic": new_branching_logic,
            },
            existing_ids=_existing_variable_ids(exclude=decision),
            available_field_types=field_type_options,
        )
        # The variable-name conflict/format problem already gets its own
        # inline st.error above — skip it here to avoid showing it twice.
        for err in create_new_errors:
            if not err.startswith("Variable/Field Name"):
                st.error(err)
        for warn in create_new_warnings:
            st.warning(warn)

    elif ignore:
        st.caption("This question will be excluded from the final export.")

    mixed_match_errors: list[str] = []
    field_overrides: dict[str, str] = dict(decision.field_overrides)

    # Check for variable name conflicts with other decisions for matched questions
    matched_variable_conflicts: list[str] = []
    if not create_new and not ignore and selected_match_labels:
        for label in selected_match_labels:
            candidate = next(
                c
                for c in candidates
                if _candidate_label(
                    c, exact_match=_is_exact_variable_match(c, source)
                )
                == label
            )
            existing_q_num = _existing_match_question_number(candidate, idx)
            if existing_q_num is not None:
                matched_variable_conflicts.append(
                    f"Variable '{candidate.question.variable}' is already matched to source question {existing_q_num}"
                )

    if not create_new and not ignore and len(selected_match_labels) == 1:
        matched_for_mix = next(
            c.question
            for c in candidates
            if _candidate_label(c, exact_match=_is_exact_variable_match(c, source))
            == selected_match_labels[0]
        )
        with st.container(border=True):
            st.markdown("**🔀 Mixed match — choose ARC vs. source per field**")
            field_overrides = {}
            mix_specs = [
                ("question", "Question text"),
                ("options", "Options"),
                ("field_type", "Type"),
                ("validation", "Validation"),
            ]

            def _sync_bulk_checkbox(changed_key: str, other_key: str) -> None:
                if st.session_state[changed_key]:
                    st.session_state[other_key] = False

            def _remember_bulk_choice(idx: int) -> None:
                if not st.session_state.get(f"mix_remember_{idx}"):
                    return
                if st.session_state.get(f"mix_arc_{idx}"):
                    st.session_state.mixed_match_default = "arc"
                elif st.session_state.get(f"mix_source_{idx}"):
                    st.session_state.mixed_match_default = "source"

            def _clear_bulk_checkboxes() -> None:
                st.session_state[f"mix_arc_{idx}"] = False
                st.session_state[f"mix_source_{idx}"] = False

            with st.popover("Bulk field selection"):
                bulk_cols = st.columns(2)
                remembered_default = st.session_state.get(
                    "mixed_match_default", "arc"
                )
                use_arc = bulk_cols[0].checkbox(
                    "Use ARC for all fields",
                    value=remembered_default == "arc",
                    key=f"mix_arc_{idx}",
                    on_change=_sync_bulk_checkbox,
                    args=(f"mix_arc_{idx}", f"mix_source_{idx}"),
                    help="Use ARC for the four fields below; the variable name stays the ARC name.",
                )
                use_source = bulk_cols[1].checkbox(
                    "Use source for all fields",
                    value=remembered_default == "source",
                    key=f"mix_source_{idx}",
                    on_change=_sync_bulk_checkbox,
                    args=(f"mix_source_{idx}", f"mix_arc_{idx}"),
                    help="Use source for the four fields below; the variable name stays the ARC name.",
                )
                remember_choice = st.checkbox(
                    "Remember this choice for next questions",
                    key=f"mix_remember_{idx}",
                    on_change=_remember_bulk_choice,
                    args=(idx,),
                )
                if remember_choice:
                    _remember_bulk_choice(idx)
            mix_cols = st.columns(4)
            for col, (key, label) in zip(mix_cols, mix_specs):
                if use_source:
                    default = "Use source"
                elif use_arc:
                    default = "Use ARC"
                else:
                    default = (
                        "Use source"
                        if decision.field_overrides.get(key) == "source"
                        else "Use ARC"
                    )
                if use_arc or use_source:
                    st.session_state[f"mix_{key}_{idx}"] = default
                choice = col.radio(
                    label,
                    ["Use ARC", "Use source"],
                    index=["Use ARC", "Use source"].index(default),
                    key=f"mix_{key}_{idx}",
                    on_change=_clear_bulk_checkboxes,
                )
                field_overrides[key] = "source" if choice == "Use source" else "arc"

            resolved = {
                key: (
                    _source_field_value(decision, key)
                    if field_overrides[key] == "source"
                    else (getattr(matched_for_mix, key) or "")
                )
                for key, _ in mix_specs
            }
            st.caption(
                f"Effective — text: {resolved['question']!r} · options: {resolved['options']!r} · "
                f"type: {resolved['field_type']!r} · validation: {resolved['validation']!r}"
            )

            mixed_match_errors, mixed_match_warnings = validate_record(
                {
                    "variable": matched_for_mix.variable,
                    "form_name": matched_for_mix.form_name or "",
                    "section": matched_for_mix.section or "",
                    "field_type": resolved["field_type"],
                    "label": resolved["question"],
                    "choices": resolved["options"],
                    "validation_type": resolved["validation"],
                    "validation_min": matched_for_mix.validation_min or "",
                    "validation_max": matched_for_mix.validation_max or "",
                    "branching_logic": matched_for_mix.branching_logic or "",
                },
                existing_ids=_existing_variable_ids(exclude=decision)
                - {matched_for_mix.variable},
                available_field_types=available_field_types(
                    st.session_state.reference_df
                ),
            )
            for err in mixed_match_errors:
                st.error(err)
            for warn in mixed_match_warnings:
                st.warning(warn)

    for conflict in matched_variable_conflicts:
        st.error(conflict)

    can_save = bool(ignore) or (
        not ignore
        and not mixed_match_errors
        and not matched_variable_conflicts
        and (
            bool(selected_match_labels)
            if not create_new
            else not (variable_name_conflict or bool(create_new_errors))
        )
    )
    if not can_save:
        st.caption(
            "Select one or more candidates, enter a section for the new question, or ignore this row."
        )

    def save_current_decision() -> None:
        if can_save:
            _save_decision(
                idx, selected_match_labels, candidates, new_section, new_text,
                create_new=create_new, ignore=ignore, new_field_name=new_field_name,
                new_variable_name_source=variable_name_source,
                new_text_source=new_text_source,
                new_form_name=new_form_name, new_field_type=new_field_type,
                new_options=new_options, new_field_note=new_field_note,
                new_validation_type=new_validation_type,
                new_validation_min=new_validation_min,
                new_validation_max=new_validation_max,
                new_identifier=new_identifier,
                new_branching_logic=new_branching_logic,
                new_required_field=new_required_field,
                new_custom_alignment=new_custom_alignment,
                new_field_annotation=new_field_annotation,
                new_matrix_group_name=new_matrix_group_name,
                new_matrix_ranking=new_matrix_ranking,
                new_question_number=new_question_number,
                field_overrides=field_overrides,
            )

    nav_cols = st.columns([1, 1, 1, 5])
    if nav_cols[0].button("⬅ Previous", disabled=idx == 0):
        save_current_decision()
        st.session_state.current_idx = max(0, idx - 1)
        st.rerun()
    if nav_cols[1].button("Save and continue ➡", type="primary", disabled=not can_save):
        save_current_decision()
        if idx < total - 1:
            st.session_state.current_idx = idx + 1
        st.rerun()
    if nav_cols[2].button("Next ➡", disabled=idx == total - 1):
        save_current_decision()
        st.session_state.current_idx = min(total - 1, idx + 1)
        st.rerun()


def _matched_arc_rows() -> pd.DataFrame:
    """Original ARC catalog rows for every MATCHED decision.

    Ignored and newly created questions have no corresponding ARC catalog
    row, so they're excluded here — only matches can produce a data
    dictionary entry (Type, Answer Options, Validation, etc. all come from
    the matched ARC row, not from the source question).
    """
    arc_catalog_df = st.session_state.arc_catalog_df
    matched_row_variable_names = []
    for decision in st.session_state.decisions:
        if decision.status not in (MatchStatus.MATCHED, MatchStatus.MATCHED_CREATED):
            continue
        for matched in decision.matched_questions:
            if matched.variable:
                matched_row_variable_names.append(matched.variable)

    return arc_catalog_df[arc_catalog_df["Variable"].isin(matched_row_variable_names)]


def _render_export():
    st.divider()
    st.subheader("4. Export result")
    st.caption('Questions marked "ignore" are excluded from this export.')
    df = QuestionCsvRepository.export(st.session_state.decisions)
    st.dataframe(df, use_container_width=True, height=250)
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Download result CSV",
        csv_bytes,
        file_name="matched_questions.csv",
        mime="text/csv",
    )

    st.markdown("**Data dictionary (REDCap format)**")
    st.caption(
        "Includes matched questions (with any per-field ARC/source overrides applied) "
        "and newly created questions — ignored questions are excluded."
    )

    language_options = [ENGLISH_OPTION] + available_languages()
    selected_language = st.selectbox(
        "Output language",
        language_options,
        key="output_language",
        help="Translate matched ARC questions/choices into another language "
        "(via ARC-Translations) before export. Newly created questions have "
        "no ARC variable to translate and stay in their original language.",
    )
    translation = (
        _load_translation_cached(selected_language)
        if selected_language != ENGLISH_OPTION
        else None
    )

    data_dictionary_df = build_data_dictionary(
        _matched_arc_rows(),
        st.session_state.arc_catalog_df,
        st.session_state.decisions,
        translation=translation,
    )
    dictionary_bytes = data_dictionary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇ Download data dictionary CSV",
        dictionary_bytes,
        file_name="datadictionary.csv",
        mime="text/csv",
        disabled=data_dictionary_df.empty,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main():
    st.title("Question Matcher against a Reference Catalog")

    separator, source_file = _render_sidebar_upload()
    use_translation, translator_type, api_key, ollama_model, ollama_base_url, source_lang = _render_translation_form()

    if st.sidebar.button("Reset"):
        _reset_session()
        st.rerun()

    if not st.session_state.get("flow_started"):
        if source_file:
            try:
                (
                    reference_df,
                    df_expanded,
                    collection_questions,
                    collection_ques_def,
                    documents,
                    ids,
                    bm25_retriever,
                    stemmer,
                ) = _load_index()
            except RuntimeError as exc:
                st.error(str(exc))
                return

            source_df = _read_csv(source_file, separator)

            st.markdown("### Column mapping")

            col_a, col_b = st.columns(2)
            with col_a:
                results_s = _render_mapping(source_df, "source", is_source=True)
            with col_b:
                results_r = _render_mapping(reference_df, "ARC", is_source=False)

            scope_filters = _render_search_scope(reference_df)
            allowed_row_indices = _allowed_row_indices(df_expanded, scope_filters)
            filter_clauses = [
                {col: {"$in": values}} for col, values in scope_filters.items()
            ]
            metadata_filters = (
                filter_clauses[0]
                if len(filter_clauses) == 1
                else {"$and": filter_clauses}
                if filter_clauses
                else None
            )

            if st.button(
                "Start comparison", type="primary", disabled=not results_s["question"]
            ):
                source_qs = QuestionCsvRepository.load(source_df, results_s)
                # IMPORTANT: loaded from `df_expanded`, not `reference_df`. The
                # ChromaDB collections and the BM25 index above were built over
                # the expanded catalog (one row per user-list item), so the
                # reference Question at position i must come from that same
                # dataframe for `reference[i]` to correspond to doc id `ids[i]`.
                reference_qs = QuestionCsvRepository.load(df_expanded, results_r)

                if use_translation:
                    try:
                        if translator_type == "DeepL":
                            translator = DeepLTranslator(api_key)
                            with st.spinner("Translating questions with DeepL..."):
                                source_qs = translate_questions(
                                    translator, questions=source_qs, source_lang=source_lang
                                )
                        else:  # Ollama
                            if not ollama_model:
                                st.error("Please select an Ollama model.")
                                return
                            translator = OllamaTranslator(ollama_model, ollama_base_url)
                            with st.spinner(f"Translating questions with Ollama ({ollama_model})..."):
                                source_qs = translate_questions(
                                    translator, questions=source_qs, source_lang=source_lang
                                )
                    except Exception as exc:
                        st.error(f"Error translating: {exc}")
                        return

                matcher = QuestionMatchingService(
                    reference=reference_qs,
                    collection_questions=collection_questions,
                    collection_ques_def=collection_ques_def,
                    documents=documents,
                    ids=ids,
                    bm25_retriever=bm25_retriever,
                    stemmer=stemmer,
                    arc_pd=reference_df,
                    allowed_row_indices=allowed_row_indices,
                    metadata_filters=metadata_filters,
                )

                _init_session(source_qs, matcher, df_expanded, reference_df, scope_filters)
                st.rerun()
        else:
            st.info(
                "Upload the CSV to process and the reference CSV in the sidebar to get started."
            )
        return

    _render_progress()

    filters = _render_candidate_filter(
        st.session_state.reference_df, st.session_state.get("scope_filters")
    )
    st.session_state.allowed_row_indices = _allowed_row_indices(
        st.session_state.arc_catalog_df, filters
    )

    _render_question_flow()
    _render_export()


if __name__ == "__main__":
    main()
