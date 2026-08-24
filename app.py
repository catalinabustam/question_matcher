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
CREATE_NEW_LABEL = "➕ Create a new question"
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



def _on_candidate_change(idx: int):
    """Callback triggered when any candidate checkbox state changes."""
    ignore_key = f"ignore_{idx}"
   
    # If any candidate gets checked, clear Ignore and Create New
    st.session_state[ignore_key] = False


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
    new_options: str = "", new_field_note: str = "", new_validation_type: str = "",
    new_validation_min: str = "", new_validation_max: str = "",
    new_identifier: str = "", new_branching_logic: str = "",
    new_required_field: str = "", new_custom_alignment: str = "",
    new_field_annotation: str = "", new_matrix_group_name: str = "",
    new_matrix_ranking: str = "", new_question_number: str = ""
):
    decision = st.session_state.decisions[idx]
    if create_new:
        matched_questions = [
            c.question for c in candidates if _candidate_label(c) in selected_labels
        ]
        sequence = sum(1 for d in st.session_state.decisions
                        if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED)) + 1
        preview = build_new_question(
            decision.source, sequence, section=new_section,
            existing_ids=_existing_variable_ids(exclude=decision)
        )
        decision.status = (MatchStatus.MATCHED_CREATED
                           if selected_labels else MatchStatus.CREATED)
        decision.matched = matched_questions[0] if matched_questions else None
        decision.matches = matched_questions
        decision.new_id = new_field_name or preview["new_id"]
        decision.new_form_name = new_form_name or preview["new_form_name"]
        decision.new_section = new_section or preview["new_section"]
        decision.new_field_type = new_field_type or preview["new_field_type"]
        decision.new_text = new_text or preview["new_text"]
        decision.new_options = new_options or preview["new_options"]
        decision.new_field_note = new_field_note or preview["new_field_note"]
        decision.new_validation_type = new_validation_type or preview.get("new_validation_type", "")
        decision.new_validation_min = new_validation_min or preview["new_validation_min"]
        decision.new_validation_max = new_validation_max or preview["new_validation_max"]
        decision.new_identifier = new_identifier or preview.get("new_identifier", "")
        decision.new_branching_logic = new_branching_logic or preview.get("new_branching_logic", "")
        decision.new_required_field = new_required_field or preview["new_required_field"]
        decision.new_custom_alignment = new_custom_alignment or preview.get("new_custom_alignment", "")
        decision.new_field_annotation = new_field_annotation or preview.get("new_field_annotation", "")
        decision.new_matrix_group_name = new_matrix_group_name or preview.get("new_matrix_group_name", "")
        decision.new_matrix_ranking = new_matrix_ranking or preview.get("new_matrix_ranking", "")
        decision.new_question_number = new_question_number or preview.get("new_question_number", "")
    elif ignore:
        decision.status = MatchStatus.IGNORED
        decision.matched = None
        decision.matches = []
        decision.new_id = decision.new_section = decision.new_text = ""
        decision.new_form_name = decision.new_field_type = decision.new_options = ""
        decision.new_field_note = decision.new_validation_type = decision.new_validation_min = ""
        decision.new_validation_max = decision.new_identifier = decision.new_branching_logic = ""
        decision.new_required_field = decision.new_custom_alignment = decision.new_field_annotation = ""
        decision.new_matrix_group_name = decision.new_matrix_ranking = decision.new_question_number = ""
    else:
        matched_questions = [
            c.question for c in candidates if _candidate_label(c) in selected_labels
        ]
        decision.status = MatchStatus.MATCHED
        decision.matched = matched_questions[0] if matched_questions else None
        decision.matches = matched_questions
        decision.new_id = decision.new_section = decision.new_text = ""
        decision.new_form_name = decision.new_field_type = decision.new_options = ""
        decision.new_field_note = decision.new_validation_type = decision.new_validation_min = ""
        decision.new_validation_max = decision.new_identifier = decision.new_branching_logic = ""
        decision.new_required_field = decision.new_custom_alignment = decision.new_field_annotation = ""
        decision.new_matrix_group_name = decision.new_matrix_ranking = decision.new_question_number = ""


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
        ("Field Annotation", "field_annotation", "")
    ]


    # Render selectors for each column
    results = {}
    for label, key, preferred in redcap_columns:
        # For required columns (id, question), don't allow None
        optional = key not in ("question")
        results[key] = _column_selector(df, f"{label}", f"{prefix}_{key}", optional=optional, preferred=(preferred, label), disabled= not is_source)

    return results


def _render_progress():
    decisions = st.session_state.decisions
    total = len(decisions)
    matched = sum(1 for d in decisions
                  if d.status in (MatchStatus.MATCHED, MatchStatus.MATCHED_CREATED))
    created = sum(1 for d in decisions
                  if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED))
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
                    f"  ·  **Answer type:** {source.field_type or '—'}") 
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
    if decision.status in (MatchStatus.MATCHED, MatchStatus.MATCHED_CREATED):
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
        st.session_state[create_new_key] = decision.status in (
            MatchStatus.CREATED, MatchStatus.MATCHED_CREATED)

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
       
    st.caption(f"✅ {len(selected_match_labels)} of {num_candidates} candidate(s) selected for matching.")

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
    variable_name_conflict = False

    if create_new:

        preview_seq = sum(1 for d in st.session_state.decisions
                   if d.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED)) + 1
       
        # Get available forms from ARC catalog

        available_forms = sorted(
            st.session_state.reference_df["Form"].dropna().astype(str).unique().tolist()
        )
        available_forms_lower = [form.lower() for form in available_forms]

        if source.form_name is not None and source.form_name.lower() not in available_forms_lower:
            available_forms = [source.form_name] + available_forms

        default_selected_form = decision.new_form_name or source.form_name or None

        available_sections = sorted([ x for x in 
            st.session_state.reference_df["Section"].dropna().astype(str).unique().tolist() if x])
        
        available_sections_lower = [section.lower() for section in available_sections]
        
        if source.section !='' and source.section.lower() not in available_sections_lower:
            available_sections = [source.section] + available_sections

        default_selected_section = decision.new_section or source.section or None

        field_name_key = f"vid_{idx}"

        preview = build_new_question(
            source, preview_seq, section=new_section,
            existing_ids=_existing_variable_ids(exclude=decision)
        )

        field_name_key = f"vid_{idx}"
        field_name_section_key = f"vid_section_{idx}"
        if field_name_section_key not in st.session_state:
            st.session_state[field_name_key] = decision.new_id or preview["new_id"]
            st.session_state[field_name_section_key] = new_section
        elif st.session_state[field_name_section_key] != new_section:
            st.session_state[field_name_key] = preview["new_id"]
            st.session_state[field_name_section_key] = new_section

        # Streamlit handles the text input inline when accept_new_options=True
        new_section = st.selectbox(
            "Section Header *",
            options=[""] + available_sections,
            index=available_sections.index(default_selected_section) + 1 if default_selected_section else 0,
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
                index= available_forms.index(default_selected_form) + 1 if default_selected_form else 0,
                key=f"form_{idx}",
                help="Select the form this question belongs to"
            )
        with col2:
            new_field_type = st.selectbox(
                "Field Type *",
                options=_KEPT_FIELD_TYPES,
                index=_KEPT_FIELD_TYPES.index(decision.new_field_type) if decision.new_field_type in _KEPT_FIELD_TYPES else _KEPT_FIELD_TYPES.index('text'),
                key=f"ftype_{idx}",
                help="REDCap field type"
            )

        # Row 2: Selected Section, Variable / Field Name
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Section: {new_section or '—'}")
        with col2:
            new_field_name = st.text_input(
                "Variable / Field Name *",
                key=field_name_key,
                help="ARC-style variable name (domain_topic_detail), auto-suggested — edit if needed"
            )
            variable_name_conflict = (
                bool(new_field_name)
                and new_field_name in _existing_variable_ids(exclude=decision)
            )
            if variable_name_conflict:
                st.error(
                    f"Variable name '{new_field_name}' is already in use "
                    "(ARC catalog or another created question). Choose a different name."
                )

        # Row 3: Field Label (Question text)
        new_text = st.text_area(
            "Field Label *",
            value=decision.new_text or preview["new_text"],
            key=f"text_{idx}",
            height=80,
            help="The question text shown to users"
        )

        # Row 4: Choices, Calculations, OR Slider Labels
        new_options = st.text_area(
            "Choices, Calculations, OR Slider Labels",
            value=decision.new_options or preview["new_options"],
            key=f"choices_{idx}",
            height=80,
            help="For radio/dropdown/checkbox: pipe-separated 'code, label' pairs. For slider: min,max,step"
        )

        # Row 5: Field Note
        new_field_note = st.text_area(
            "Field Note",
            value=decision.new_field_note or preview["new_field_note"],
            key=f"fnote_{idx}",
            height=60,
            help="Optional note shown below the field"
        )

        # Row 6: Text Validation Type, Text Validation Min, Text Validation Max
        col1, col2, col3 = st.columns(3)
        with col1:
            new_validation_type = st.text_input(
                "Text Validation Type OR Show Slider Number",
                value=decision.new_validation_type or preview.get("new_validation_type", ""),
                key=f"vtype_{idx}",
                help="Validation type (e.g., integer, number, date_ymd, email, etc.)"
            )
        with col2:
            new_validation_min = st.text_input(
                "Text Validation Min",
                value=decision.new_validation_min or preview["new_validation_min"],
                key=f"vmin_{idx}",
                help="Minimum value for validation"
            )
        with col3:
            new_validation_max = st.text_input(
                "Text Validation Max",
                value=decision.new_validation_max or preview["new_validation_max"],
                key=f"vmax_{idx}",
                help="Maximum value for validation"
            )

        # Row 7: Required Field?
        new_required_field = st.selectbox(
            "Required Field?",
            options=["", "yes", "no"],
            index=["", "yes", "no"].index(decision.new_required_field) if decision.new_required_field in ["yes", "no"] else 0,
            key=f"req_{idx}",
            help="Whether this field is required"
        )
    elif ignore:
        st.caption("This question will be excluded from the final export.")

    can_save = (
        (bool(selected_match_labels) or (create_new and bool(new_section)))
        and not (create_new and variable_name_conflict)
    ) or ignore
    if not can_save:
        st.caption("Select one or more candidates, enter a section for the new question, or ignore this row.")

    nav_cols = st.columns([1, 1, 1, 5])
    if nav_cols[0].button("⬅ Previous", disabled=idx == 0):
        st.session_state.current_idx = max(0, idx - 1)
        st.rerun()
    if nav_cols[1].button("Save and continue ➡", type="primary", disabled=not can_save):
        _save_decision(
            idx, selected_match_labels, candidates, new_section, new_text,
            create_new=create_new, ignore=ignore,
            new_field_name=new_field_name, new_form_name=new_form_name,
            new_field_type=new_field_type, new_options=new_options,
            new_field_note=new_field_note, new_validation_type=new_validation_type,
            new_validation_min=new_validation_min, new_validation_max=new_validation_max,
            new_identifier=new_identifier, new_branching_logic=new_branching_logic,
            new_required_field=new_required_field, new_custom_alignment=new_custom_alignment,
            new_field_annotation=new_field_annotation, new_matrix_group_name=new_matrix_group_name,
            new_matrix_ranking=new_matrix_ranking, new_question_number=new_question_number
        )
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
                results_s = _render_mapping(source_df, "source", is_source=True)
            with col_b:
                results_r = _render_mapping(reference_df, "ARC", is_source=False)

        
            if st.button("Start comparison", type="primary", disabled=not results_s['question']):
                source_qs = QuestionCsvRepository.load(
                    source_df, results_s)
                # IMPORTANT: loaded from `df_expanded`, not `reference_df`. The
                # ChromaDB collections and the BM25 index above were built over
                # the expanded catalog (one row per user-list item), so the
                # reference Question at position i must come from that same
                # dataframe for `reference[i]` to correspond to doc id `ids[i]`.
                reference_qs = QuestionCsvRepository.load(
                    df_expanded, results_r)
                
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