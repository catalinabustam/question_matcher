"""Build a REDCap-style data dictionary CSV from ARC catalog rows.

Rows are always arranged in the original ARC form, section, and variable
order. New questions are anchored within their selected form/section block;
new sections and forms are appended after the known blocks. This keeps each
REDCap form contiguous without relying on source-CSV order.

Input rows must come from the ARC catalog (same columns as `ARC.csv`:
`Form`, `Section`, `Variable`, `Type`, `Question`, `Answer Options`,
`Validation`, `Minimum`, `Maximum`, `Skip Logic`).
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from models import MatchDecision, MatchStatus, StandaloneQuestion, iter_ordered_items
from translations import apply_translation

_DESCRIPTIVE_LABEL_TEMPLATE = (
    '<div class="rich-text-field-label"><h5 style="text-align: center;">'
    '<span style="color: #236fa1;">{label}</span></h5></div>'
)

_KEPT_FIELD_TYPES = [
    "text",
    "notes",
    "radio",
    "dropdown",
    "calc",
    "file",
    "checkbox",
    "yesno",
    "truefalse",
    "descriptive",
    "slider",
]

_TEXT_ONLY_TYPES = ["date_dmy", "number", "integer", "datetime_dmy"]

_SOURCE_COLUMNS = [
    "Form",
    "Section",
    "Variable",
    "Type",
    "Question",
    "Answer Options",
    "Validation",
    "Minimum",
    "Maximum",
    "Skip Logic",
]

_CRF_COLUMNS = [
    "Form Name",
    "Section Header",
    "Variable / Field Name",
    "Field Type",
    "Field Label",
    "Choices, Calculations, OR Slider Labels",
    "Text Validation Type OR Show Slider Number",
    "Text Validation Min",
    "Text Validation Max",
    "Branching Logic (Show field only if...)",
]

_REDCAP_COLUMNS = [
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

_FIELDNAME_COLUMN = "Variable / Field Name"
_FORM_COLUMN = "Form Name"
_SECTION_COLUMN = "Section Header"
_BRANCHING_LOGIC_COLUMN = "Branching Logic (Show field only if...)"
_CHOICES_COLUMN = "Choices, Calculations, OR Slider Labels"
_FIELD_ANNOTATION_COLUMN = "Field Annotation"

# Matches the variable name inside a branching-logic or calculation
# reference, e.g. pulls "pres_firstsym" out of both "[pres_firstsym]='1'"
# and the checkbox form "[pres_firstsym(88)]='1'" (word chars stop right
# before the "("), while keeping whatever comes right after it (the closing
# "]" or the opening "(") so a replacement can be spliced back in untouched.
_VARIABLE_REFERENCE_RE = re.compile(r"\[(\w+)")
_VARIABLE_REFERENCE_WITH_SUFFIX_RE = re.compile(r"\[(\w+)(\]|\()")

# Columns that can contain `[variable]`-style references to other rows and
# so need renaming when a referenced variable's name changes (see
# `build_variable_rename_map`) / scanning for missing dependencies (see
# `_insert_missing_branching_logic_rows`). The Choices column only actually
# contains such references for `calc`-type rows, but scanning it
# unconditionally is harmless: plain choice text (e.g. "1, Yes | 2, No")
# never matches the `[variable]` pattern.
_RENAMABLE_COLUMNS = (
    _BRANCHING_LOGIC_COLUMN,
    _FIELD_ANNOTATION_COLUMN,
    _CHOICES_COLUMN,
)

# Maps a MatchDecision.field_overrides key to the REDCap-schema column it
# overrides when the decision picks "source" instead of "arc" for that
# field, applied directly to the already-built REDCap rows.
_OVERRIDE_REDCAP_COLUMNS = {
    "form_name": _FORM_COLUMN,
    "question": "Field Label",
    "options": _CHOICES_COLUMN,
    "field_type": "Field Type",
    "validation": "Text Validation Type OR Show Slider Number",
    "section": _SECTION_COLUMN,
    "branching_logic": _BRANCHING_LOGIC_COLUMN,
    "field_note": "Field Note",
    "identifier": "Identifier?",
    "required_field": "Required Field?",
    "custom_alignment": "Custom Alignment",
    "question_number": "Question Number (surveys only)",
    "matrix_group": "Matrix Group Name",
    "matrix_ranking": "Matrix Ranking?",
    "field_annotation": _FIELD_ANNOTATION_COLUMN,
}

_DEFAULT_LISTS_PATH = "arc_data/Lists"


def _reorder_with_other_options(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each user_list/multi_list row followed by its _otherl2/_otherl3 rows."""
    new_rows = []
    used_indices = set()

    for index, row in df.iterrows():
        if index in used_indices:
            continue

        new_rows.append(row)
        used_indices.add(index)

        if row["Type"] in ("multi_list", "user_list"):
            prefix = "_".join(row["Variable"].split("_")[:2])
            for suffix in ("_otherl2", "_otherl3"):
                mask = df["Variable"].str.startswith(prefix + suffix)
                for i in df[mask].index:
                    new_rows.append(df.loc[i])
                    used_indices.add(i)

    return pd.DataFrame(new_rows)


def _custom_alignment(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Field Type"].isin(["checkbox", "radio"]) & (
        (df["Choices, Calculations, OR Slider Labels"].str.split("|").str.len() < 4)
        & (df["Choices, Calculations, OR Slider Labels"].str.len() <= 40)
    )
    df.loc[mask, "Custom Alignment"] = "RH"
    return df


def _build_core(rows: pd.DataFrame, lists_path: str = _DEFAULT_LISTS_PATH) -> pd.DataFrame:
    """Turn a batch of ARC-schema source rows into REDCap-schema rows.

    Column order/set is always `_REDCAP_COLUMNS` on the way out, regardless
    of how many rows come in (including zero).
    """
    if rows.empty:
        return pd.DataFrame(columns=_REDCAP_COLUMNS)

    # Keep any extra source columns (e.g. `List`) through the reorder step
    full_df = rows.fillna("").reset_index(drop=True)
    full_df = _reorder_with_other_options(full_df).reset_index(drop=True)
    df = full_df.reindex(columns=_SOURCE_COLUMNS).fillna("").reset_index(drop=True)

    # If the original rows included a `List` identifier for `user_list` rows,
    # read the corresponding CSVs from `lists_path` and populate the
    # `Answer Options` column with the pipe-separated values.
    if "List" in full_df.columns:
        list_vals = full_df["List"].fillna("").astype(str).reset_index(drop=True)
        unknown_synonyms = {
            "unknown",
            "dont know",
            "don't know",
            "not known",
            "n/a",
            "na",
            "missing",
        }
        for i, list_identifier in list_vals.items():
            if not list_identifier:
                continue
            try:
                folder, file_name = list_identifier.split("_", 1)
            except ValueError:
                folder = ""
                file_name = list_identifier

            file_path = os.path.join(lists_path, folder, f"{file_name}.csv")
            if os.path.exists(file_path):
                try:
                    list_df = pd.read_csv(file_path, dtype=str)
                    items = list_df.iloc[:, 0].dropna().astype(str).tolist()
                    if items:
                        enumerated = []
                        unknowns = []
                        code = 1
                        for it in items:
                            txt = str(it).strip()
                            if not txt:
                                continue
                            if txt.lower() in unknown_synonyms:
                                unknowns.append(txt)
                            else:
                                enumerated.append(f"{code}, {txt}")
                                code += 1
                        for u in unknowns:
                            enumerated.append(f"99, {u}")
                        if enumerated:
                            df.loc[i, "Answer Options"] = " | ".join(enumerated)
                except Exception:
                    pass

    df.loc[df["Type"] == "user_list", "Type"] = "radio"
    df.loc[df["Type"] == "multi_list", "Type"] = "checkbox"
    df.loc[df["Type"] == "list", "Type"] = "radio"

    df.columns = _CRF_COLUMNS
    # `reindex` fills the brand-new REDCap-only columns (Field Note,
    # Identifier?, Custom Alignment, ...) with NaN as float64. Force object
    # dtype and fill with "" so downstream string operations never see a
    # raw NaN, whether or not the caller ever sets those columns.
    df = df.reindex(columns=_REDCAP_COLUMNS).astype(object).fillna("")

    df.loc[df["Field Type"].isin(_TEXT_ONLY_TYPES), "Field Type"] = "text"
    df = df.loc[df["Field Type"].isin(_KEPT_FIELD_TYPES)]

    df.loc[
        df["Text Validation Type OR Show Slider Number"] == "units",
        "Text Validation Type OR Show Slider Number",
    ] = ""

    df = _custom_alignment(df)

    descriptive_mask = df["Field Type"] == "descriptive"
    df.loc[descriptive_mask, "Field Label"] = df.loc[
        descriptive_mask, "Field Label"
    ].apply(lambda label: _DESCRIPTIVE_LABEL_TEMPLATE.format(label=label))

    return df.reset_index(drop=True)


def _referenced_variables(text: str) -> set:
    if not isinstance(text, str):
        return set()
    return set(_VARIABLE_REFERENCE_RE.findall(text))


def _row_references(row: pd.Series) -> set:
    references = set()
    for column in _RENAMABLE_COLUMNS:
        references |= _referenced_variables(row.get(column, ""))
    return references


# --------------------------------------------------------------------------- #
# Per-decision row building (drives source-question ordering directly)
# --------------------------------------------------------------------------- #


def _source_field_value(decision: MatchDecision, key: str) -> str:
    """The source-question value for one of the overridable fields."""
    source = decision.source
    if key == "form_name":
        return source.form_name or ""
    if key == "question":
        return (
            source.question
            or decision.edited_translated_question
            or source.translated_question
            or ""
        )
    if key == "options":
        return source.options or ""
    if key == "field_type":
        return source.field_type or ""
    if key == "validation":
        return source.validation or ""
    if key == "section":
        return source.section or ""
    if key == "branching_logic":
        return source.branching_logic or ""
    if key == "field_note":
        return source.field_note or ""
    if key == "identifier":
        return source.identifier or ""
    if key == "required_field":
        return source.required_field or ""
    if key == "custom_alignment":
        return source.custom_alignment or ""
    if key == "question_number":
        return source.question_number or ""
    if key == "matrix_group":
        return source.matrix_group or ""
    if key == "matrix_ranking":
        return source.matrix_ranking or ""
    if key == "field_annotation":
        return source.field_annotation or ""
    return ""


def _apply_decision_overrides(built: pd.DataFrame, decision: MatchDecision) -> pd.DataFrame:
    """Overwrite ARC cells with the source value for any field this
    decision picked "source" for. `built` only ever holds this one
    decision's matched rows, so the override applies to all of them
    uniformly — no per-row variable matching needed.
    """
    if built.empty or not decision.field_overrides:
        return built
    built = built.copy()
    for key, column in _OVERRIDE_REDCAP_COLUMNS.items():
        if decision.field_overrides.get(key) == "source":
            built[column] = _source_field_value(decision, key)
    return built


def _decision_matched_frame(
    decision: MatchDecision, arc_catalog: pd.DataFrame, lists_path: str
) -> pd.DataFrame:
    """REDCap rows for a MATCHED/MATCHED_CREATED decision's matched ARC
    question(s), in the order `decision.matched_questions` lists them."""
    variables = [q.variable for q in decision.matched_questions if q.variable]
    if not variables:
        return pd.DataFrame(columns=_REDCAP_COLUMNS)

    matched = arc_catalog[arc_catalog["Variable"].isin(variables)]
    if matched.empty:
        return pd.DataFrame(columns=_REDCAP_COLUMNS)

    order = {variable: position for position, variable in enumerate(variables)}
    matched = (
        matched.assign(_order=matched["Variable"].map(order))
        .sort_values("_order")
        .drop(columns="_order")
    )

    built = _build_core(matched, lists_path)
    return _apply_decision_overrides(built, decision)


def _decision_created_frame(decision: MatchDecision, lists_path: str) -> pd.DataFrame:
    """REDCap row for a CREATED/MATCHED_CREATED decision's new question."""
    if decision.status not in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED):
        return pd.DataFrame(columns=_REDCAP_COLUMNS)

    source_row = pd.DataFrame(
        [
            {
                "Form": decision.new_form_name,
                "Section": decision.new_section,
                "Variable": decision.new_id,
                "Type": decision.new_field_type,
                "Question": decision.new_text,
                "Answer Options": decision.new_options,
                "Validation": decision.new_validation_type,
                "Minimum": decision.new_validation_min,
                "Maximum": decision.new_validation_max,
                "Skip Logic": decision.new_branching_logic,
            }
        ]
    )
    built = _build_core(source_row, lists_path)
    if built.empty:
        return built

    row_index = built.index[0]
    built.loc[row_index, "Field Note"] = decision.new_field_note
    built.loc[row_index, "Identifier?"] = decision.new_identifier
    built.loc[row_index, "Required Field?"] = decision.new_required_field
    built.loc[row_index, "Field Annotation"] = decision.new_field_annotation
    built.loc[row_index, "Matrix Group Name"] = decision.new_matrix_group_name
    built.loc[row_index, "Matrix Ranking?"] = decision.new_matrix_ranking
    built.loc[row_index, "Question Number (surveys only)"] = (
        decision.new_question_number
    )
    if decision.new_custom_alignment:
        built.loc[row_index, "Custom Alignment"] = decision.new_custom_alignment

    return built


def _standalone_created_frame(
    question: StandaloneQuestion, lists_path: str
) -> pd.DataFrame:
    """REDCap row for a standalone question added outside the source CSV."""
    source_row = pd.DataFrame(
        [
            {
                "Form": question.new_form_name,
                "Section": question.new_section,
                "Variable": question.new_id or question.st_id,
                "Type": question.new_field_type,
                "Question": question.new_text,
                "Answer Options": question.new_options,
                "Validation": question.new_validation_type,
                "Minimum": question.new_validation_min,
                "Maximum": question.new_validation_max,
                "Skip Logic": question.new_branching_logic,
            }
        ]
    )
    built = _build_core(source_row, lists_path)
    if built.empty:
        return built

    row_index = built.index[0]
    built.loc[row_index, "Field Note"] = question.new_field_note
    built.loc[row_index, "Identifier?"] = question.new_identifier
    built.loc[row_index, "Required Field?"] = question.new_required_field
    built.loc[row_index, "Field Annotation"] = question.new_field_annotation
    built.loc[row_index, "Matrix Group Name"] = question.new_matrix_group_name
    built.loc[row_index, "Matrix Ranking?"] = question.new_matrix_ranking
    built.loc[row_index, "Question Number (surveys only)"] = (
        question.new_question_number
    )
    if question.new_custom_alignment:
        built.loc[row_index, "Custom Alignment"] = question.new_custom_alignment

    return built


def _ordered_dictionary_rows(
    decisions: list[MatchDecision],
    arc_catalog: pd.DataFrame,
    lists_path: str,
    standalone_questions: list[StandaloneQuestion] | None = None,
) -> pd.DataFrame:
    """Every decision's row(s), concatenated in source (decision) order.

    Ignored/pending decisions contribute nothing. Standalone questions are
    inserted at the position chosen by the user (`after_source_index`).
    A MATCHED_CREATED decision contributes its matched row(s) immediately
    followed by its newly-created row, keeping the two adjacent in the output.
    """
    standalone_questions = standalone_questions or []
    frames: list[pd.DataFrame] = []
    most_recent_variable = ""

    for kind, item in iter_ordered_items(decisions, standalone_questions):
        if kind == "standalone":
            frame = _standalone_created_frame(item, lists_path)
            if not frame.empty and most_recent_variable:
                # Keep the selected "after question" relationship available
                # through the later block and dependency ordering passes.
                frame = frame.copy()
                frame["_placement_anchor"] = most_recent_variable
        else:
            decision = item
            if decision.status not in (
                MatchStatus.MATCHED,
                MatchStatus.CREATED,
                MatchStatus.MATCHED_CREATED,
            ):
                continue
            matched_frame = _decision_matched_frame(decision, arc_catalog, lists_path)
            created_frame = _decision_created_frame(decision, lists_path)
            for frame in (matched_frame, created_frame):
                if not frame.empty:
                    if frame is created_frame and most_recent_variable:
                        frame = frame.copy()
                        frame["_placement_anchor"] = most_recent_variable
                    frames.append(frame)
                    most_recent_variable = frame.iloc[-1][_FIELDNAME_COLUMN]
            continue

        if not frame.empty:
            frames.append(frame)
            most_recent_variable = frame.iloc[-1][_FIELDNAME_COLUMN]

    if not frames:
        return pd.DataFrame(columns=_REDCAP_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Missing branching-logic dependencies
# --------------------------------------------------------------------------- #


def _insert_missing_branching_logic_rows(
    df: pd.DataFrame, arc_catalog: pd.DataFrame, lists_path: str
) -> pd.DataFrame:
    """Add missing referenced ARC rows after the initial block ordering.

    Their original ARC form and section are retained; the caller lays them
    out by the raw catalog afterwards. Existing rows are never changed.
    """
    if arc_catalog.empty:
        return df

    arc_rows_by_variable = (
        arc_catalog.reindex(columns=_SOURCE_COLUMNS)
        .dropna(subset=["Variable"])
        .drop_duplicates(subset="Variable", keep="first")
        .set_index("Variable")
    )

    known = set(df[_FIELDNAME_COLUMN])
    added_rows: list[pd.Series] = []

    def insert_dependencies(row: pd.Series) -> None:
        for variable in _row_references(row):
            if variable in known or variable not in arc_rows_by_variable.index:
                continue
            known.add(variable)

            dependency = _build_core(
                arc_rows_by_variable.loc[[variable]].reset_index(), lists_path
            )
            if dependency.empty:
                continue

            dep_row = dependency.iloc[0].copy()
            insert_dependencies(dep_row)
            added_rows.append(dep_row)

    for _, row in df.iterrows():
        insert_dependencies(row)

    return (
        pd.concat([df, pd.DataFrame(added_rows)], ignore_index=True)
        if added_rows
        else df
    )


def _catalog_layout(
    arc_catalog: pd.DataFrame,
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """Read the form, section, and variable order directly from raw ARC."""
    forms: list[str] = []
    sections: dict[str, list[str]] = defaultdict(list)
    variable_order: dict[str, int] = {}
    for index, row in arc_catalog.reset_index(drop=True).iterrows():
        form = "" if pd.isna(row.get("Form", "")) else str(row.get("Form", ""))
        section = "" if pd.isna(row.get("Section", "")) else str(row.get("Section", ""))
        variable = "" if pd.isna(row.get("Variable", "")) else str(row.get("Variable", ""))
        if form not in forms:
            forms.append(form)
        if section not in sections[form]:
            sections[form].append(section)
        if variable:
            variable_order.setdefault(variable, index)
    return forms, sections, variable_order


def _reorder_by_arc_catalog(df: pd.DataFrame, arc_catalog: pd.DataFrame) -> pd.DataFrame:
    """Lay out rows by raw ARC form/section blocks and variable index.

    Rows not present in ARC are kept directly after the prior selected row
    when that row is in the same block. A new section is instead appended as
    its own final block in the known form; an entirely new form is appended
    after all ARC forms.
    """
    if df.empty:
        return df
    forms, sections_by_form, variable_order = _catalog_layout(arc_catalog)
    rows = df.copy().reset_index(drop=True)
    rows["_source_rank"] = range(len(rows))
    rows["_form"] = rows[_FORM_COLUMN].fillna("").astype(str)
    rows["_section"] = rows[_SECTION_COLUMN].fillna("").astype(str)

    present_forms = list(dict.fromkeys(rows["_form"]))
    form_order = forms + [form for form in present_forms if form not in forms]
    output: list[pd.Series] = []

    for form in form_order:
        form_rows = rows[rows["_form"] == form]
        if form_rows.empty:
            continue
        present_sections = list(dict.fromkeys(form_rows["_section"]))
        section_order = sections_by_form.get(form, []) + [
            section for section in present_sections
            if section not in sections_by_form.get(form, [])
        ]
        for section in section_order:
            block = form_rows[form_rows["_section"] == section]
            if block.empty:
                continue
            known = block[block[_FIELDNAME_COLUMN].isin(variable_order)].copy()
            new = block[~block[_FIELDNAME_COLUMN].isin(variable_order)].copy()
            known = known.sort_values(
                _FIELDNAME_COLUMN,
                key=lambda values: values.map(variable_order),
                kind="stable",
            )
            if new.empty:
                output.extend(row for _, row in known.iterrows())
                continue

            # The initial frame order already encodes a created question's
            # attached match and a standalone question's selected "after"
            # question. Preserve that anchor only inside this exact block.
            anchors: dict[str, list[pd.Series]] = defaultdict(list)
            tail: list[pd.Series] = []
            for _, new_row in new.sort_values("_source_rank", kind="stable").iterrows():
                prior = block[
                    (block["_source_rank"] < new_row["_source_rank"])
                    & block[_FIELDNAME_COLUMN].isin(variable_order)
                ]
                if prior.empty:
                    tail.append(new_row)
                else:
                    anchor = prior.iloc[-1][_FIELDNAME_COLUMN]
                    anchors[anchor].append(new_row)
            for _, known_row in known.iterrows():
                output.append(known_row)
                output.extend(anchors.get(known_row[_FIELDNAME_COLUMN], []))
            output.extend(tail)

    return pd.DataFrame(output).drop(
        columns=["_source_rank", "_form", "_section"], errors="ignore"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Branching-logic dependency ordering
# --------------------------------------------------------------------------- #


def _dependency_graph(df: pd.DataFrame) -> dict[str, set[str]]:
    """Map each row's Variable / Field Name to the *other* row variables it
    references (via branching logic, calc choices, or field annotation)
    that must therefore end up placed before it.

    Only references to variables that actually have a row in `df` count —
    a dangling reference (already reported elsewhere, or left as-is by
    design) has nothing to order against.
    """
    known_variables = set(df[_FIELDNAME_COLUMN])
    dependencies: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        variable = row[_FIELDNAME_COLUMN]
        referenced = _row_references(row) & known_variables
        referenced.discard(variable)
        dependencies[variable] = referenced
    return dependencies


def _ensure_dependency_order(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Move dependencies before dependants only within a form/section block.

    Cross-block dependencies cannot be moved without breaking the ARC form
    and section layout, so their original positions are retained and a clear
    warning is returned for export.
    """
    if df.empty:
        return df, []

    variables_to_block = {
        row[_FIELDNAME_COLUMN]: (row[_FORM_COLUMN], row[_SECTION_COLUMN])
        for _, row in df.iterrows()
    }
    warnings: list[str] = []
    warned: set[tuple[str, str]] = set()
    output: list[pd.Series] = []

    for _, block in df.groupby([_FORM_COLUMN, _SECTION_COLUMN], sort=False, dropna=False):
        rows_by_variable = {
            row[_FIELDNAME_COLUMN]: row for _, row in block.iterrows()
        }
        dependencies: dict[str, set[str]] = {}
        for variable, row in rows_by_variable.items():
            same_block: set[str] = set()
            placement_anchor = row.get("_placement_anchor", "")
            if placement_anchor in rows_by_variable and placement_anchor != variable:
                # This is the saved attachment/"after" choice for a newly
                # created or standalone question, not a branching reference.
                same_block.add(placement_anchor)
            for referenced in _row_references(row):
                target_block = variables_to_block.get(referenced)
                if target_block is None or referenced == variable:
                    continue
                if target_block == (row[_FORM_COLUMN], row[_SECTION_COLUMN]):
                    same_block.add(referenced)
                else:
                    warning_key = (variable, referenced)
                    if warning_key not in warned:
                        warnings.append(
                            f"Question '{variable}' references '{referenced}' in a "
                            "different form or section; it was not moved."
                        )
                        warned.add(warning_key)
            dependencies[variable] = same_block

        placed: list[str] = []
        placed_set: set[str] = set()
        pending = list(rows_by_variable)
        while pending:
            ready = [v for v in pending if dependencies[v] <= placed_set]
            if not ready:
                # A cycle remains in its already-ordered ARC position.
                placed.extend(pending)
                break
            placed.extend(ready)
            placed_set.update(ready)
            pending = [v for v in pending if v not in ready]
        output.extend(rows_by_variable[variable] for variable in placed)

    return pd.DataFrame(output).reset_index(drop=True), warnings


def _dedupe_section_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Blank out a Section Header everywhere except the first row it appears on.

    Must run last, once rows have their final order and text (including
    translation) are settled. Keeps only the very first occurrence of each
    section value — not just consecutive repeats — so REDCap prints each
    header once per block even if a section were to resurface later.
    """
    df = df.copy()
    section = df[_SECTION_COLUMN]
    keep = (section != "") & ~df.duplicated([_FORM_COLUMN, _SECTION_COLUMN])
    df[_SECTION_COLUMN] = section.where(keep, "")
    return df


def _drop_duplicate_fieldnames(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first row for each Variable / Field Name."""
    return df.drop_duplicates(subset=_FIELDNAME_COLUMN, keep="first").reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- #
# Variable renaming (source variable -> whatever it ended up exported as)
# --------------------------------------------------------------------------- #


def build_variable_rename_map(decisions: list[MatchDecision]) -> dict[str, str]:
    """Map each source question's *original* variable name to whatever name
    it ended up with in the export/data dictionary, for every decision
    where that name actually changed.

    - MATCHED (exactly one candidate): `source.variable` -> the matched
      ARC row's `variable`. Decisions matched to zero or several
      candidates are skipped — there's no single unambiguous new name to
      point references at.
    - CREATED / MATCHED_CREATED: `source.variable` -> `decision.new_id`.
    """
    rename_map: dict[str, str] = {}
    for decision in decisions:
        old_name = decision.source.variable
        if not old_name:
            continue

        if decision.status == MatchStatus.MATCHED:
            matched = decision.matched_questions
            if len(matched) == 1 and matched[0].variable and matched[0].variable != old_name:
                rename_map[old_name] = matched[0].variable
        elif decision.status in (MatchStatus.CREATED, MatchStatus.MATCHED_CREATED):
            if decision.new_id and decision.new_id != old_name:
                rename_map[old_name] = decision.new_id

    return rename_map


def rename_variable_references(text: str, rename_map: dict[str, str]) -> str:
    """Replace `[old_var]` / `[old_var(...)]` references in a branching-logic
    or calculation formula with the renamed variable from `rename_map`.

    Only the variable-name token inside the brackets is swapped — nothing
    else in the formula is touched.
    """
    if not text or not rename_map:
        return text

    def _sub(match: re.Match) -> str:
        variable, suffix = match.group(1), match.group(2)
        return f"[{rename_map.get(variable, variable)}{suffix}"

    return _VARIABLE_REFERENCE_WITH_SUFFIX_RE.sub(_sub, text)


def _apply_variable_renames(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    if not rename_map:
        return df
    df = df.copy()
    for column in _RENAMABLE_COLUMNS:
        df[column] = df[column].apply(
            lambda text: rename_variable_references(text, rename_map)
        )
    return df


def _translation_override_skip_sets(
    decisions: list[MatchDecision],
) -> tuple[set[str], set[str]]:
    """Matched ARC variables whose Field Label / Choices a mixed-match
    decision explicitly kept from the *source* CSV, for every matched
    question of that decision — those must be excluded from
    `apply_translation`, which only ever carries ARC's own wording.
    """
    skip_label = set()
    skip_choices = set()
    for decision in decisions:
        for matched in decision.matched_questions:
            if decision.field_overrides.get("question") == "source":
                skip_label.add(matched.variable)
            if decision.field_overrides.get("options") == "source":
                skip_choices.add(matched.variable)
    return skip_label, skip_choices


def available_field_types(arc_catalog_df: pd.DataFrame) -> list[str]:
    """Field types to offer for a new/overridden question, grounded in what
    the ARC catalog actually uses. Falls back to the full kept list if the
    catalog has nothing usable, so the dropdown is never empty.
    """
    if (
        arc_catalog_df is None
        or arc_catalog_df.empty
        or "Type" not in arc_catalog_df.columns
    ):
        return list(_KEPT_FIELD_TYPES)

    seen = set(arc_catalog_df["Type"].dropna().astype(str))
    normalized = {
        "radio"
        if t in ("user_list", "list")
        else "checkbox"
        if t == "multi_list"
        else t
        for t in seen
    }
    kept = sorted(normalized & set(_KEPT_FIELD_TYPES))
    return kept or list(_KEPT_FIELD_TYPES)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_data_dictionary(
    arc_catalog: pd.DataFrame,
    decisions: list[MatchDecision],
    translation: pd.DataFrame | None = None,
    lists_path: str = _DEFAULT_LISTS_PATH,
    standalone_questions: list[StandaloneQuestion] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the data dictionary in raw ARC form/section/variable order.

    The returned list contains only cross-block branching warnings; it does
    not block export.
    """
    df = _ordered_dictionary_rows(
        decisions, arc_catalog, lists_path, standalone_questions
    )
    if df.empty:
        return pd.DataFrame(columns=_REDCAP_COLUMNS), []

    df = _drop_duplicate_fieldnames(df)

    # First establish every known form/section block from raw ARC, then make
    # dependency moves only where they do not cross a block boundary.
    df = _reorder_by_arc_catalog(df, arc_catalog)
    df, warnings = _ensure_dependency_order(df)

    # Dependencies are deliberately fetched only after the user-selected
    # questions have been ordered. Re-laying out the augmented set preserves
    # the ARC form/section/variable index for each fetched row.
    df = _insert_missing_branching_logic_rows(df, arc_catalog, lists_path)
    df = _drop_duplicate_fieldnames(df)
    df = _reorder_by_arc_catalog(df, arc_catalog)
    df, dependency_warnings = _ensure_dependency_order(df)
    warnings.extend(warning for warning in dependency_warnings if warning not in warnings)

    rename_map = build_variable_rename_map(decisions)
    df = _apply_variable_renames(df, rename_map)

    if translation is not None:
        skip_label, skip_choices = _translation_override_skip_sets(decisions)
        df = apply_translation(df, translation, skip_label, skip_choices)

    df = _dedupe_section_headers(df)

    df = df.reindex(columns=_REDCAP_COLUMNS).fillna("")
    return df, warnings
