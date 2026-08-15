"""Build a REDCap-style data dictionary CSV from ARC catalog rows.

Ported directly from `generate.py` (`_generate_crf` / `_custom_alignment`,
plus the descriptive-label wrapping done in `on_generate_click`) — the
reference CRF-building logic already used elsewhere in the ISARIC tooling.
Kept as its own module so `app.py` can reuse it without depending on the
Dash `bridge` package that `generate.py` lives in.

On top of that ported logic, three REDCap-validity rules are enforced on
the final dictionary:

1. No two rows share the same `Variable / Field Name` (first one wins).
2. Each `Form Name` appears as a single contiguous block — a form can't
   start, stop, and then start again later.
3. Every variable referenced in someone else's branching logic exists as
   its own row, pulled from the *original* (non-expanded) ARC catalog if
   it isn't already present.

Input rows must come from the ARC catalog (same columns as `ARC.csv`:
`Form`, `Section`, `Variable`, `Type`, `Question`, `Answer Options`,
`Validation`, `Minimum`, `Maximum`, `Skip Logic`) — the same schema
`generate.py` expects.
"""
import os
import re

import numpy as np
import pandas as pd

_DESCRIPTIVE_LABEL_TEMPLATE = (
    '<div class="rich-text-field-label"><h5 style="text-align: center;">'
    '<span style="color: #236fa1;">{label}</span></h5></div>'
)

_KEPT_FIELD_TYPES = [
    "text", "notes", "radio", "dropdown", "calc", "file",
    "checkbox", "yesno", "truefalse", "descriptive", "slider",
]

_TEXT_ONLY_TYPES = ["date_dmy", "number", "integer", "datetime_dmy"]

_SOURCE_COLUMNS = [
    "Form", "Section", "Variable", "Type", "Question",
    "Answer Options", "Validation", "Minimum", "Maximum", "Skip Logic",
]

_CRF_COLUMNS = [
    "Form Name", "Section Header", "Variable / Field Name", "Field Type",
    "Field Label", "Choices, Calculations, OR Slider Labels",
    "Text Validation Type OR Show Slider Number", "Text Validation Min",
    "Text Validation Max", "Branching Logic (Show field only if...)",
]

_REDCAP_COLUMNS = [
    "Variable / Field Name", "Form Name", "Section Header", "Field Type",
    "Field Label", "Choices, Calculations, OR Slider Labels", "Field Note",
    "Text Validation Type OR Show Slider Number", "Text Validation Min",
    "Text Validation Max", "Identifier?", "Branching Logic (Show field only if...)",
    "Required Field?", "Custom Alignment", "Question Number (surveys only)",
    "Matrix Group Name", "Matrix Ranking?", "Field Annotation",
]

_SYMPT_DN4_ANNOTATION = (
    "@CALCTEXT(if([sympt_dn4_pain]='1',if([sympt_dn4_score]>=4,"
    "'Neuropathic pain','No neuropathic pain'),''))"
)

_FIELDNAME_COLUMN = "Variable / Field Name"
_FORM_COLUMN = "Form Name"
_SECTION_COLUMN = "Section Header"
_BRANCHING_LOGIC_COLUMN = "Branching Logic (Show field only if...)"

# Matches the variable name inside a branching-logic reference, e.g. pulls
# "pres_firstsym" out of both "[pres_firstsym]='1'" and the checkbox form
# "[pres_firstsym(88)]='1'" (word chars stop right before the "(").
_VARIABLE_REFERENCE_RE = re.compile(r"\[(\w+)")


def _reorder_with_other_options(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each user_list/multi_list row followed by its _otherl2/_otherl3 rows.

    Ported unchanged (aside from variable names) from `generate.py`'s
    `_generate_crf` reordering loop.
    """
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
    """Ported unchanged from `generate.py`'s `_custom_alignment`."""
    mask = df["Field Type"].isin(["checkbox", "radio"]) & (
        (df["Choices, Calculations, OR Slider Labels"].str.split("|").str.len() < 4)
        & (df["Choices, Calculations, OR Slider Labels"].str.len() <= 40)
    )
    df.loc[mask, "Custom Alignment"] = "RH"
    return df


def _build_core(rows: pd.DataFrame, lists_path: str = "ARC_Lists/") -> pd.DataFrame:
    """Turn a batch of ARC source rows into REDCap-schema rows.

    Ported from `generate.py`'s `_generate_crf`, minus the Section Header
    de-duplication step — that only makes sense once, across the *final*
    fully-ordered dictionary (see `_dedupe_section_headers`), since this
    function may run more than once per export (once for the matched rows,
    again for any branching-logic rows added afterwards).
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
        # synonyms that should be coded as 99 and placed at the end
        unknown_synonyms = {"unknown", "dont know", "don't know", "not known", "n/a", "na", "missing"}
        for i, list_identifier in list_vals.items():
            if not list_identifier:
                continue
            # Expect format like "folder_filename" (matches vector_db logic)
            try:
                folder, file_name = list_identifier.split("_", 1)
            except ValueError:
                # Fallback: treat the whole identifier as filename
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
                        # append unknowns with code 99 at the end
                        for u in unknowns:
                            enumerated.append(f"99, {u}")
                        if enumerated:
                            df.loc[i, "Answer Options"] = " | ".join(enumerated)
                except Exception:
                    # If reading fails, leave Answer Options unchanged
                    pass

    df.loc[df["Type"] == "user_list", "Type"] = "radio"
    df.loc[df["Type"] == "multi_list", "Type"] = "checkbox"
    df.loc[df["Type"] == "list", "Type"] = "radio"

    df.columns = _CRF_COLUMNS
    # `reindex` fills the brand-new REDCap-only columns (Field Note,
    # Identifier?, Custom Alignment, ...) with NaN as float64. Force object
    # dtype so later string assignments (e.g. "RH" below) don't raise.
    df = df.reindex(columns=_REDCAP_COLUMNS).astype(object)

    if "sympt_dn4_result" in df[_FIELDNAME_COLUMN].values:
        df.loc[
            df[_FIELDNAME_COLUMN] == "sympt_dn4_result", "Field Annotation"
        ] = _SYMPT_DN4_ANNOTATION

    df.loc[df["Field Type"].isin(_TEXT_ONLY_TYPES), "Field Type"] = "text"
    df = df.loc[df["Field Type"].isin(_KEPT_FIELD_TYPES)]

    df.loc[
        df["Text Validation Type OR Show Slider Number"] == "units",
        "Text Validation Type OR Show Slider Number",
    ] = np.nan

    df = _custom_alignment(df)

    descriptive_mask = df["Field Type"] == "descriptive"
    df.loc[descriptive_mask, "Field Label"] = df.loc[descriptive_mask, "Field Label"].apply(
        lambda label: _DESCRIPTIVE_LABEL_TEMPLATE.format(label=label)
    )

    return df


def _referenced_variables(branching_logic: str) -> set:
    return set(_VARIABLE_REFERENCE_RE.findall(branching_logic or ""))


def _add_missing_branching_logic_rows(df: pd.DataFrame, arc_catalog: pd.DataFrame) -> pd.DataFrame:
    """Add a row for every branching-logic variable missing from column A.

    Some fields' branching logic references variables that never made it
    into the dictionary as their own row. For each such variable, pull its
    row from the *original* (non-expanded) ARC catalog and append it —
    repeating, since a newly added row can itself reference further
    variables — until nothing new turns up.
    """
    if arc_catalog.empty:
        return df

    arc_rows_by_variable = (
        arc_catalog.reindex(columns=_SOURCE_COLUMNS)
        .dropna(subset=["Variable"])
        .drop_duplicates(subset="Variable", keep="first")
        .set_index("Variable")
    )

    # Tracks every variable we've already looked up, whether or not it
    # ended up producing a kept row, so a variable whose Field Type gets
    # filtered out by `_build_core` can't be re-attempted forever.
    resolved: set = set()

    while True:
        known = set(df[_FIELDNAME_COLUMN])
        referenced: set = set()
        for logic in df[_BRANCHING_LOGIC_COLUMN]:
            referenced |= _referenced_variables(logic)

        to_resolve = (referenced - known - resolved) & set(arc_rows_by_variable.index)
        if not to_resolve:
            return df

        resolved |= to_resolve
        missing_rows = arc_rows_by_variable.loc[sorted(to_resolve)].reset_index()
        added = _build_core(missing_rows)
        if not added.empty:
            df = pd.concat([df, added], ignore_index=True)


def _drop_duplicate_fieldnames(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first row for each Variable / Field Name."""
    return df.drop_duplicates(subset=_FIELDNAME_COLUMN, keep="first").reset_index(drop=True)


def _make_forms_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """Group rows so each Form Name appears as one contiguous block.

    REDCap requires forms to be sequential in the data dictionary — once a
    form's rows end, that form can't reappear later. Rows are grouped by
    first-appearance order (stable sort), so each form's own internal row
    order is preserved.
    """
    form_rank = {form: rank for rank, form in enumerate(dict.fromkeys(df[_FORM_COLUMN]))}
    return (
        df.assign(_form_rank=df[_FORM_COLUMN].map(form_rank))
        .sort_values("_form_rank", kind="stable")
        .drop(columns="_form_rank")
        .reset_index(drop=True)
    )


def _dedupe_section_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Blank out a Section Header when it repeats the row directly above it.

    Ported from `generate.py`'s `_generate_crf`. Must run last, once rows
    have their final order — adding branching-logic rows and grouping forms
    can both change which row now sits above which, so deduping any earlier
    would blank headers against the wrong neighbor.
    """
    df = df.copy()
    df[_SECTION_COLUMN] = df[_SECTION_COLUMN].where(
        df[_SECTION_COLUMN] != df[_SECTION_COLUMN].shift(), np.nan
    )
    df = df.fillna("")
    df[_SECTION_COLUMN] = df[_SECTION_COLUMN].replace({"": np.nan})
    return df


def build_data_dictionary(matched_rows: pd.DataFrame, arc_catalog: pd.DataFrame) -> pd.DataFrame:
    """Convert matched ARC catalog rows into a REDCap-style data dictionary.

    Parameters
    ----------
    matched_rows : ARC source rows (see `_SOURCE_COLUMNS`) for every
        matched question — typically taken from the *expanded* catalog used
        for retrieval.
    arc_catalog : the *original*, non-expanded ARC catalog (one row per
        `Variable`), used to look up rows for variables referenced only in
        someone else's branching logic (see `_add_missing_branching_logic_rows`).

    Mirrors `generate.py`'s `_generate_crf` + `_custom_alignment` and the
    descriptive-label wrapping from `on_generate_click`, plus three extra
    rules applied to the final result: no duplicate field names, forms
    grouped into sequential blocks, and every branching-logic variable
    present as its own row.
    """
    if matched_rows.empty:
        return pd.DataFrame(columns=_REDCAP_COLUMNS)

    df = _build_core(matched_rows)
    df = _add_missing_branching_logic_rows(df, arc_catalog)
    df = _drop_duplicate_fieldnames(df)
    df = _make_forms_sequential(df)
    df = _dedupe_section_headers(df)

    return df.fillna("")
