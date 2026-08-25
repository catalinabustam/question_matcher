"""REDCap data-dictionary validation shared by the "create new" form and
the mixed-match (ARC vs. source) per-field override resolution in `app.py`.

Only structural rules REDCap itself would reject on import are enforced as
blocking errors. A branching-logic reference to an unknown variable is
reported as a warning only, since it may legitimately point at a variable
created later in the same session.
"""

import re
from collections.abc import Iterable, Mapping

REDCAP_VALIDATION_TYPES: frozenset[str] = frozenset(
    {
        "date_mdy",
        "date_dmy",
        "date_ymd",
        "datetime_mdy",
        "datetime_dmy",
        "datetime_ymd",
        "datetime_seconds_mdy",
        "datetime_seconds_dmy",
        "datetime_seconds_ymd",
        "time",
        "time_mm_ss",
        "email",
        "integer",
        "number",
        "number_1dp",
        "number_2dp",
        "phone",
        "zipcode",
        "alpha_only",
        "signature",
    }
)

_CHOICE_FIELD_TYPES = {"radio", "dropdown", "checkbox"}

_VARIABLE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VARIABLE_MAX_LEN = 26

_BRANCHING_LOGIC_VARIABLE_RE = re.compile(r"\[(\w+)")


def _try_parse_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def validate_record(
    fields: Mapping[str, str],
    existing_ids: set[str],
    available_field_types: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Validate one REDCap data-dictionary row.

    `fields` keys: variable, form_name, section, field_type, label, choices,
    validation_type, validation_min, validation_max, branching_logic.
    `existing_ids` must NOT include the row's own variable name, or it will
    always be reported as a duplicate of itself.

    Returns `(errors, warnings)` — non-empty `errors` should block Save.
    """
    errors: list[str] = []
    warnings: list[str] = []

    variable = (fields.get("variable") or "").strip()
    if not variable:
        errors.append("Variable/Field Name is required.")
    else:
        if not _VARIABLE_NAME_RE.match(variable):
            errors.append(
                f"Variable/Field Name '{variable}' must start with a lowercase "
                "letter and contain only lowercase letters, numbers, and underscores."
            )
        if len(variable) > _VARIABLE_MAX_LEN:
            errors.append(
                f"Variable/Field Name '{variable}' exceeds REDCap's "
                f"{_VARIABLE_MAX_LEN}-character limit."
            )
        if variable in existing_ids:
            errors.append(f"Variable/Field Name '{variable}' is already in use.")

    if not (fields.get("form_name") or "").strip():
        errors.append("Form Name is required.")
    if not (fields.get("label") or "").strip():
        errors.append("Field Label is required.")

    field_type = (fields.get("field_type") or "").strip()
    available = set(available_field_types)
    if not field_type:
        errors.append("Field Type is required.")
    elif field_type not in available:
        errors.append(f"'{field_type}' is not a valid Field Type.")

    choices = (fields.get("choices") or "").strip()
    if field_type in _CHOICE_FIELD_TYPES:
        if not choices:
            errors.append(f"Choices are required for field type '{field_type}'.")
        else:
            codes = []
            for part in (p.strip() for p in choices.split("|")):
                if not part:
                    continue
                if "," not in part:
                    errors.append(
                        f"Choice '{part}' must be formatted as 'code, label'."
                    )
                    continue
                code = part.split(",", 1)[0].strip()
                if not code:
                    errors.append(f"Choice '{part}' is missing a code.")
                codes.append(code)
            dupes = sorted({c for c in codes if codes.count(c) > 1})
            if dupes:
                errors.append(f"Duplicate choice code(s): {', '.join(dupes)}.")
    elif field_type == "slider" and choices:
        segments = [s for s in choices.split("|")]
        if len(segments) > 3:
            errors.append(
                "Slider labels must be at most 3 pipe-separated segments "
                "(left | middle | right)."
            )

    validation_type = (fields.get("validation_type") or "").strip()
    if validation_type and validation_type not in REDCAP_VALIDATION_TYPES:
        errors.append(f"'{validation_type}' is not a recognized Text Validation Type.")

    validation_min = (fields.get("validation_min") or "").strip()
    validation_max = (fields.get("validation_max") or "").strip()
    if validation_min and validation_max:
        parsed_min = _try_parse_number(validation_min)
        parsed_max = _try_parse_number(validation_max)
        if (
            parsed_min is not None
            and parsed_max is not None
            and parsed_min > parsed_max
        ):
            errors.append(
                f"Text Validation Min ({validation_min}) must be ≤ "
                f"Max ({validation_max})."
            )

    branching_logic = (fields.get("branching_logic") or "").strip()
    if branching_logic:
        referenced = set(_BRANCHING_LOGIC_VARIABLE_RE.findall(branching_logic))
        known = existing_ids | ({variable} if variable else set())
        unknown = sorted(referenced - known)
        if unknown:
            warnings.append(
                f"Branching logic references unknown variable(s): {', '.join(unknown)}."
            )

    return errors, warnings
