"""Optional translated question/choice text from ARC-Translations.

ARC's own catalog (`ARC.csv`) is English-only. ARC-Translations
(https://github.com/ISARICResearch/ARC-Translations) tracks a translated
`ARCH.csv` per language and per ARC version, kept in sync with the ARC
catalog and keyed by the same `Variable` names.

`build_index.py` downloads one `ARCH.csv` per language (see
`download_translation`) alongside the ARC catalog itself, and persists each
to `TRANSLATIONS_DATA_DIR`. `datadictionary.build_data_dictionary` uses
`apply_translation` to swap the Field Label / Choices text of matched rows
into the user's chosen output language at export time.

Only `Question` (-> Field Label) and `Answer Options` (-> Choices) are
translated. `Answer Options` in `ARCH.csv` is only populated for genuine
inline choice text ("1, Yes | 2, No"), in the same "code, label" shape
REDCap expects — choices built from a `List` CSV (`user_list`/`multi_list`
rows, expanded at build time from `ARC_Lists/`) aren't covered by
ARC-Translations, so those rows keep their English choice text.

Newly created questions (no ARC `Variable`) aren't covered either — there's
nothing in ARC-Translations to look them up by — and stay in their original
language.
"""

import io
import re
from pathlib import Path

import pandas as pd
import requests

TRANSLATIONS_REPO_API = "https://api.github.com/repos/ISARICResearch/ARC-Translations"
TRANSLATIONS_RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ISARICResearch/ARC-Translations/"
    "refs/heads/main/{version}/{language}/ARCH.csv"
)
TRANSLATIONS_DATA_DIR = Path("translations_data")

# Used only if the GitHub API can't be reached (e.g. rate-limited) when
# resolving the latest version/languages — see `build_index.py`.
FALLBACK_VERSION = "ARCH1.5.0"
FALLBACK_LANGUAGES = ["French", "Portuguese", "Spanish"]

_VERSION_RE = re.compile(r"^ARCH(\d+)\.(\d+)\.(\d+)$")


def _list_repo_dirs(path: str) -> list[str]:
    """Directory names directly under `path` in the ARC-Translations repo."""
    response = requests.get(f"{TRANSLATIONS_REPO_API}/contents/{path}", timeout=(3.05, 10))
    response.raise_for_status()
    return [entry["name"] for entry in response.json() if entry["type"] == "dir"]


def latest_translations_version() -> str:
    """The most recent `ARCHx.y.z` version folder in ARC-Translations."""
    versions = [name for name in _list_repo_dirs("") if _VERSION_RE.match(name)]
    if not versions:
        raise RuntimeError("No ARCH version folders found in ARC-Translations.")
    return max(versions, key=lambda v: tuple(map(int, _VERSION_RE.match(v).groups())))


def translation_languages(version: str) -> list[str]:
    """Language folder names available for `version` (e.g. "Spanish")."""
    return _list_repo_dirs(version)


def download_translation(language: str, version: str) -> pd.DataFrame:
    """Download one language's `ARCH.csv` from ARC-Translations."""
    url = TRANSLATIONS_RAW_URL_TEMPLATE.format(version=version, language=language)
    response = requests.get(url, timeout=(3.05, 10))
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), dtype=str).fillna("")


def persist_translation(
    language: str, df: pd.DataFrame, data_dir: Path = TRANSLATIONS_DATA_DIR
) -> None:
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / f"{language}.csv", index=False)


def available_languages(data_dir: Path = TRANSLATIONS_DATA_DIR) -> list[str]:
    """Languages with a translation persisted by `build_index.py`, sorted."""
    if not data_dir.exists():
        return []
    return sorted(path.stem for path in data_dir.glob("*.csv"))


def load_translation(language: str, data_dir: Path = TRANSLATIONS_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / f"{language}.csv", dtype=str).fillna("")


def apply_translation(
    df: pd.DataFrame,
    translation: pd.DataFrame,
    skip_label_variables: set[str] = frozenset(),
    skip_choices_variables: set[str] = frozenset(),
) -> pd.DataFrame:
    """Overlay translated Field Label / Choices onto matched rows of `df`.

    Only rows whose `Variable / Field Name` exists in `translation` (an
    ARC-Translations `ARCH.csv`) are affected.

    `skip_label_variables` / `skip_choices_variables` exclude variables
    whose Field Label / Choices the user explicitly kept from the *source*
    CSV (a "use source" mixed-match override, see `app._render_question_flow`)
    instead of ARC's own text — the translation only ever carries ARC's own
    wording, so overlaying it there would silently discard that choice.
    """
    if translation.empty:
        return df

    by_variable = translation.drop_duplicates(subset="Variable", keep="first").set_index(
        "Variable"
    )
    df = df.copy()
    for row_index, variable in df["Variable / Field Name"].items():
        if variable not in by_variable.index:
            continue
        translated = by_variable.loc[variable]
        if variable not in skip_label_variables and translated["Question"]:
            df.loc[row_index, "Field Label"] = translated["Question"]
        if variable not in skip_choices_variables and translated["Answer Options"]:
            df.loc[row_index, "Choices, Calculations, OR Slider Labels"] = translated[
                "Answer Options"
            ]
    return df
