# Question Matcher

A Streamlit app that matches questions from a source CSV against the
[ISARIC ARC](https://github.com/ISARICResearch/ARC) reference catalog, and
exports a REDCap-compatible data dictionary.

Matching is powered by [`arc-hybrid-search`](https://github.com/catalinabustam/arc-hybrid-search),
a standalone Python package that combines dense embeddings (ChromaDB) and
sparse retrieval (BM25) via Reciprocal Rank Fusion. This app only handles
the UI and decisions — all retrieval logic lives in that package.

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` installs `arc-hybrid-search` directly from GitHub. To
use a local development copy instead:

```bash
pip install -e ../arc-hybrid-search
```

Optionally, create a `.env` file with `DEEPL_API_KEY=...` to pre-fill the
DeepL translation field.

## Build the reference index (one-time)

The app never builds the index itself — that's a separate manual step so
starting the app is instant:

```bash
python build_index.py
```

This downloads the ARC catalog, builds the ChromaDB + BM25 index via
`arc-hybrid-search`, and saves everything to `arc_data/`. Re-run it any
time the ARC catalog should be refreshed, or from the sidebar's
"Create/Recreate ARC index" button.

## Run the app

```bash
streamlit run app.py
```

## Process

1. **Upload** the source CSV and map its columns (question, section,
   variable, answer type, etc.) to the REDCap data dictionary schema.
2. **Translate** (optional) — DeepL or a local Ollama model translates
   questions to English before matching.
3. **Match, question by question** — `arc-hybrid-search` returns ranked ARC
   candidates for each question. For each one, either:
   - pick one or more matching ARC questions (optionally mixing which
     fields come from ARC vs. the source CSV),
   - create a new question (ARC-style variable name auto-generated), or
   - ignore it (excluded from export).
4. **Save/resume progress** — download a JSON snapshot at any point and
   re-upload it later (with the same source CSV) to continue.
5. **Export** — a plain CSV of every decision, plus a REDCap data
   dictionary CSV in ARC form/section order, optionally translated via
   ARC-Translations.

## Architecture

| File                   | Responsibility                                                            |
|------------------------|----------------------------------------------------------------------------|
| `arc-hybrid-search`    | External package: ARC catalog download, ChromaDB + BM25 indexing, hybrid retrieval. |
| `build_index.py`       | CLI: builds the index via `arc-hybrid-search` and persists it to `arc_data/`. Run manually, not by the app. |
| `matching_service.py`  | Thin adapter between the app's `Question` model and `arc-hybrid-search`'s retrieval API. |
| `models.py`            | Domain entities (`Question`, `MatchDecision`, `StandaloneQuestion`, ...). |
| `rules.py`             | Fixed rules for naming/building a new question (ARC naming convention). |
| `redcap_validation.py` | Structural REDCap validation shared by the "create new" and mixed-match flows. |
| `datadictionary.py`    | Builds the REDCap data dictionary CSV in ARC form/section/variable order. |
| `csv_io.py`            | Loads source/reference CSVs into `Question`s; exports decisions to CSV.  |
| `progress_io.py`       | Save/restore an in-progress session as JSON.                              |
| `translate.py`         | DeepL / Ollama translation clients.                                       |
| `translations.py`      | Applies ARC-Translations text to the exported data dictionary.           |
| `app.py`               | Streamlit UI: upload, column mapping, matching flow, export.             |

To change the matching algorithm, edit `arc-hybrid-search` itself — this
app only consumes its public API via `matching_service.py`.
