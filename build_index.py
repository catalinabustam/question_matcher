"""CLI to (re)build the ARC reference index consumed by the Streamlit app.

Downloads the ARC catalog, expands it, builds the two ChromaDB collections
and the BM25 sparse index, and persists everything to disk:

- ChromaDB collections -> `vector_db.CHROMADB_PATH` (chromadb_data/)
- BM25 index           -> `bm25.BM25_INDEX_PATH` (arc_index_bm25/)
- Raw + expanded ARC tables -> `vector_db.INDEX_DATA_DIR` (index_data/)
- ARC-Translations `ARCH.csv` per language -> `translations.TRANSLATIONS_DATA_DIR`
  (translations_data/), used by `datadictionary.build_data_dictionary` to
  translate the exported data dictionary — see `translations.py`.

Run this once, and again any time the ARC catalog needs to be refreshed:

    python build_index.py

`app.py` only ever *reads* these artifacts (see
`vector_db.load_chromadb_collections` and `bm25.load_bm25_retriever`), so it
starts instantly instead of re-embedding the whole catalog on every run.
"""
import argparse
import io

import pandas as pd
import requests

from bm25 import BM25_INDEX_PATH, create_bm25_retriever
from translations import (
    FALLBACK_LANGUAGES,
    FALLBACK_VERSION,
    download_translation,
    latest_translations_version,
    persist_translation,
    translation_languages,
)
from vector_db import (
    ARC_LISTS_PATH,
    EMBEDDING_MODEL,
    INDEX_DATA_DIR,
    create_chromadb_collections,
)

ARC_URL = "https://raw.githubusercontent.com/ISARICResearch/ARC/refs/heads/main/ARC.csv"


def download_arc_catalog(url: str = ARC_URL) -> pd.DataFrame:
    response = requests.get(url, timeout=(3.05, 10))
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def download_arc_translations() -> None:
    """Download and persist the latest ARC-Translations `ARCH.csv` per
    language, so `datadictionary.build_data_dictionary` can offer them as
    output-language options at export time — see `translations.py`.

    Falls back to the last known version/language list (`FALLBACK_VERSION`,
    `FALLBACK_LANGUAGES`) if the GitHub API can't be reached (e.g.
    rate-limited), since it's only used here to discover what's currently
    available, not to fetch the files themselves.
    """
    try:
        version = latest_translations_version()
    except requests.RequestException:
        version = FALLBACK_VERSION
    print(f"Using ARC-Translations version {version}.")

    try:
        languages = translation_languages(version)
    except requests.RequestException:
        languages = FALLBACK_LANGUAGES

    for language in languages:
        print(f"Downloading {language} translation...")
        try:
            translation_df = download_translation(language, version)
        except requests.RequestException as exc:
            print(f"  Skipping {language}: {exc}")
            continue
        persist_translation(language, translation_df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lists-path", default=ARC_LISTS_PATH,
                         help="Directory with the ARC_Lists CSVs (default: %(default)s)")
    parser.add_argument("--model-name", default=EMBEDDING_MODEL,
                         help="Sentence-transformers embedding model (default: %(default)s)")
    args = parser.parse_args()

    print("Downloading ARC reference catalog...")
    arc_df = download_arc_catalog()

    download_arc_translations()

    print("Building ChromaDB collections (this loads the embedding model)...")
    _, _, documents, _, df_expanded = create_chromadb_collections(
        arc_df, lists_path=args.lists_path, model_name=args.model_name)

    print("Building BM25 index...")
    create_bm25_retriever(documents, index_path=BM25_INDEX_PATH)

    print(f"Persisting reference tables to {INDEX_DATA_DIR}/...")
    INDEX_DATA_DIR.mkdir(exist_ok=True)
    arc_df.to_csv(INDEX_DATA_DIR / "arc_raw.csv", index=False)
    df_expanded.to_csv(INDEX_DATA_DIR / "arc_expanded.csv", index=False)

    print("Done. Start (or restart) the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
