"""Build (and load) the ChromaDB dense collections used for hybrid retrieval.

Building the collections is the expensive step (loading the embedding
model, embedding the whole ARC catalog) and is meant to run exactly once,
from the standalone `build_index.py` script — see `create_chromadb_collections`.
The Streamlit app never builds anything itself; it only *loads* the
already-built collections at startup, via `load_chromadb_collections`.
"""
import os
import re
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd

# Shared locations/config, so `app.py` and `build_index.py` always agree on
# where the index lives without duplicating the constants in both places.
CHROMADB_PATH = "chromadb_data"
INDEX_DATA_DIR = Path("index_data")
ARC_LISTS_PATH = "ARC_Lists/"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

_QUESTIONS_COLLECTION = "arc_questions"
_QUES_DEF_COLLECTION = "ques_def_arc"

_TEXT_COLUMNS = ["Question", "Definition"]
_METADATA_COLUMNS = ["Form", "Section", "Question", "Body System"]


def create_expanded_arc_dataframe(arc_df: pd.DataFrame, lists_path: str) -> pd.DataFrame:
    """Expand the ARC DataFrame by replacing user list questions with individual items from the corresponding CSV files.

    Args:
        arc_df (pd.DataFrame): The original ARC DataFrame.
        lists_path (str): The path to the directory containing the list CSV files.

    Returns:
        pd.DataFrame: The expanded DataFrame with individual questions.
    """
    expanded_rows = []

    for _, row in arc_df.iterrows():
        question_type = str(row["Type"]).strip().lower()
        base_question = str(row["Question"]).strip()

        if question_type in ["user_list", "multilist"] and pd.notna(row["List"]):
            list_identifier = str(row["List"]).strip()

            folder, file_name = list_identifier.split("_", 1)
            file_path = os.path.join(lists_path, folder, f"{file_name}.csv")

            if os.path.exists(file_path):
                list_df = pd.read_csv(file_path)
                items = list_df.iloc[:, 0].dropna().astype(str).tolist()  # Get the first column as a list of strings

                # Duplicate the full row and update only the question column
                for item in items:
                    new_row = row.copy()
                    new_row["Question"] = f"{base_question}, {file_name}: {item}"
                    expanded_rows.append(new_row)
            else:
                print(f"Warning: List file '{file_path}' not found. Skipping expansion for this row.")
                expanded_rows.append(row)  # Keep the original row if the list file is missing
        elif question_type in ["radio", "checkbox"]:

            options = str(row["Answer Options"]).split('|')
            cleaned_options = [res for s in options
                               if (res := re.sub(r"\d+|,|unknown|yes|no", "", s, flags=re.IGNORECASE).strip())]

            options_to_add = ", Options: " + ", ".join(cleaned_options) if len(cleaned_options) > 0 else ''
            row["Question"] = row["Question"] + options_to_add

            expanded_rows.append(row)
        else:
            expanded_rows.append(row)  # Keep the original row if not a list type or List column is NaN

    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def build_documents(df_expanded: pd.DataFrame) -> List[str]:
    """Build the "Question. Definition. " text embedded/indexed for each row.

    A pure function of `df_expanded` alone, so `documents` never needs to be
    persisted on its own: both `create_chromadb_collections` (at build time)
    and the app (at load time, from the persisted `df_expanded` CSV) derive
    it from this single function and are guaranteed to agree.
    """
    return [
        " ".join(f"{value}. " for _, value in row[_TEXT_COLUMNS].items())
        for _, row in df_expanded.iterrows()
    ]


def build_ids(df_expanded: pd.DataFrame) -> List[str]:
    """Doc ids shared by both Chroma collections and the BM25 index.

    Purely positional (row i -> "arc_{i}"), so — like `build_documents` —
    they're derived the same way at build time and at load time instead of
    being persisted separately.
    """
    return [f"arc_{i}" for i in range(len(df_expanded))]


def _build_metadatas(df_expanded: pd.DataFrame) -> List[dict]:
    return [
        {col: row[col] for col in _METADATA_COLUMNS if col in row.index}
        for _, row in df_expanded.iterrows()
    ]


def create_chromadb_collections(arc_df: pd.DataFrame, lists_path: str = ARC_LISTS_PATH,
                                 model_name: str = EMBEDDING_MODEL):
    """Build (or refresh) the two Chroma collections from a raw ARC dataframe.

    This is the expensive step (embedding-model load + embedding the whole
    catalog). It's only ever called from `build_index.py` now — see the
    module docstring — never from the Streamlit app itself.

    Returns
    -------
    collection_questions : Chroma collection indexed on "Question" text only.
    collection_ques_def  : Chroma collection indexed on "Question: ... Definition: ..." text.
    documents            : list[str] — same as `build_documents(df_expanded)`, returned here
                            too so `build_index.py` can hand it straight to
                            `bm25.create_bm25_retriever` without recomputing.
    ids                  : list[str] — same as `build_ids(df_expanded)`.
    df_expanded          : the expanded ARC dataframe. Its row order matches `ids`/`documents`
                            exactly (row i <-> "arc_{i}"). This is the dataframe that gets
                            persisted to `index_data/arc_expanded.csv` for the app to load.
    """
    client = chromadb.PersistentClient(path=CHROMADB_PATH)

    df_expanded = create_expanded_arc_dataframe(arc_df, lists_path).fillna("")

    documents = build_documents(df_expanded)
    ids = build_ids(df_expanded)
    metadatas = _build_metadatas(df_expanded)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    collection_ques_def = client.get_or_create_collection(
        name=_QUES_DEF_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    # upsert (not add): re-running the build script against a refreshed
    # ARC catalog should refresh the collection in place, not fail with a
    # "duplicate ID" error.
    collection_ques_def.upsert(ids=ids, documents=documents, metadatas=metadatas)

    collection_questions = client.get_or_create_collection(
        name=_QUESTIONS_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    collection_questions.upsert(ids=ids, documents=df_expanded["Question"].tolist(), metadatas=metadatas)

    return collection_questions, collection_ques_def, documents, ids, df_expanded


def load_chromadb_collections(model_name: str = EMBEDDING_MODEL) -> Tuple:
    """Load the two Chroma collections previously built by `build_index.py`.

    Connects to the same on-disk persistent client/collections that
    `create_chromadb_collections` upserts into, without loading the
    embedding model to build or re-embed anything new (the embedding
    function is still needed to *query* the collections later, though).
    """
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    try:
        collection_questions = client.get_collection(
            name=_QUESTIONS_COLLECTION, embedding_function=embedding_fn)
        collection_ques_def = client.get_collection(
            name=_QUES_DEF_COLLECTION, embedding_function=embedding_fn)
    except Exception as exc:
        raise RuntimeError(
            "ChromaDB collections not found. Build the index first: python build_index.py"
        ) from exc
    return collection_questions, collection_ques_def
