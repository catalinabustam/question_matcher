"""Build the ChromaDB dense collections used for hybrid retrieval."""
import os

import chromadb
from chromadb.utils import embedding_functions
import pandas as pd


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

        if question_type == "user_list" and pd.notna(row["List"]):
            list_identifier = str(row["List"]).strip()

            folder, file_name = list_identifier.split("_", 1)
            file_path = os.path.join(lists_path, folder, f"{file_name}.csv")

            if os.path.exists(file_path):
                list_df = pd.read_csv(file_path)
                items = list_df.iloc[:, 0].dropna().astype(str).tolist()  # Get the first column as a list of strings

                # Duplicate the full row and update only the question column
                for item in items:
                    new_row = row.copy()
                    new_row["Question"] = f"{base_question},  {file_name}: {item}"
                    expanded_rows.append(new_row)
            else:
                print(f"Warning: List file '{file_path}' not found. Skipping expansion for this row.")
                expanded_rows.append(row)  # Keep the original row if the list file is missing
        else:
            expanded_rows.append(row)  # Keep the original row if not a list type or List column is NaN

    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def create_chromadb_collections(arc_df: pd.DataFrame, lists_path: str = "ARC_Lists/", model_name: str = None):
    """Build the two Chroma collections (question+definition, question-only).

    This is the expensive step (embedding-model load + embedding the whole
    catalog) and is meant to be called ONCE per app run — see
    `app._create_collections_and_index`, which wraps this in
    `st.cache_resource`.

    Built over the *expanded* reference catalog: every `user_list` question
    is exploded into one row per list item first (`create_expanded_arc_dataframe`),
    so the index actually contains the individual list options rather than
    just the generic "pick one from the list" question text.

    Returns
    -------
    collection_questions : Chroma collection indexed on "Question" text only.
    collection_ques_def  : Chroma collection indexed on "Question: ... Definition: ..." text.
    documents            : list[str] — the raw text indexed in collection_ques_def, in `ids` order.
    ids                  : list[str] — the doc IDs shared by BOTH collections. Pass this exact
                            list (and `documents`) to `bm25.create_bm25_retriever` too, so BM25
                            and the dense collections use the same IDs and can be fused with RRF.
    df_expanded          : the expanded ARC dataframe. Its row order matches `ids`/`documents`
                            exactly (row i <-> "arc_{i}"). Callers should build the reference
                            `Question` objects from THIS dataframe (not the original `arc_df`) —
                            otherwise reference rows won't line up with retrieval results whenever
                            any list expansion happened.
    """
    client = chromadb.PersistentClient(path="chromadb_data")

    df_expanded = create_expanded_arc_dataframe(arc_df, lists_path)

    # Clean empty fields to ensure empty strings instead of "NaN" text
    df_expanded = df_expanded.fillna("")

    # Generate the raw text as a combination of specific columns
    documents = []
    metadatas = []
    ids = []

    selected_columns = ['Question', 'Definition']
    metadata_columns = ['Form', 'Section', 'Question', 'Body System']

    for index, row in df_expanded.iterrows():
        row_text = " ".join([f"{val}. " for _, val in row[selected_columns].items()])
        documents.append(row_text)

        # Store individual column data as metadata for structural filtering later.
        # Guard against columns that may not exist in every ARC export.
        metadatas.append({col: row[col] for col in metadata_columns if col in row.index})

        # Generate unique IDs for each record entry — shared across BOTH Chroma
        # collections and the BM25 index, so results can be fused later.
        ids.append(f"arc_{index}")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    collection_ques_def = client.get_or_create_collection(
        name="ques_def_arc",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    # upsert (not add): the Chroma client is persistent on disk (`chromadb_data/`),
    # so re-running the app against the same reference catalog would otherwise
    # raise a "duplicate ID" error instead of just refreshing the collection.
    collection_ques_def.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # Create another collection using only the questions
    collection_questions = client.get_or_create_collection(
        name="arc_questions",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    collection_questions.upsert(ids=ids, documents=df_expanded['Question'].tolist(), metadatas=metadatas)

    return collection_questions, collection_ques_def, documents, ids, df_expanded
