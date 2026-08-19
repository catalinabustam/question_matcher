"""BM25 sparse retrieval index over the reference catalog (bm25s + stemming).

Ported from the exploratory `hybrid_search.ipynb` notebook. Building the
index (`create_bm25_retriever`) is only ever done once, from the standalone
`build_index.py` script, together with the ChromaDB collections (see
`vector_db.create_chromadb_collections`) — both are built over the same
`documents`/`ids` so they can be fused later with Reciprocal Rank Fusion
(`retrieve_functions.hybrid_retrieve`). The Streamlit app just loads the
already-built index at startup via `load_bm25_retriever`.
"""
from typing import List, Tuple

import bm25s
import Stemmer

BM25_INDEX_PATH = "arc_index_bm25"


def create_bm25_retriever(documents: List[str], index_path: str = BM25_INDEX_PATH):
    """Build a BM25 index over `documents` and persist it to `index_path`.

    Only ever called from `build_index.py` — see module docstring. Returns
    the retriever directly (rather than requiring a reload from disk) so
    the calling script can hand it straight to a hybrid-retrieve call if it
    ever needs to, but the persisted copy at `index_path` is what the app
    loads back later via `load_bm25_retriever`.

    Returns
    -------
    retriever : bm25s.BM25 — the indexed retriever, ready for `.retrieve(...)`.
    stemmer   : Stemmer.Stemmer — the same English stemmer used to tokenize the
                index, must be reused to tokenize queries too.
    """
    stemmer = Stemmer.Stemmer("english")

    corpus_tokens = bm25s.tokenize(documents, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    retriever.save(index_path, corpus=documents)

    return retriever, stemmer


def load_bm25_retriever(index_path: str = BM25_INDEX_PATH) -> Tuple[bm25s.BM25, Stemmer.Stemmer]:
    """Load a BM25 index previously built and persisted by `create_bm25_retriever`.

    This used to be unsafe to do from the app (a stale on-disk index could
    silently get paired with a `documents`/`ids` list from a different run).
    That risk is gone now that the index is only ever produced by
    `build_index.py`, in the same run that also persists the exact
    `df_expanded` the app derives its `documents`/`ids` from — see
    `vector_db.build_documents` / `vector_db.build_ids`.
    """
    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25.load(index_path, load_corpus=False)
    return retriever, stemmer


def bm25s_retrieve(query: str, retriever, doc_ids: List[str], stemmer,
                    top_k: int = 10) -> List[Tuple[str, float]]:
    """Retrieve top-k `(doc_id, score)` pairs using BM25 sparse search."""
    query_tokens = bm25s.tokenize(query, stemmer=stemmer, stopwords="en")
    k = max(1, min(top_k, len(doc_ids)))
    lexical_indices, scores = retriever.retrieve(query_tokens, k=k)
    return [(doc_ids[idx], float(score)) for idx, score in zip(lexical_indices[0], scores[0])]
