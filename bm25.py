"""BM25 sparse retrieval index over the reference catalog (bm25s + stemming).

Ported from the exploratory `hybrid_search.ipynb` notebook. Built once (see
`vector_db.create_chromadb_collections` for the dense side, and
`app._create_collections_and_index` which wraps both in `st.cache_resource`
so this only runs once when the app starts) over the same `documents` list
used for the dense collections, so BM25 and dense results share the same
ids and can be fused later with Reciprocal Rank Fusion
(`retrieve_functions.hybrid_retrieve`).
"""
from typing import List, Tuple

import bm25s
import Stemmer


def create_bm25_retriever(documents: List[str], index_path: str = "arc_index_bm25"):
    """Build a BM25 index over `documents`.

    Returns
    -------
    retriever : bm25s.BM25 — the indexed retriever, ready for `.retrieve(...)`.
    stemmer   : Stemmer.Stemmer — the same English stemmer used to tokenize the
                index, must be reused to tokenize queries too.

    Note: the index is also persisted to `index_path` (as in the reference
    notebook), but it is always rebuilt *in memory* here rather than loaded
    back from disk. Reloading a stale on-disk index would silently pair BM25
    scores with a `documents`/`ids` list from a *different* run (e.g. a
    different reference CSV), which is worse than just rebuilding — building
    is fast, and this function is already only called once per app run via
    `st.cache_resource`.
    """
    stemmer = Stemmer.Stemmer("english")

    corpus_tokens = bm25s.tokenize(documents, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    if index_path:
        try:
            retriever.save(index_path, corpus=documents)
        except Exception:
            pass  # persistence to disk is a nice-to-have, not required for retrieval

    return retriever, stemmer


def bm25s_retrieve(query: str, retriever, doc_ids: List[str], stemmer,
                    top_k: int = 10) -> List[Tuple[str, float]]:
    """Retrieve top-k `(doc_id, score)` pairs using BM25 sparse search."""
    query_tokens = bm25s.tokenize(query, stemmer=stemmer, stopwords="en")
    k = max(1, min(top_k, len(doc_ids)))
    lexical_indices, scores = retriever.retrieve(query_tokens, k=k)
    return [(doc_ids[idx], float(score)) for idx, score in zip(lexical_indices[0], scores[0])]
