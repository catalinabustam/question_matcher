"""Hybrid retrieval: BM25 (sparse) + ChromaDB (dense), fused with weighted
Reciprocal Rank Fusion (RRF).

Ported from the exploratory `hybrid_search.ipynb` notebook and wired to work
with the two dense collections and the bm25s index that are built once at
application startup (see `vector_db.py` and `bm25.py`).
"""
from typing import Dict, List, Optional, Set, Tuple

from bm25 import bm25s_retrieve


def dense_retrieve(query: str, collection, top_k: int = 10,
                   where: Optional[dict] = None) -> List[Tuple[str, float]]:
    """Retrieve top-k `(doc_id, similarity)` pairs from a ChromaDB collection."""
    results = collection.query(query_texts=[query], n_results=top_k, where=where)
    ids = results["ids"][0]
    distances = results["distances"][0]
    # Cosine distance -> similarity
    return [(doc_id, 1 - dist) for doc_id, dist in zip(ids, distances)]

def joint_dense_retrieve(
    query: str, 
    dense_collections: List,
    top_k: int = 5,
    where: Optional[dict] = None,
) -> List[Tuple[str, float]]:
    """
    Query two ChromaDB collections, keep the maximum score for each document ID,
    and return a single list of (doc_id, score) tuples matching the format:
    [(doc_id, 1 - dist), ...]
    """
    # 1. Fetch results from both collections using your existing function

    results = []
    for collection in dense_collections:
        if collection is None:
            raise ValueError("Collection cannot be None")
        results.append(dense_retrieve(query, collection=collection, top_k=top_k, where=where))
    
    # 2. Track the maximum score per document ID
    max_scores = {}
    for sublist in results:
        for item_id, score in sublist:
            max_scores[item_id] = max(max_scores.get(item_id, score), score)

    # 3. Sort by score descending
    sorted_results = sorted(max_scores.items(), key=lambda x: x[1], reverse=True)

    # 4. Return top_k list of (doc_id, score) tuples
    return sorted_results[:top_k]

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, Dict[str, float]]]:
    """Merge multiple ranked lists using Weighted Reciprocal Rank Fusion.

    RRF score = sum(weight * (1 / (k + rank))) across all lists.
    """

    rrf_scores: Dict[str, float] = {}
    
    for ranked in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # Calculate the theoretical maximum score (rank 1 across all lists)
    num_lists = len(ranked_lists)
    max_theoretical = num_lists * (1.0 / (k + 1.0))

    detailed_scores = {
        doc_id: {
            "rrf_score": score,
            "normalized_score": score / max_theoretical
        }
        for doc_id, score in rrf_scores.items()
    }

    return sorted(detailed_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)


def hybrid_retrieve(
    query: str, 
    densecollections: List,
    retrieverbm25=None, 
    stemmer=None,
    documents=None,
    doc_ids= None,
    top_k: int = 5,
    where: Optional[dict] = None,
    allowed_doc_ids: Optional[Set[str]] = None,
) -> List[Dict]:
    """Full hybrid retrieval: BM25 + dense via ChromaDB, fused with weighted RRF."""
    results = []
    
    # for collection in densecollections:
    #     if collection is None:
    #         raise ValueError("Collection cannot be None")
    #     results.append(dense_retrieve(query, collection=collection, top_k=len(documents)))
    
    search_size = len(allowed_doc_ids) if allowed_doc_ids is not None else len(documents)
    if search_size == 0:
        return []

    results = joint_dense_retrieve(
        query, dense_collections=densecollections, top_k=search_size, where=where
    )
     
    bm25s_results = bm25s_retrieve(
        query,
        retriever=retrieverbm25,
        doc_ids=doc_ids,
        stemmer=stemmer,
        top_k=search_size,
        allowed_doc_ids=allowed_doc_ids,
    )

    hybrid_results = [results] + [bm25s_results]

    documents_ids = dict(zip(doc_ids, documents))
    
    fused = reciprocal_rank_fusion(hybrid_results, k=10) 


    return [
        {"id": doc_id, "rrf_score": scores["rrf_score"], "normalized_score": scores["normalized_score"]}
        for doc_id, scores in fused[:top_k]
    ]
