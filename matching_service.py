"""Service that matches questions against a reference catalog.

Uses hybrid retrieval (BM25 + dense embeddings, fused with Reciprocal Rank
Fusion — see `retrieve_functions.hybrid_retrieve`) instead of pairwise text
similarity.

IMPORTANT: this service does NOT build the ChromaDB collections or the BM25
index itself. Building them is the expensive part (loading the embedding
model, embedding the whole catalog, tokenizing + indexing for BM25) and must
only happen ONCE per app run — that's done in `app.py`
(`_create_collections_and_index`, wrapped in `st.cache_resource`). This
class just receives the already-built resources and turns a `Question` into
a ranked list of `MatchCandidate`.
"""
from typing import List, Optional, Set

from models import MatchCandidate, Question
from retrieve_functions import hybrid_retrieve


class QuestionMatchingService:
    """Finds the best reference candidates for a given source question."""

    def __init__(self, reference: List[Question], collection_questions, collection_ques_def,
                 documents: List[str], ids: List[str], bm25_retriever, stemmer):
        """
        Parameters
        ----------
        reference : the reference catalog as `Question` objects, built from the
            SAME `df_expanded` dataframe used to build the collections/index
            below (see `vector_db.create_chromadb_collections`), so that
            `reference[i]` corresponds to doc id `ids[i]`. Building it from the
            original, non-expanded reference dataframe instead would misalign
            every result once any question got expanded into list items.
        collection_questions, collection_ques_def : the two pre-built Chroma collections.
        documents, ids : shared document text / id lists used to build the collections and BM25.
        bm25_retriever, stemmer : the pre-built BM25 index and its stemmer.
        """
        self._reference = reference
        self._collection_questions = collection_questions
        self._collection_ques_def = collection_ques_def
        self._documents = documents
        self._ids = ids
        self._bm25 = bm25_retriever
        self._stemmer = stemmer
        self._id_to_index = {doc_id: i for i, doc_id in enumerate(ids)}

    def find_candidates(self, source: Question, top_n: int = 5,
                         override_question: Optional[str] = None,
                         override_definition: Optional[str] = None,
                         allowed_row_indices: Optional[Set[int]] = None) -> List[MatchCandidate]:
        """Return up to `top_n` candidates sorted by descending score.

        Priority order for the comparison text (question and definition
        resolved independently):
        1. `override_question` / `override_definition`: manually edited by
           the user (allows recalculating without depending on the
           automatic translation).
        2. `source.translated_question` / `source.translated_definition`.
        3. `source.question` / `source.definition`, if no translation was made.

        Parameters
        ----------
        top_n : how many ranked candidates to return. The UI can call this
            with a larger `top_n` (e.g. 50) and paginate the *display* of
            that already-computed list client-side ("load more"), rather
            than re-running retrieval for every page — see
            `app._render_question_flow`.
        allowed_row_indices : if given, only reference rows whose `row_index`
            is in this set are eligible to be returned as candidates (e.g. a
            user-picked filter on ARC columns like Section/Body System/Form).
            When `None`, no restriction is applied.

        Note on retrieval scope: hybrid retrieval always runs over the FULL
        reference catalog (`top_k=len(self._reference)`), regardless of
        `top_n` or whether `allowed_row_indices` is set. This keeps the
        expensive similarity computation (dense + BM25 + RRF fusion)
        identical every time — both the filter and the pagination are pure
        post-processing steps on the same already-ranked results, rather
        than something that reshapes the retrieval pool itself.
        """
        query_question = override_question or source.translated_question or source.question
        query_definition = (override_definition or source.translated_definition
                             or source.definition or "")

        query = f"{query_question}. {query_definition}".strip()

        if not self._reference:
            return []

        results = hybrid_retrieve(
            query=query,
            densecollections=[self._collection_ques_def, self._collection_questions],
            retrieverbm25=self._bm25,
            documents=self._documents,
            doc_ids=self._ids,
            stemmer=self._stemmer,
            top_k=len(self._reference),
        )

        candidates: List[MatchCandidate] = []
        for result in results:
            ref_index = self._id_to_index.get(result["id"])
            if ref_index is None or ref_index >= len(self._reference):
                continue
            if allowed_row_indices is not None and ref_index not in allowed_row_indices:
                continue
            ref = self._reference[ref_index]
            candidates.append(MatchCandidate(question=ref, score=result["normalized_score"]))
            if len(candidates) >= top_n:
                break  # `results` is already sorted by hybrid_retrieve, safe to stop early

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_n]
