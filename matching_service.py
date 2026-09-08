"""Adapter between the application models and the ARC hybrid-search package."""
import re
from typing import List, Optional, Set

from arc_hybrid_search import HybridSearchIndex
from models import MatchCandidate, Question

_OPTION_NOISE_RE = re.compile(r"\d+|[^\w\s]|\b(?:yes|no|unknown)\b", re.IGNORECASE)


def _clean_options_text(options: Optional[str]) -> str:
    """Clean a (possibly pipe-separated) options string for query use."""
    if not options:
        return ""
    cleaned_parts = [
        re.sub(r"\s+", " ", _OPTION_NOISE_RE.sub(" ", part)).strip()
        for part in options.split("|")
    ]
    return ", ".join(part for part in cleaned_parts if part)


def definition_with_options(source: Question) -> str:
    definition = source.translated_definition or source.definition or ""
    options = _clean_options_text(source.translated_options or source.options)
    if not options:
        return definition
    return f"{definition}. {options}".strip(". ")


class QuestionMatchingService:
    """Finds the best reference candidates for a given source question."""

    def __init__(self, reference: List[Question], hybrid_index: HybridSearchIndex,
                 allowed_row_indices: Optional[Set[int]] = None,
                 metadata_filter: Optional[dict[str, List[str]]] = None):
        """
        Parameters
        ----------
        reference : the expanded reference catalog as `Question` objects. Its
            list position must match the package's returned `row_index`.
        hybrid_index : the package-owned hybrid search index.
        """
        self._reference = reference
        self._hybrid_index = hybrid_index
        self._allowed_row_indices = allowed_row_indices
        self._metadata_filter = metadata_filter

    def find_candidates(self, source: Question, top_n: int = 5,
                         override_question: Optional[str] = None,
                         override_definition: Optional[str] = None,
                         allowed_row_indices: Optional[Set[int]] = None) -> List[MatchCandidate]:
        """Return up to `top_n` candidates sorted by descending score.

        Priority order for the comparison text (question and definition
        resolved independently):
        1. `override_question` / `override_definition`: manually edited by
           the user (allows recalculating without depending on the
           automatic translation). `override_definition`, when given, is
           used verbatim — it already includes the source question's
           cleaned answer options, since that's what the editable
           "Translated definition" box shows by default (see
           `definition_with_options`).
        2. `source.translated_question` / `definition_with_options(source)`.
        3. `source.question` / `source.definition`, if no translation was made
           (still with cleaned options appended to the definition half).

        Parameters
        ----------
        top_n : how many ranked candidates to return. The UI can call this
            with a larger `top_n` (e.g. 50) and paginate the *display* of
            that already-computed list client-side ("load more"), rather
            than re-running retrieval for every page — see
            `app._render_question_flow`.
        allowed_row_indices : if given, only reference rows whose `row_index`
            is in this set are eligible to be returned as candidates.
            When `None`, no restriction is applied.

        The package returns only the fields needed for retrieval, so its
        `row_index` is used to recover the complete application `Question`.
        """
        query_question = override_question or source.translated_question or source.question
        query_definition = (
            override_definition
            if override_definition is not None
            else definition_with_options(source)
        )

        query = f"{query_question}. {query_definition}".strip()

        if not self._reference:
            return []

        selected_rows = self._allowed_row_indices
        if allowed_row_indices is not None:
            selected_rows = (
                allowed_row_indices
                if selected_rows is None
                else selected_rows & allowed_row_indices
            )
        results = self._hybrid_index.retrieve(
            query=query,
            catalog="expanded",
            top_k=len(self._reference),
            metadata_filter=self._metadata_filter,
        )

        candidates: List[MatchCandidate] = []
        for result in results:
            ref_index = result["row_index"]
            if ref_index >= len(self._reference):
                continue
            if selected_rows is not None and ref_index not in selected_rows:
                continue

            ref = self._reference[ref_index]
            candidates.append(MatchCandidate(question=ref, score=result["score"]))
            if len(candidates) >= top_n:
                break

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_n]
