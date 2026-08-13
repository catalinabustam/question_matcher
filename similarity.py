"""Text similarity strategies (Strategy pattern).

`difflib` from the standard library is used to keep the application very
lightweight (no external NLP/fuzzy-matching dependencies).
"""
from abc import ABC, abstractmethod
from difflib import SequenceMatcher


class SimilarityStrategy(ABC):
    """Contract for any text-comparison strategy."""

    @abstractmethod
    def score(self, a: str, b: str) -> float:
        """Return a similarity score between 0.0 and 1.0."""


class DifflibSimilarity(SimilarityStrategy):
    """Similarity based on `difflib.SequenceMatcher` (no dependencies)."""

    def score(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# Single configuration point: change this to use a different strategy
# (e.g. an embeddings-based one) without touching the rest of the app.
DEFAULT_STRATEGY: SimilarityStrategy = DifflibSimilarity()
