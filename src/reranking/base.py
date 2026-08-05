from abc import ABC, abstractmethod
from typing import List, Tuple
from src.metadata.schema import Document


class BaseReranker(ABC):
    """Abstract Base Class for Re-ranking retrieved chunks."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_n: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank retrieved (Document, initial_score) pairs and return top_n best."""
        pass
