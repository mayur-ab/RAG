from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract Base Class for swappable embedding generators."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text string."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of text strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return vector embedding dimension size."""
        pass
