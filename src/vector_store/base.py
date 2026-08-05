from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from src.metadata.schema import Document


class BaseVectorStore(ABC):
    """Abstract Base Class for Vector Database Abstraction."""

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents and corresponding embedding vectors to vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        allowed_roles: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for top_k similar documents returning (Document, similarity_score)."""
        pass

    @abstractmethod
    def delete(self, document_ids: List[str]) -> int:
        """Delete documents by document_id from store. Returns count of deleted records."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return collection stats (total vectors, dimensions, memory usage)."""
        pass
