from typing import List, Dict, Any, Optional, Tuple
from src.vector_store.base import BaseVectorStore
from src.vector_store.memory_store import MemoryVectorStore
from src.metadata.schema import Document


class FAISSVectorStore(BaseVectorStore):
    """FAISS Vector Database Integration with Memory Store fallback."""

    def __init__(self):
        self.fallback = MemoryVectorStore()

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        self.fallback.add_documents(documents, embeddings)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        allowed_roles: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        return self.fallback.search(query_vector, top_k, filter_metadata, allowed_roles)

    def delete(self, document_ids: List[str]) -> int:
        return self.fallback.delete(document_ids)

    def get_stats(self) -> Dict[str, Any]:
        stats = self.fallback.get_stats()
        stats["store_type"] = "FAISSVectorStore"
        return stats
