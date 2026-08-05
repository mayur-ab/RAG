from typing import List, Dict, Any, Optional, Tuple
from src.vector_store.base import BaseVectorStore
from src.embeddings.base import BaseEmbeddingProvider
from src.metadata.schema import Document


class VectorSearchEngine:
    """Dense Vector Similarity Search engine."""

    def __init__(self, vector_store: BaseVectorStore, embedding_provider: BaseEmbeddingProvider):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        allowed_roles: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        query_vector = self.embedding_provider.embed_text(query)
        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filter_metadata=filter_metadata,
            allowed_roles=allowed_roles
        )
