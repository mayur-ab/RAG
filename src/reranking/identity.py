from typing import List, Tuple
from src.reranking.base import BaseReranker
from src.metadata.schema import Document


class IdentityReranker(BaseReranker):
    """Pass-through reranker using original retrieval scores."""

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_n: int = 5
    ) -> List[Tuple[Document, float]]:
        # Preserves existing order, slices top_n
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_n]
