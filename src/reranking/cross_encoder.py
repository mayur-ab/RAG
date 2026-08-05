from typing import List, Tuple
from src.reranking.base import BaseReranker
from src.metadata.schema import Document


class CrossEncoderReranker(BaseReranker):
    """SentenceTransformer CrossEncoder reranker (or BGE reranker)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = False

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_n: int = 5
    ) -> List[Tuple[Document, float]]:
        self._load()
        if not documents:
            return []

        if self._model:
            pairs = [[query, doc.content] for doc, _ in documents]
            scores = self._model.predict(pairs)
            scored_docs = [(documents[i][0], float(scores[i])) for i in range(len(documents))]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:top_n]
        else:
            # Fall back to identity reranker if sentence_transformers cross_encoder isn't available
            from src.reranking.identity import IdentityReranker
            return IdentityReranker().rerank(query, documents, top_n)
