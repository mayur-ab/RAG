import re
from typing import List, Dict, Any, Optional, Tuple
from src.metadata.schema import Document


class BM25SearchEngine:
    """Sparse Keyword Search engine using BM25 scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self._bm25 = None
        self._corpus_tokens: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index(self, documents: List[Document]) -> None:
        self.documents = documents
        self._corpus_tokens = [self._tokenize(doc.content) for doc in documents]
        if not self._corpus_tokens:
            self._bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus_tokens, k1=self.k1, b=self.b)
        except Exception:
            self._bm25 = None

    def search(
        self,
        query: str,
        top_k: int = 10,
        allowed_roles: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = []
        if self._bm25:
            scores = self._bm25.get_scores(query_tokens).tolist()
        else:
            # Fallback term-frequency keyword matching score
            for tokens in self._corpus_tokens:
                score = sum(1.0 for q in query_tokens if q in tokens)
                scores.append(score)

        results = []
        for doc, score in zip(self.documents, scores):
            if allowed_roles is not None:
                doc_roles = getattr(doc.metadata, "allowed_roles", ["user", "admin"])
                if not any(role in allowed_roles for role in doc_roles):
                    continue

            if score > 0:
                results.append((doc, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
