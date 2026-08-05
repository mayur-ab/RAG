from typing import List, Dict, Any, Optional, Tuple
from src.metadata.schema import Document
from src.retrieval.vector_search import VectorSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from config.settings import settings


class HybridSearchEngine:
    """Hybrid Retrieval engine combining Dense Vector Search & Sparse BM25 Search."""

    def __init__(
        self,
        vector_engine: VectorSearchEngine,
        bm25_engine: BM25SearchEngine,
        alpha: float = settings.HYBRID_ALPHA,
        rrf_k: int = settings.RRF_K
    ):
        self.vector_engine = vector_engine
        self.bm25_engine = bm25_engine
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        allowed_roles: Optional[List[str]] = None,
        use_rrf: bool = True
    ) -> List[Tuple[Document, float]]:

        vector_results = self.vector_engine.search(
            query=query,
            top_k=top_k * 2,
            filter_metadata=filter_metadata,
            allowed_roles=allowed_roles
        )

        bm25_results = self.bm25_engine.search(
            query=query,
            top_k=top_k * 2,
            allowed_roles=allowed_roles
        )

        if not vector_results and not bm25_results:
            return []
        if not vector_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return vector_results[:top_k]

        if use_rrf:
            return self._rrf_fusion(vector_results, bm25_results, top_k)
        else:
            return self._weighted_fusion(vector_results, bm25_results, top_k)

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion algorithm: RRF_score = sum(1 / (k + rank))."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, (doc, _) in enumerate(vector_results):
            doc_map[doc.id] = doc
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, (doc, _) in enumerate(bm25_results):
            doc_map[doc.id] = doc
            rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs[:top_k]]

    def _weighted_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        top_k: int
    ) -> List[Tuple[Document, float]]:
        """Min-Max normalized weighted score combination."""
        doc_map: Dict[str, Document] = {}
        vec_scores: Dict[str, float] = {}
        bm_scores: Dict[str, float] = {}

        for doc, score in vector_results:
            doc_map[doc.id] = doc
            vec_scores[doc.id] = score

        for doc, score in bm25_results:
            doc_map[doc.id] = doc
            bm_scores[doc.id] = score

        # Min-max normalize vector scores
        v_min = min(vec_scores.values()) if vec_scores else 0.0
        v_max = max(vec_scores.values()) if vec_scores else 1.0
        v_diff = (v_max - v_min) if (v_max - v_min) > 0 else 1.0

        # Min-max normalize BM25 scores
        b_min = min(bm_scores.values()) if bm_scores else 0.0
        b_max = max(bm_scores.values()) if bm_scores else 1.0
        b_diff = (b_max - b_min) if (b_max - b_min) > 0 else 1.0

        final_scores: Dict[str, float] = {}
        for doc_id, doc in doc_map.items():
            norm_v = (vec_scores.get(doc_id, v_min) - v_min) / v_diff
            norm_b = (bm_scores.get(doc_id, b_min) - b_min) / b_diff
            combined = (self.alpha * norm_v) + ((1 - self.alpha) * norm_b)
            final_scores[doc_id] = combined

        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs[:top_k]]
