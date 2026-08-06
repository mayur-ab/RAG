import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.vector_store.base import BaseVectorStore
from src.metadata.schema import Document, ChunkMetadata


class MemoryVectorStore(BaseVectorStore):
    """Native Python & NumPy in-memory vector store with cosine similarity and filtering."""

    def __init__(self, persist_path: Optional[str] = None):
        self.documents: Dict[str, Document] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self._load_from_disk()

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        for doc, emb in zip(documents, embeddings):
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.documents[doc.id] = doc
            self.vectors[doc.id] = vec

        if self.persist_path:
            self._save_to_disk()

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        allowed_roles: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        if not self.vectors:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = []
        for doc_id, doc in self.documents.items():
            # Security / RBAC role check: admin role can access all docs, or match doc roles
            if allowed_roles is not None and "admin" not in allowed_roles:
                doc_roles = getattr(doc.metadata, "allowed_roles", ["user", "admin"])
                if not any(role in allowed_roles for role in doc_roles):
                    continue

            # Metadata key-value filtering
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    val = getattr(doc.metadata, k, doc.metadata.extra.get(k))
                    if val != v:
                        match = False
                        break
                if not match:
                    continue

            v_vec = self.vectors[doc_id]
            sim = float(np.dot(q_vec, v_vec))
            scores.append((doc, sim))

        # Sort descending by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def delete(self, document_ids: List[str]) -> int:
        deleted_count = 0
        to_delete = []
        for doc_id, doc in self.documents.items():
            if doc.metadata.document_id in document_ids or doc_id in document_ids:
                to_delete.append(doc_id)

        for d_id in to_delete:
            del self.documents[d_id]
            del self.vectors[d_id]
            deleted_count += 1

        if self.persist_path:
            self._save_to_disk()

        return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        dim = 0
        if self.vectors:
            first_key = next(iter(self.vectors))
            dim = len(self.vectors[first_key])
        return {
            "total_documents": len(self.documents),
            "vector_dimension": dim,
            "store_type": "MemoryVectorStore"
        }

    def get_all_documents(self) -> List[Document]:
        return list(self.documents.values())

    def get_ingest_record_by_source(self, source: str) -> Optional[Dict[str, Any]]:
        from src.ingestion.fingerprint import normalize_source_path

        normalized = normalize_source_path(source)
        matches = [
            doc
            for doc in self.documents.values()
            if normalize_source_path(doc.metadata.source) == normalized
            or (doc.metadata.source_path or "") == normalized
        ]
        if not matches:
            return None
        meta = matches[0].metadata
        return {
            "document_id": meta.document_id,
            "ingest_fingerprint": meta.ingest_fingerprint or "",
            "chunking_strategy": meta.chunking_strategy or "",
            "updated_at": meta.updated_at or "",
            "chunk_count": len(matches),
        }

    def delete_by_source(self, source: str) -> int:
        from src.ingestion.fingerprint import normalize_source_path

        normalized = normalize_source_path(source)
        to_delete = [
            doc_id
            for doc_id, doc in self.documents.items()
            if normalize_source_path(doc.metadata.source) == normalized
            or (doc.metadata.source_path or "") == normalized
        ]
        for doc_id in to_delete:
            del self.documents[doc_id]
            del self.vectors[doc_id]
        if to_delete and self.persist_path:
            self._save_to_disk()
        return len(to_delete)

    def list_indexed_sources(self) -> List[Dict[str, Any]]:
        from src.ingestion.fingerprint import normalize_source_path

        grouped: Dict[str, Dict[str, Any]] = {}
        for doc in self.documents.values():
            source = doc.metadata.source_path or doc.metadata.source or "unknown"
            normalized = normalize_source_path(source)
            if normalized not in grouped:
                grouped[normalized] = {
                    "source": normalized,
                    "title": doc.metadata.title or "",
                    "document_id": doc.metadata.document_id or "",
                    "chunk_count": 0,
                    "updated_at": doc.metadata.updated_at or "",
                    "ingest_fingerprint": doc.metadata.ingest_fingerprint or "",
                    "chunking_strategy": doc.metadata.chunking_strategy or "",
                    "status": "indexed",
                }
            grouped[normalized]["chunk_count"] += 1
        return sorted(grouped.values(), key=lambda item: item["source"].lower())

    def _save_to_disk(self):
        if not self.persist_path:
            return
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        data = {
            "documents": {k: v.model_dump() for k, v in self.documents.items()},
            "vectors": {k: v.tolist() for k, v in self.vectors.items()}
        }
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load_from_disk(self):
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.get("documents", {}).items():
                    self.documents[k] = Document(**v)
                for k, v in data.get("vectors", {}).items():
                    self.vectors[k] = np.array(v, dtype=np.float32)
        except Exception:
            pass
