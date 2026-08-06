import os
import uuid
from typing import Any, Dict, List, Optional

from src.vector_store.chroma_store import _create_persistent_client

try:
    import chromadb  # noqa: F401
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class EpisodicMemoryStore:
    """Separate Chroma collection for conversation summaries and semantic user memories."""

    COLLECTION_NAME = "user_memory"

    def __init__(self, persist_directory: str):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is required for episodic memory.")
        self.persist_directory = os.path.abspath(persist_directory)
        self.client = _create_persistent_client(self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_memory(
        self,
        text: str,
        embedding: List[float],
        user_id: str,
        memory_type: str = "summary",
        memory_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        mid = memory_id or str(uuid.uuid4())
        metadata: Dict[str, Any] = {
            "user_id": user_id,
            "memory_type": memory_type,
            "created_at": created_at or "",
        }
        self.collection.add(
            ids=[mid],
            embeddings=[list(map(float, embedding))],
            documents=[text],
            metadatas=[metadata],
        )
        return mid

    def search(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else 1.0
            meta = metadatas[idx] if idx < len(metadatas) else {}
            hits.append(
                {
                    "id": doc_id,
                    "text": documents[idx] if idx < len(documents) else "",
                    "memory_type": meta.get("memory_type", "summary"),
                    "created_at": meta.get("created_at", ""),
                    "score": max(0.0, 1.0 - float(distance)),
                }
            )
        return hits

    def delete_user_memories(self, user_id: str) -> int:
        """Remove episodic summaries for one user. Does not touch the RAG knowledge base."""
        try:
            existing = self.collection.get(where={"user_id": user_id}, include=[])
            ids = existing.get("ids") or []
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0
