import os
import json
from typing import List, Dict, Any, Optional, Tuple
from src.metadata.schema import Document

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class ChromaVectorStore:
    """ChromaDB Integration - requires chromadb to be installed and a persist directory."""

    def __init__(self, persist_directory: Optional[str] = None):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Please install chromadb to use ChromaVectorStore.")
        if persist_directory is None:
            raise ValueError("persist_directory must be provided for ChromaVectorStore")
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Use cosine similarity for consistency with memory store
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def _document_from_chroma(self, doc_id: str, doc_content: str, metadata: Dict[str, Any]) -> Document:
        from src.metadata.schema import ChunkMetadata

        converted_metadata: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                if value.startswith("[") and value.endswith("]"):
                    try:
                        converted_metadata[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                if value.startswith("{") and value.endswith("}"):
                    try:
                        converted_metadata[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                converted_metadata[key] = value
            else:
                converted_metadata[key] = value

        chunk_meta_obj = ChunkMetadata(**converted_metadata)
        return Document(content=doc_content, id=doc_id, metadata=chunk_meta_obj)

    def get_all_documents(self) -> List[Document]:
        """Load all indexed chunks from Chroma for BM25 rehydration on startup."""
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas", "documents"])
        documents: List[Document] = []
        for doc_id, doc_content, metadata in zip(
            results.get("ids", []),
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            documents.append(self._document_from_chroma(doc_id, doc_content, metadata))
        return documents

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        if not documents:
            raise ValueError("Cannot add an empty document list to the vector store.")
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Document/embedding count mismatch: {len(documents)} docs vs {len(embeddings)} embeddings."
            )
        if any(not emb for emb in embeddings):
            raise ValueError("All documents must have non-empty embedding vectors.")

        ids = [doc.id for doc in documents]  # Use Document's id (chunk_id)
        embeddings_list = [list(map(float, emb)) for emb in embeddings]  # ensure float
        documents_list = [doc.content for doc in documents]
        # Flatten metadata to ChromaDB-compatible types (str, int, float, bool)
        metadatas_list = []
        for doc in documents:
            metadata_dict = doc.metadata.model_dump()
            storage_metadata = {}
            for key, value in metadata_dict.items():
                if isinstance(value, (str, int, float, bool)):
                    storage_metadata[key] = value
                elif value is None:
                    storage_metadata[key] = ""  # ChromaDB doesn't support None, use empty string
                elif isinstance(value, (list, dict)):
                    storage_metadata[key] = json.dumps(value)
                else:
                    # For any other type (e.g., datetime, UUID), convert to string
                    storage_metadata[key] = str(value)
            metadatas_list.append(storage_metadata)
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents_list,
            metadatas=metadatas_list
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        allowed_roles: Optional[List[str]] = None
    ) -> List[Tuple[Document, float]]:
        # Convert filter_metadata to storage strings for the where clause
        where_clause = None
        if filter_metadata is not None:
            where_clause = {}
            for key, value in filter_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    where_clause[key] = value
                elif value is None:
                    where_clause[key] = ""
                elif isinstance(value, (list, dict)):
                    where_clause[key] = json.dumps(value)
                else:
                    where_clause[key] = str(value)
        # ChromaDB query
        results = self.collection.query(
            query_embeddings=[list(map(float, query_vector))],
            n_results=top_k,
            where=where_clause,
            include=["metadatas", "documents", "distances"]
        )
        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        documents = results.get('documents', [[]])[0]

        results_list: List[Tuple[Document, float]] = []
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i]
            doc_content = documents[i]
            distance = distances[i]
            similarity = 1.0 - distance
            doc = self._document_from_chroma(doc_id, doc_content, metadata)
            results_list.append((doc, similarity))

        # Filter by allowed_roles if provided (post-filtering)
        # This matches the logic in the original MemoryVectorStore:
        # If the user is NOT an admin (i.e., "admin" not in their allowed_roles),
        # then check if they have access to the document.
        # If they ARE an admin, they get access to everything.
        if allowed_roles is not None:
            filtered_results = []
            is_admin = "admin" in allowed_roles
            for doc, score in results_list:
                if is_admin:
                    # Admin users get access to all documents
                    filtered_results.append((doc, score))
                else:
                    # Non-admin users must have at least one role in common with the document
                    doc_roles = getattr(doc.metadata, "allowed_roles", ["user", "admin"])
                    if any(role in allowed_roles for role in doc_roles):
                        filtered_results.append((doc, score))
            results_list = filtered_results

        return results_list

    def delete(self, document_ids: List[str]) -> int:
        if not document_ids:
            return 0
        ids_to_delete = set()
        # First, try to get by metadata.document_id
        try:
            results = self.collection.get(where={"document_id": {"$in": document_ids}}, include=[])
            ids_to_delete.update(results.get('ids', []))
        except Exception:
            pass
        # Then, try to get by id (chunk id)
        try:
            results = self.collection.get(ids=document_ids, include=[])
            ids_to_delete.update(results.get('ids', []))
        except Exception:
            pass
        if ids_to_delete:
            self.collection.delete(ids=list(ids_to_delete))
            return len(ids_to_delete)
        return 0

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            "store_type": "ChromaVectorStore",
            "total_vectors": count,
            "dimension": 0  # ChromaDB doesn't expose dimension easily; we can set to 0 or unknown
        }

    def reset(self) -> None:
        """Delete and recreate the collection (required when switching embedding models)."""
        self.client.delete_collection("rag_collection")
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )