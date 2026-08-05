import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metadata.schema import Document, ChunkMetadata
from src.embeddings.mock import MockEmbeddingProvider
from src.vector_store.chroma_store import ChromaVectorStore


def test_chroma_vector_store():
    # Create a temporary directory for persistence
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = ChromaVectorStore(persist_directory=tmpdir)
        emb_provider = MockEmbeddingProvider(dim=64)

        meta1 = ChunkMetadata(document_id="doc1", source="s1.txt", allowed_roles=["user"])
        meta2 = ChunkMetadata(document_id="doc2", source="s2.txt", allowed_roles=["admin"])

        doc1 = Document(id="chk1", content="Rishi Kanada discovered Parmanu atom concept.", metadata=meta1)
        doc2 = Document(id="chk2", content="Top secret nuclear physics equation.", metadata=meta2)

        vecs = emb_provider.embed_documents([doc1.content, doc2.content])
        store.add_documents([doc1, doc2], vecs)

        stats = store.get_stats()
        assert stats["total_vectors"] == 2

        # Query with user role -> doc2 should be filtered out by RBAC
        q_vec = emb_provider.embed_text("atom concept")
        res_user = store.search(q_vec, top_k=5, allowed_roles=["user"])
        assert len(res_user) == 1
        assert res_user[0][0].id == "chk1"

        # Query with admin role -> returns both
        res_admin = store.search(q_vec, top_k=5, allowed_roles=["admin"])
        assert len(res_admin) == 2

        # Deletion test
        deleted = store.delete(["doc1"])
        assert deleted == 1
        assert store.get_stats()["total_vectors"] == 1

        # Release Chroma file handles before tempdir cleanup (Windows)
        store.reset()
        del store.client
        del store


if __name__ == "__main__":
    test_chroma_vector_store()
    print("All tests passed!")