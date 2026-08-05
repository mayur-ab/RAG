import os
import tempfile
import shutil
import sys

# Add the project root to the Python path
sys.path.insert(0, r'C:\Users\Admin\Mayur-ab\RAG')

def test_service_uses_chroma():
    # Create a temporary directory for persistence
    tmpdir = tempfile.mkdtemp()
    try:
        print(f"Using temporary directory: {tmpdir}")
        # Set environment variables to use mock providers and chroma
        os.environ['EMBEDDING_PROVIDER'] = 'mock'
        os.environ['LLM_PROVIDER'] = 'mock'
        os.environ['VECTOR_STORE_PROVIDER'] = 'chroma'
        os.environ['VECTOR_DB_PATH'] = tmpdir
        os.environ['RERANKER_PROVIDER'] = 'identity'

        # Reload the settings module to pick up the environment variables
        import importlib
        import config.settings
        importlib.reload(config.settings)
        from config.settings import settings

        # Initialize the service
        from src.service import RAGPipelineService
        service = RAGPipelineService()

        # Check that the vector store is indeed a ChromaVectorStore
        from src.vector_store.chroma_store import ChromaVectorStore
        assert isinstance(service.vector_store, ChromaVectorStore), \
            f"Expected ChromaVectorStore, got {type(service.vector_store)}"
        print("SUCCESS: Service is using ChromaVectorStore")

        # Additionally, we can do a quick operation to ensure it works
        from src.metadata.schema import Document, ChunkMetadata
        from src.embeddings.mock import MockEmbeddingProvider

        # Create a test document
        doc = Document(
            content="Test document for verification.",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_id="chunk1",
                source="test.txt",
                title="Test",
                section="Test",
                author="Test",
                created_at="2024-01-01T00:00:00",
                updated_at="2024-01-01T00:00:00",
                tags=["test"],
                version="1.0",
                allowed_roles=["user"],
                extra={}
            )
        )
        # Add document
        service.vector_store.add_documents([doc], [[0.1, 0.2, 0.3, 0.4, 0.5]])
        # Check stats
        stats = service.vector_store.get_stats()
        assert stats['total_vectors'] == 1, f"Expected 1 vector, got {stats['total_vectors']}"
        print("SUCCESS: Added document and verified vector count")

        # Clean up the document
        service.vector_store.delete([doc.id])
        stats_after = service.vector_store.get_stats()
        assert stats_after['total_vectors'] == 0, f"Expected 0 vectors after delete, got {stats_after['total_vectors']}"
        print("SUCCESS: Deleted document and verified vector count is zero")

    except Exception as e:
        print(f"ERROR: {e}")
        raise
    finally:
        # Clean up the temporary directory
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {tmpdir}")
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {tmpdir}: {e}")
        # Clean up environment variables
        for var in ['EMBEDDING_PROVIDER', 'LLM_PROVIDER', 'VECTOR_STORE_PROVIDER', 'VECTOR_DB_PATH', 'RERANKER_PROVIDER']:
            if var in os.environ:
                del os.environ[var]
        # Reload the settings module to reset to defaults
        import importlib
        import config.settings
        importlib.reload(config.settings)

if __name__ == '__main__':
    test_service_uses_chroma()
    print("All tests passed!")