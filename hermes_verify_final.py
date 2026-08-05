import os
import tempfile
import shutil
import sys

# Add the project root to the Python path
sys.path.insert(0, r'C:\Users\Admin\Mayur-ab\RAG')

def test_service_uses_chroma_and_works():
    # Create a temporary directory for persistence
    tmpdir = tempfile.mkdtemp()
    print(f"Using temporary directory: {tmpdir}")
    
    try:
        # Set environment variable for vector DB path
        os.environ['VECTOR_DB_PATH'] = tmpdir
        # Reload the settings module to pick up the environment variable
        from importlib import reload
        import config.settings
        reload(config.settings)
        from config.settings import settings

        # Import and create the service
        from src.service import RAGPipelineService
        service = RAGPipelineService()

        # Check that the vector store is a ChromaVectorStore
        from src.vector_store.chroma_store import ChromaVectorStore
        assert isinstance(service.vector_store, ChromaVectorStore), \
            f"Expected ChromaVectorStore, got {type(service.vector_store)}"
        print("✓ Service is using ChromaVectorStore")

        # Create a simple test document
        from src.metadata.schema import Document, ChunkMetadata
        from src.embeddings.mock import MockEmbeddingProvider

        doc = Document(
            content="Test document for ChromaDB verification.",
            metadata=ChunkMetadata(
                document_id="doc1",
                chunk_id="chunk1",
                source="test.txt",
                title="Test",
                allowed_roles=["user"]
            )
        )

        # Get embedding from the service's embedding provider
        embedding = service.embedding_provider.embed_documents([doc.content])[0]

        # Add document to vector store via the service
        service.vector_store.add_documents([doc], [embedding])
        print("✓ Added document to vector store")

        # Verify the document was added by checking stats
        stats = service.vector_store.get_stats()
        assert stats['total_vectors'] == 1, f"Expected 1 vector, got {stats['total_vectors']}"
        print("✓ Vector store reports 1 vector")

        # Search for the document using the same embedding
        results = service.vector_store.search(embedding, top_k=1)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        found_doc, score = results[0]
        assert found_doc.id == "chunk1", f"Expected doc id 'chunk1', got '{found_doc.id}'"
        print("✓ Successfully retrieved document by vector similarity")

        # Clean up the document
        delete_count = service.vector_store.delete([doc.id])
        assert delete_count == 1, f"Expected to delete 1 document, got {delete_count}"
        stats_after = service.vector_store.get_stats()
        assert stats_after['total_vectors'] == 0, f"Expected 0 vectors after deletion, got {stats_after['total_vectors']}"
        print("✓ Successfully deleted document")

        print("\n🎉 All tests passed! ChromaDB integration is working correctly.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up the temporary directory with retries for Windows
        for attempt in range(3):
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
                print(f"Cleaned up temporary directory: {tmpdir}")
                break
            except PermissionError as e:
                if attempt == 2:  # Last attempt
                    print(f"Warning: Could not remove temporary directory {tmpdir} after 3 attempts: {e}")
                else:
                    import time
                    time.sleep(0.5)  # Wait before retry

if __name__ == '__main__':
    success = test_service_uses_chroma_and_works()
    sys.exit(0 if success else 1)