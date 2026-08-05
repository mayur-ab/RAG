import os
import tempfile
import shutil
import sys
import uuid

# Add the project root to the Python path
sys.path.insert(0, r'C:\Users\Admin\Mayur-ab\RAG')

def test_chroma_integration():
    """Test that the service uses ChromaDB and it works correctly."""
    # Create a temporary directory for persistence
    tmpdir = tempfile.mkdtemp()
    print(f"Using temporary directory: {tmpdir}")
    
    try:
        from src.metadata.schema import Document, ChunkMetadata
        from src.embeddings.mock import MockEmbeddingProvider
        from src.vector_store.chroma_store import ChromaVectorStore
        
        # Use a predictable document ID for testing
        test_doc_id = "test-doc-id-123"
        test_chunk_id = "test-chunk-id-456"
        
        store = ChromaVectorStore(persist_directory=tmpdir)
        emb_provider = MockEmbeddingProvider(dim=64)

        # Create test document with explicit ID
        doc = Document(
            id=test_doc_id,  # Explicitly set the ID
            content="Test document for ChromaDB integration.",
            metadata=ChunkMetadata(
                document_id="doc123",
                chunk_id=test_chunk_id,
                source="test.txt",
                title="Test Document",
                section="Test Section",
                author="Test Author",
                created_at="2024-01-01T00:00:00",
                updated_at="2024-01-01T00:00:00",
                tags=["test", "chromadb"],
                version="1.0",
                allowed_roles=["user", "admin"],
                extra={"test_key": "test_value"}
            )
        )
        
        print(f"Created document with ID: {doc.id}")
        print(f"Document metadata chunk_id: {doc.metadata.chunk_id}")

        # Get embedding
        embedding = emb_provider.embed_documents([doc.content])[0]
        print(f"Generated embedding (first 5 values): {embedding[:5]}")

        # Add document to vector store
        store.add_documents([doc], [embedding])
        print("✓ Added document to vector store")

        # Verify the document was added by checking stats
        stats = store.get_stats()
        assert stats['total_vectors'] == 1, f"Expected 1 vector, got {stats['total_vectors']}"
        print("✓ Vector store reports 1 vector")

        # Search for the document using the same embedding
        results = store.search(embedding, top_k=1)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        found_doc, score = results[0]
        
        # The returned document should have the same ID we set
        assert found_doc.id == test_doc_id, f"Expected doc id '{test_doc_id}', got '{found_doc.id}'"
        print(f"✓ Successfully retrieved document by vector similarity (score: {score:.6f})")

        # Verify metadata was preserved correctly
        assert found_doc.metadata.document_id == "doc123"
        assert found_doc.metadata.chunk_id == test_chunk_id
        assert found_doc.metadata.source == "test.txt"
        assert set(found_doc.metadata.tags) == {"test", "chromadb"}
        assert found_doc.metadata.extra["test_key"] == "test_value"
        print("✓ Metadata correctly preserved and retrieved")

        # Test deletion by document ID
        delete_count = store.delete([test_doc_id])
        assert delete_count == 1, f"Expected to delete 1 document, got {delete_count}"
        stats_after = store.get_stats()
        assert stats_after['total_vectors'] == 0, f"Expected 0 vectors after deletion, got {stats_after['total_vectors']}"
        print("✓ Successfully deleted document by ID")

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
    success = test_chroma_integration()
    sys.exit(0 if success else 1)