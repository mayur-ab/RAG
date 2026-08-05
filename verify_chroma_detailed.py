import os
import tempfile
import shutil
import sys

# Add the project root to the Python path
sys.path.insert(0, r'C:\Users\Admin\Mayur-ab\RAG')

def test_chroma_basic_functionality():
    """Test that ChromaVectorStore works correctly."""
    # Create a temporary directory for persistence
    tmpdir = tempfile.mkdtemp()
    print(f"Using temporary directory: {tmpdir}")
    
    try:
        from src.metadata.schema import Document, ChunkMetadata
        from src.embeddings.mock import MockEmbeddingProvider
        from src.vector_store.chroma_store import ChromaVectorStore
        
        store = ChromaVectorStore(persist_directory=tmpdir)
        emb_provider = MockEmbeddingProvider(dim=64)

        # Test data
        meta1 = ChunkMetadata(document_id="doc1", source="s1.txt", allowed_roles=["user"])
        meta2 = ChunkMetadata(document_id="doc2", source="s2.txt", allowed_roles=["admin"])

        doc1 = Document(id="chk1", content="Rishi Kanada discovered Parmanu atom concept.", metadata=meta1)
        doc2 = Document(id="chk2", content="Top secret nuclear physics equation.", metadata=meta2)

        vecs = emb_provider.embed_documents([doc1.content, doc2.content])
        print(f"Document 1 embedding (first 5 vals): {vecs[0][:5]}")
        print(f"Document 2 embedding (first 5 vals): {vecs[1][:5]}")
        
        store.add_documents([doc1, doc2], vecs)

        stats = store.get_stats()
        print(f"Store stats: {stats}")
        assert stats["total_vectors"] == 2, f"Expected 2 vectors, got {stats['total_vectors']}"
        print("✓ Added 2 documents successfully")

        # Test search with query similar to doc1
        query_text = "atom concept"
        q_vec = emb_provider.embed_text(query_text)
        print(f"Query '{query_text}' vector (first 5 vals): {q_vec[:5]}")
        
        results = store.search(q_vec, top_k=5)
        print(f"Search results count: {len(results)}")
        for i, (doc, score) in enumerate(results):
            print(f"  Result {i+1}: id='{doc.id}', score={score:.6f}, content='{doc.content[:50]}...'")
        
        # Check that we got results and the first one is the expected document
        assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
        # The first result should be doc1 since it's more relevant to "atom concept"
        assert results[0][0].id == "chk1", f"Expected doc id 'chk1' as most relevant, got '{results[0][0].id}'"
        print("✓ Search by vector similarity works and returns most relevant result first")

        # Test metadata filtering
        filtered = store.search(q_vec, top_k=5, filter_metadata={"source": "s1.txt"})
        print(f"Filtered by source='s1.txt': {len(results)} results")
        assert len(filtered) == 1, f"Expected 1 result with source filter, got {len(filtered)}"
        assert filtered[0][0].id == "chk1", f"Expected doc id 'chk1' with source filter, got '{filtered[0][0].id}'"
        print("✓ Metadata filtering works")

        # Test role-based filtering (non-admin should get only user doc)
        user_results = store.search(q_vec, top_k=5, allowed_roles=["user"])
        print(f"User role results: {len(user_results)}")
        assert len(user_results) == 1, f"Expected 1 result for user role, got {len(user_results)}"
        assert user_results[0][0].id == "chk1", f"Expected doc id 'chk1' for user, got '{user_results[0][0].id}'"
        
        # Test role-based filtering (admin should get both)
        admin_results = store.search(q_vec, top_k=5, allowed_roles=["admin"])
        print(f"Admin role results: {len(admin_results)}")
        assert len(admin_results) == 2, f"Expected 2 results for admin role, got {len(admin_results)}"
        print("✓ Role-based filtering works")

        # Test deletion by document_id
        deleted_count = store.delete(["doc1"])
        print(f"Deleted {deleted_count} documents by document_id")
        assert deleted_count == 1, f"Expected to delete 1 document, got {deleted_count}"
        stats_after = store.get_stats()
        print(f"Store stats after deletion: {stats_after}")
        assert stats_after["total_vectors"] == 1, f"Expected 1 vector after deletion, got {stats_after['total_vectors']}"
        print("✓ Deletion by document_id works")

        # Verify the right document remains
        remaining_results = store.search(q_vec, top_k=5)
        print(f"After deletion, search results: {len(remaining_results)}")
        assert len(remaining_results) == 1, f"Expected 1 result after deletion, got {len(remaining_results)}"
        assert remaining_results[0][0].id == "chk2", f"Expected doc id 'chk2' to remain, got '{remaining_results[0][0].id}'"
        print("✓ Correct document remains after deletion")

        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up with retries to handle Windows file locking
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

if __name__ == "__main__":
    success = test_chroma_basic_functionality()
    sys.exit(0 if success else 1)