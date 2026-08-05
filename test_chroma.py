import os
import tempfile
from src.service import RAGPipelineService
from src.metadata.schema import Document, ChunkMetadata

def test_chroma():
    # Create a temporary directory for the vector store
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set the environment variable for the vector db path
        os.environ['VECTOR_DB_PATH'] = tmpdir
        # Reload the settings to pick up the environment variable
        from importlib import reload
        import config.settings
        reload(config.settings)
        from config.settings import settings
        print(f"Using vector db path: {settings.VECTOR_DB_PATH}")
        
        # Create the service
        service = RAGPipelineService()
        print(f"Vector store type: {type(service.vector_store)}")
        
        # Create a test document
        doc = Document(
            content="This is a test document about machine learning.",
            metadata=ChunkMetadata(
                document_id="test_doc_1",
                chunk_id="chunk_1",
                source="test.txt",
                title="Test Document",
                section="Introduction",
                author="Test Author",
                tags=["test", "machine learning"],
                allowed_roles=["user", "admin"]
            )
        )
        
        # Ingest the document
        print("Ingesting document...")
        service.vector_store.add_documents([doc], [[0.1, 0.2, 0.3, 0.4, 0.5]])
        print("Document ingested.")
        
        # Check stats
        stats = service.vector_store.get_stats()
        print(f"Vector store stats: {stats}")
        
        # Search for the document
        print("Searching for similar documents...")
        results = service.vector_store.search([0.1, 0.2, 0.3, 0.4, 0.5], top_k=1)
        print(f"Search results: {results}")
        
        if results:
            found_doc, score = results[0]
            print(f"Found document: {found_doc.content}")
            print(f"Score: {score}")
            print(f"Metadata: {found_doc.metadata}")
        else:
            print("No results found.")
            
        # Test filtering by metadata
        print("\nSearching with filter metadata {'author': 'Test Author'}...")
        filtered_results = service.vector_store.search(
            [0.1, 0.2, 0.3, 0.4, 0.5], 
            top_k=1, 
            filter_metadata={"author": "Test Author"}
        )
        print(f"Filtered results: {filtered_results}")
        
        # Test filtering by allowed_roles
        print("\nSearching with allowed_roles=['admin']...")
        role_results = service.vector_store.search(
            [0.1, 0.2, 0.3, 0.4, 0.5], 
            top_k=1, 
            allowed_roles=["admin"]
        )
        print(f"Role-based results: {role_results}")
        
        # Test deleting
        print("\nDeleting document...")
        delete_count = service.vector_store.delete([doc.id])
        print(f"Deleted {delete_count} document(s).")
        
        # Check stats after deletion
        stats_after = service.vector_store.get_stats()
        print(f"Vector store stats after deletion: {stats_after}")

if __name__ == "__main__":
    test_chroma()