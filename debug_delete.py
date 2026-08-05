import os
import tempfile
from src.metadata.schema import Document, ChunkMetadata
from src.embeddings.mock import MockEmbeddingProvider
from src.vector_store.chroma_store import ChromaVectorStore

# Create a temporary directory for persistence
tmpdir = tempfile.mkdtemp()
print(f"Using temporary directory: {tmpdir}")
store = ChromaVectorStore(persist_directory=tmpdir)
emb_provider = MockEmbeddingProvider(dim=64)

meta1 = ChunkMetadata(document_id="doc1", source="s1.txt", allowed_roles=["user"])
meta2 = ChunkMetadata(document_id="doc2", source="s2.txt", allowed_roles=["admin"])

doc1 = Document(id="chk1", content="Rishi Kanada discovered Parmanu atom concept.", metadata=meta1)
doc2 = Document(id="chk2", content="Top secret nuclear physics equation.", metadata=meta2)

vecs = emb_provider.embed_documents([doc1.content, doc2.content])
print(f"Embeddings: {vecs}")
store.add_documents([doc1, doc2], vecs)

stats = store.get_stats()
print(f"Store stats after adding: {stats}")
assert stats["total_vectors"] == 2

# Let's see what's in the collection
results = store.collection.get(include=["metadatas"])
print(f"All ids: {results.get('ids')}")
print(f"All metadatas: {results.get('metadatas')}")

# Now try to delete by document_id
print("\nTrying to delete by document_id=['doc1']")
deleted = store.delete(["doc1"])
print(f"Delete returned: {deleted}")

stats_after = store.get_stats()
print(f"Store stats after delete: {stats_after}")

# Let's see what's left
results_after = store.collection.get(include=["metadatas"])
print(f"All ids after delete: {results_after.get('ids')}")
print(f"All metadatas after delete: {results_after.get('metadatas')}")

# Clean up
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)