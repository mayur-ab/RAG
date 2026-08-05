import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metadata.schema import Document, ChunkMetadata
from src.embeddings.mock import MockEmbeddingProvider
from src.vector_store.memory_store import MemoryVectorStore
from src.retrieval.vector_search import VectorSearchEngine
from src.retrieval.keyword_search import BM25SearchEngine
from src.retrieval.hybrid import HybridSearchEngine


def test_hybrid_retrieval():
    v_store = MemoryVectorStore()
    emb_provider = MockEmbeddingProvider(dim=64)

    meta = ChunkMetadata(document_id="doc1", source="paper.txt")
    doc1 = Document(id="chk1", content="Parmanu represents the indivisible fundamental particle in Vedic logic.", metadata=meta)
    doc2 = Document(id="chk2", content="Acoustic Rishis studied Shabda Brahma and Chladni figure vibrations.", metadata=meta)

    vecs = emb_provider.embed_documents([doc1.content, doc2.content])
    v_store.add_documents([doc1, doc2], vecs)

    v_engine = VectorSearchEngine(v_store, emb_provider)
    b_engine = BM25SearchEngine()
    b_engine.index([doc1, doc2])

    hybrid = HybridSearchEngine(v_engine, b_engine)

    results = hybrid.search(query="Parmanu indivisible particle", top_k=2)
    assert len(results) >= 1
    assert results[0][0].id == "chk1"
