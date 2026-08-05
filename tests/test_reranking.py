import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metadata.schema import Document, ChunkMetadata
from src.reranking.identity import IdentityReranker


def test_identity_reranker():
    meta = ChunkMetadata(document_id="d1", source="s1")
    d1 = Document(id="1", content="Text one", metadata=meta)
    d2 = Document(id="2", content="Text two", metadata=meta)

    reranker = IdentityReranker()
    res = reranker.rerank(query="query", documents=[(d1, 0.5), (d2, 0.9)], top_n=1)
    assert len(res) == 1
    assert res[0][0].id == "2"
