import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification.llm_classifier import LLMDocumentClassifier, _extract_json
from src.classification.metadata_store import DocumentMetadataStore
from src.classification.models import DocumentClassification
from src.classification.clustering import cluster_documents, kmeans
import numpy as np


@pytest.fixture
def meta_store(tmp_path):
    return DocumentMetadataStore(str(tmp_path / "doc_meta.db"))


def test_extract_json():
    raw = '```json\n{"category":"Technical","tags":["rag"]}\n```'
    data = _extract_json(raw)
    assert data["category"] == "Technical"


def test_metadata_store_upsert(meta_store):
    classification = DocumentClassification(
        category="Technical",
        subcategory="RAG",
        tags=["rag", "chromadb"],
        summary="RAG architecture guide.",
    )
    record = meta_store.upsert_classification(
        document_id="doc-1",
        source="Docs/rag.pdf",
        title="RAG Architecture",
        classification=classification,
        doc_embedding=[0.1, 0.2, 0.3],
    )
    assert record.category == "Technical"
    tree = meta_store.get_category_tree()
    assert tree["total_classified"] == 1


def test_kmeans_basic():
    vectors = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=float)
    labels, _ = kmeans(vectors, k=2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]


def test_cluster_documents():
    items = [
        {"document_id": "a", "source": "a.pdf", "title": "A", "embedding": [1, 0]},
        {"document_id": "b", "source": "b.pdf", "title": "B", "embedding": [0.9, 0.1]},
        {"document_id": "c", "source": "c.pdf", "title": "C", "embedding": [0, 1]},
        {"document_id": "d", "source": "d.pdf", "title": "D", "embedding": [0.1, 0.9]},
    ]
    groups = cluster_documents(items, k=2)
    assert len(groups) == 2


def test_mock_classifier():
    from src.llm.mock import MockLLMProvider

    clf = LLMDocumentClassifier(MockLLMProvider())
    result = clf.classify("RAG Guide", "Content about chromadb and retrieval.")
    assert result.category == "Technical"
    assert "rag" in result.tags


def test_documents_categories_api():
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)
    res = client.get("/documents/categories")
    assert res.status_code == 200
    assert "categories" in res.json()


def test_classify_endpoint():
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)
    res = client.post("/documents/classify?only_missing=true")
    assert res.status_code == 200
    data = res.json()
    assert "classified" in data
