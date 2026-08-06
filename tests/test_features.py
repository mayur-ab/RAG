import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.app import app
from src.cache.query_cache import QueryCache
from src.context.structured_prompt import is_structured_query

client = TestClient(app)


def test_query_cache_roundtrip():
    cache = QueryCache()
    payload = {"answer": "cached answer", "mode": "rag"}
    cache.set("hello", True, "mock", None, payload)
    hit = cache.get("hello", True, "mock", None)
    assert hit is not None
    assert hit["answer"] == "cached answer"
    assert hit["cached"] is True


def test_structured_query_detection():
    assert is_structured_query("List manufacturers in Mumbai with email and phone")
    assert not is_structured_query("What is Shabda Brahma?")


def test_query_endpoint_direct_mode():
    payload = {"query": "What is 2 plus 2?", "use_rag": False}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("mode") == "direct"
    assert data.get("not_in_documents") is True


def test_query_stream_endpoint():
    payload = {"query": "What is Rishi Kanada theory?", "use_rag": True}
    response = client.post("/query/stream", json=payload)
    assert response.status_code == 200
    body = response.text
    assert "data:" in body
    assert "done" in body or "token" in body


def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "current_model" in data
    assert "models" in data


def test_indexed_documents_endpoint():
    response = client.get("/documents/indexed")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "document_count" in data
