import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data


def test_query_endpoint_empty_knowledge_base():
    payload = {"query": "What is Rishi Kanada theory?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "match_percent" in data
    assert "retrieved_context" in data
    assert isinstance(data["match_percent"], (int, float))
    assert data.get("mode") == "rag"


def test_evaluate_endpoint():
    payload = {
        "query": "What is Rishi Kanada theory?",
        "use_llm_judge": True,
        "use_cache": False,
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "heuristic" in data
    assert "faithfulness" in data["heuristic"]
    assert data.get("llm_judge") is not None


def test_query_endpoint_direct_mode():
    payload = {"query": "What is 2 plus 2?", "use_rag": False}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("mode") == "direct"
    assert data["retrieved_chunks_count"] == 0
    assert data["reranked_chunks_count"] == 0
    assert "answer" in data
