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
    assert data["context"]["default_context_tokens"] == 4000
    assert data["context"]["ollama_num_ctx"] == 32768
    assert data["context"]["chunk_size_chars"] == 800
    assert data["retrieval"]["top_k_rerank"] == 5


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
    assert "chat_compact" in data


def test_session_hierarchy_endpoints():
    user_id = "test-session-user-001"
    start = client.post(f"/memory/session/start?user_id={user_id}")
    assert start.status_code == 200
    session_id = start.json()["session_id"]

    chat_end = client.post(
        "/memory/session/chat-end",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "chat_id": "chat-test-001",
            "chat_compact": "User asked about RAG; assistant explained retrieval.",
        },
    )
    assert chat_end.status_code == 200
    assert chat_end.json()["merged"] is True

    query = client.post(
        "/query",
        json={
            "query": "What is RAG?",
            "user_id": user_id,
            "session_id": session_id,
            "chat_id": "chat-test-002",
            "chat_compact": "Prior chat covered RAG basics.",
            "use_cache": False,
        },
    )
    assert query.status_code == 200
    assert query.json().get("chat_compact")

    end = client.post(
        "/memory/session/end",
        json={"user_id": user_id, "session_id": session_id},
    )
    assert end.status_code == 200
    assert end.json()["archived"] is True


def test_duolingo_followup_topic_anchoring():
    follow_up = client.post(
        "/query",
        json={
            "query": "how do i start the same business from scratch",
            "chat_compact": "User asked about Duolingo Strategy, gamification, and adaptive learning.",
            "routing_turns": [
                {"role": "user", "content": "what should i learn from this"},
                {"role": "assistant", "content": "Apply gamification and measurable goals."},
            ],
            "pinned_sources": ["C:/Docs/Duo Lingo Strategy.doc"],
            "use_cache": False,
        },
    )
    assert follow_up.status_code == 200
    data = follow_up.json()
    ctx = data.get("retrieved_context", "").lower()
    answer = data.get("answer", "").lower()
    rewritten = (data.get("rewritten_query") or "").lower()
    assert "duolingo" in ctx or "duolingo" in rewritten
    assert "kapha" not in answer
    assert "pitta" not in answer
    assert "vata" not in answer
    assert data.get("pinned_sources")
