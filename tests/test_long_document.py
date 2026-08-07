import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context.document_request import (
    is_long_document_request,
    parse_requested_pages,
    plan_section_count,
    estimate_target_words,
)
from src.generation.long_document import parse_outline


def test_parse_requested_pages():
    assert parse_requested_pages("Generate a 20-page research report") == 20
    assert parse_requested_pages("Write 5 pages on RAG") == 5
    assert parse_requested_pages("What is RAG?") is None


def test_is_long_document_request():
    assert is_long_document_request("Generate a 20-page research report on RAG architectures")
    assert is_long_document_request("Write a 3000 word comprehensive report on chunking")
    assert is_long_document_request("Create a detailed research report on hybrid retrieval")
    assert not is_long_document_request("What is RAG?")
    assert not is_long_document_request("Summarize this in 2 sentences")


def test_plan_section_count():
    assert plan_section_count("Generate a 20-page report") >= 4
    assert plan_section_count("Write 5000 words on RAG") >= 4


def test_parse_outline():
    text = "1. Introduction\n2. RAG Basics\n3. Conclusion"
    assert parse_outline(text, 10) == ["Introduction", "RAG Basics", "Conclusion"]


def test_long_document_query_mode():
    from fastapi.testclient import TestClient
    from src.api.app import app

    client = TestClient(app)
    payload = {
        "query": "Generate a 10-page research report on RAG architecture and retrieval",
        "use_cache": False,
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "long_document"
    assert data.get("long_document", {}).get("generation_strategy") == "hierarchical"
    assert len(data["answer"]) > 100
    assert "##" in data["answer"] or "#" in data["answer"]
