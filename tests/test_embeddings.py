import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.mock import MockEmbeddingProvider


def test_mock_embeddings():
    provider = MockEmbeddingProvider(dim=128)
    vec1 = provider.embed_text("Quantum Computing")
    vec2 = provider.embed_text("Quantum Computing")
    vec3 = provider.embed_text("Vedic Philosophy")

    assert len(vec1) == 128
    assert vec1 == vec2  # Deterministic hash check
    assert vec1 != vec3
