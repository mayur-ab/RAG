import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.mock import MockLLMProvider


def test_mock_llm_grounding():
    provider = MockLLMProvider()
    context = "Micro-Macro Mirroring is the ability to perceive small objects reflecting the universe."
    
    # Grounded query
    res = provider.generate(prompt="What is Micro-Macro Mirroring?", context=context)
    assert "Micro-Macro Mirroring" in res["answer"]
    assert res["model"] == "mock-grounded-llm"

    # Empty context query -> Strict Grounding Fallback
    res_empty = provider.generate(prompt="Who won the world cup in 2022?", context="")
    assert "I could not find that information in the provided knowledge base." in res_empty["answer"]
