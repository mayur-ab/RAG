import os
import sys
import logging

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.context_budget import build_token_usage, log_context_usage, get_context_window
from src.memory.models import UserMemoryContext, UserProfile, UserPreferences, UserStyle
from src.memory.prompt import format_memory_context


def test_build_token_usage_includes_context_window():
    usage = build_token_usage(4000, 512, model="test", log=False)
    assert usage["prompt_tokens"] == 4000
    assert usage["completion_tokens"] == 512
    assert usage["context_window"] == get_context_window()
    assert usage["context_used_percent"] is not None


def test_build_token_usage_zero_prompt():
    usage = build_token_usage(0, 0, log=False)
    assert usage["context_used_percent"] is None


def test_log_context_usage_warns_near_limit(caplog):
    with caplog.at_level(logging.WARNING):
        log_context_usage(28000, 1000, model="test", context_window=32768)
    assert any("Context window pressure" in r.message for r in caplog.records)


def test_format_memory_context_truncates_long_block():
    profile = UserProfile(
        user_id="test-user",
        display_name="Test User",
        preferences=UserPreferences(response_style="detailed"),
        style=UserStyle(preferred_answer_style="detailed", likes_examples=True),
        interests=["topic"] * 50,
        projects=["project"] * 50,
        facts=["fact " + "x" * 200 for _ in range(20)],
    )
    ctx = UserMemoryContext(
        profile=profile,
        frequent_topics=["RAG", "AI", "memory", "testing"],
        episodic_memories=[],
    )
    block = format_memory_context(ctx)
    assert len(block) <= 4000
