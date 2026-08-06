import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context.conversation import (
    is_refinement_follow_up,
    build_rag_user_message,
    resolve_retrieval_query,
)
from src.retrieval.query_rewriter import QueryRewriter
from src.llm.mock import MockLLMProvider
from src.context.response_length import resolve_max_output_tokens


def test_refinement_follow_up_detection():
    assert is_refinement_follow_up("explain in 2 words only")
    assert is_refinement_follow_up("okay - explain in very funny way in short")
    assert not is_refinement_follow_up("What is AI in Learning Space?")


def test_build_rag_user_message_includes_history():
    history = [
        {"role": "user", "content": "Tell me about AI in Learning Space"},
        {"role": "assistant", "content": "AI helps personalize learning."},
    ]
    message = build_rag_user_message("context block", "explain in 2 words only", history)
    assert "Conversation so far" in message
    assert "AI in Learning Space" in message
    assert "explain in 2 words only" in message


def test_resolve_retrieval_query_skips_rewrite_for_refinement():
    rewriter = QueryRewriter(MockLLMProvider())
    history = [{"role": "user", "content": "Tell me about AI in Learning Space"}]
    query = resolve_retrieval_query("explain in 2 words only", history, rewriter, True)
    assert "AI in Learning Space" in query
    assert "2 words" in query


def test_short_word_limit_tokens():
    assert resolve_max_output_tokens("explain in 2 words only") <= 20
