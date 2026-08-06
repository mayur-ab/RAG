import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context.conversation import (
    is_refinement_follow_up,
    build_rag_user_message,
    resolve_retrieval_query,
    requires_conversation_only,
    is_conversational_only,
)
from src.context.user_memory_messages import (
    extract_user_name,
    is_user_memory_message,
    should_skip_query_rewrite,
)
from src.retrieval.query_rewriter import QueryRewriter
from src.llm.mock import MockLLMProvider
from src.context.response_length import resolve_max_output_tokens


def test_refinement_follow_up_detection():
    assert is_refinement_follow_up("explain in 2 words only")
    assert is_refinement_follow_up("okay - explain in very funny way in short")
    assert not is_refinement_follow_up("What is AI in Learning Space?")


def test_build_rag_user_message_uses_compact_not_history():
    message = build_rag_user_message(
        "context block",
        "explain in 2 words only",
        chat_compact="User asked about AI in Learning Space; assistant explained personalization.",
    )
    assert "Conversation summary so far" in message
    assert "AI in Learning Space" in message
    assert "explain in 2 words only" in message
    assert "Conversation so far" not in message


def test_resolve_retrieval_query_skips_rewrite_for_refinement():
    rewriter = QueryRewriter(MockLLMProvider())
    routing = [{"role": "user", "content": "Tell me about AI in Learning Space"}]
    query = resolve_retrieval_query(
        "explain in 2 words only",
        "User asked about AI in Learning Space.",
        routing,
        rewriter,
        True,
    )
    assert "AI in Learning Space" in query
    assert "2 words" in query


def test_short_word_limit_tokens():
    assert resolve_max_output_tokens("explain in 2 words only") <= 20


def test_requires_conversation_only_for_translation():
    routing = [
        {"role": "user", "content": "create hindi text"},
        {"role": "assistant", "content": "देवी का अर्थ..."},
    ]
    assert requires_conversation_only("give same in japanese language", routing, "User asked for Hindi text.")


def test_requires_conversation_only_for_greeting_and_confusion():
    routing = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hi"}]
    assert requires_conversation_only("hey", routing)
    assert requires_conversation_only("??", routing)
    assert is_conversational_only("??")


def test_user_memory_message_detection():
    assert is_user_memory_message("hey my name is mayur")
    assert is_user_memory_message("remember my name please")
    assert should_skip_query_rewrite("remember my name please")
    assert not is_user_memory_message("explain about deviglow kit")


def test_requires_conversation_only_for_name_intro_without_history():
    assert requires_conversation_only("hey my name is mayur", None)
    assert requires_conversation_only(
        "remember my name please",
        [{"role": "user", "content": "hey my name is mayur"}],
    )


def test_extract_user_name_from_intro_and_history():
    assert extract_user_name("hey my name is mayur", None) == "Mayur"
    routing = [{"role": "user", "content": "hey my name is mayur"}]
    assert extract_user_name("remember my name please", routing) == "Mayur"


def test_resolve_retrieval_query_skips_rewrite_for_memory_messages():
    rewriter = QueryRewriter(MockLLMProvider())
    routing = [{"role": "user", "content": "hey my name is mayur"}]
    query = resolve_retrieval_query(
        "remember my name please",
        "User introduced themselves as Mayur.",
        routing,
        rewriter,
        True,
    )
    assert query == "remember my name please"
