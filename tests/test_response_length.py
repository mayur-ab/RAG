import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context.response_length import (
    parse_requested_words,
    resolve_max_output_tokens,
    resolve_context_tokens,
    augment_for_length,
)
from src.llm.base import SYSTEM_PROMPT


def test_parse_requested_words():
    assert parse_requested_words("tell me about AI in 5000 words") == 5000
    assert parse_requested_words("give a short summary") is None
    assert parse_requested_words("write at least 1200 words on safety") == 1200


def test_resolve_max_output_tokens_scales_with_words():
    assert resolve_max_output_tokens("hello") == 1024
    assert resolve_max_output_tokens("explain in 5000 words") >= 6000
    assert resolve_max_output_tokens("explain in 2 words only") <= 20


def test_resolve_context_tokens_scales_with_words():
    assert resolve_context_tokens("hello") == 4000
    assert resolve_context_tokens("explain in 5000 words") == 8000


def test_augment_for_length_replaces_brevity_rule():
    prompt = augment_for_length(SYSTEM_PROMPT, "explain in 2000 words")
    assert "1-3 short paragraphs max" not in prompt
    assert "2000 words" in prompt
