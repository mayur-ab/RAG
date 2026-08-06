import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.query_rewriter import QueryRewriter
from src.llm.mock import MockLLMProvider


class RewriteMockLLM(MockLLMProvider):
    def generate(self, prompt, context, system_prompt=None, temperature=0.0, max_tokens=1000):
        if "Follow-up question:" in prompt and "hindustan pencils" in context.lower():
            return {
                "answer": "What is the email id of Hindustan Pencils?",
                "model": "mock",
                "prompt_tokens": 10,
                "completion_tokens": 8,
            }
        return {"answer": prompt.replace("Follow-up question: ", ""), "model": "mock", "prompt_tokens": 5, "completion_tokens": 5}


def test_rewrite_follow_up_question():
    rewriter = QueryRewriter(RewriteMockLLM())
    routing = [
        {"role": "user", "content": "give me address of hindustan pencils"},
        {"role": "assistant", "content": "The address is Voltas House, Mumbai."},
    ]
    rewritten = rewriter.rewrite(
        "their email id?",
        chat_compact="User asked for Hindustan Pencils address; assistant gave Mumbai address.",
        routing_turns=routing,
    )
    assert "Hindustan Pencils" in rewritten
    assert "email" in rewritten.lower()


def test_rewrite_skips_without_history():
    rewriter = QueryRewriter(RewriteMockLLM())
    assert rewriter.rewrite("their email id?", None) == "their email id?"
