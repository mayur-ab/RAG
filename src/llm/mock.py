import re
from typing import Dict, Any
from src.llm.base import BaseLLMProvider, SYSTEM_PROMPT


class MockLLMProvider(BaseLLMProvider):
    """Deterministic grounded mock LLM provider for zero-dependency offline operation."""

    def generate(
        self,
        prompt: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        if not context or not context.strip():
            return {
                "answer": "I could not find that information in the provided knowledge base.",
                "model": "mock-grounded-llm",
                "prompt_tokens": len(prompt.split()) + len(context.split()),
                "completion_tokens": 12
            }

        # Extract sentences from context that match prompt keywords
        query_words = set(re.findall(r"\w+", prompt.lower())) - {"what", "is", "the", "a", "an", "and", "of", "in", "to", "for", "how"}
        lines = [line.strip() for line in context.split("\n") if line.strip() and not line.startswith("---")]

        matching_lines = []
        for line in lines:
            line_words = set(re.findall(r"\w+", line.lower()))
            if query_words & line_words:
                matching_lines.append(line)

        if matching_lines:
            answer = "Based on the knowledge base:\n\n" + "\n".join(matching_lines[:5])
        else:
            # Return first snippet if query words are generic
            snippet = lines[0] if lines else ""
            answer = f"Based on the provided context: {snippet}" if snippet else "I could not find that information in the provided knowledge base."

        return {
            "answer": answer,
            "model": "mock-grounded-llm",
            "prompt_tokens": len(prompt.split()) + len(context.split()),
            "completion_tokens": len(answer.split())
        }
