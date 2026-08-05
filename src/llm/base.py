from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

SYSTEM_PROMPT = """You answer questions using only the provided context passages.

Rules:
1. Write a clear, concise answer in natural language (1-3 short paragraphs max).
2. Reference sources inline using citation numbers like [1] or [2] that match the [N] labels in the context.
3. Do NOT include file paths, document titles, or chunk IDs in your answer.
4. Do NOT add a "Sources", "References", or "Citations" section — not even a list like [1] Section name. Sources are shown separately in the UI.
5. End your response after the answer text. Nothing after the final sentence.
6. If the answer cannot be found in the context, say exactly:
   "I could not find that information in the provided knowledge base."
7. Do not fabricate information."""


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Generate grounded response given user prompt and context block.
        Returns dict containing 'answer', 'model', 'prompt_tokens', 'completion_tokens'.
        """
        pass
