import re
from typing import Optional
from config.settings import settings

_BREVITY_RULE = (
    "1. Write a clear, concise answer in natural language (1-3 short paragraphs max)."
)
_EXPANDED_RULE = (
    "1. Write a thorough, well-structured answer grounded in the context. "
    "Use multiple paragraphs and cover all relevant details when the user asks for depth or a specific length."
)

_WORD_COUNT_RE = re.compile(
    r"(?:"
    r"(?:in|about|around|at least|up to|maximum|max|min|minimum|roughly|~)?\s*"
    r"(\d{1,5})\s*(?:word|words)\b"
    r"|"
    r"(\d{1,5})\s*(?:token|tokens)\b"
    r")",
    re.IGNORECASE,
)


def parse_requested_words(query: str) -> Optional[int]:
    """Extract the largest explicit word/token target from the user query."""
    if not query:
        return None

    target = None
    for match in _WORD_COUNT_RE.finditer(query):
        raw = next((g for g in match.groups() if g), None)
        if not raw:
            continue
        value = int(raw)
        if "token" in match.group(0).lower():
            value = int(value / 1.35)
        target = max(target or 0, value)
    return target


def resolve_max_output_tokens(query: str) -> int:
    """Map user length hints to an LLM output token budget."""
    words = parse_requested_words(query)
    if not words:
        return settings.DEFAULT_MAX_OUTPUT_TOKENS

    if words <= 20:
        return max(16, min(settings.DEFAULT_MAX_OUTPUT_TOKENS, words * 6))

    estimated_tokens = int(words * 1.35)
    return min(
        settings.MAX_OUTPUT_TOKENS_CAP,
        max(settings.DEFAULT_MAX_OUTPUT_TOKENS, estimated_tokens),
    )


def resolve_context_tokens(query: str) -> int:
    """Use a larger retrieval context budget when the user asks for long answers."""
    words = parse_requested_words(query)
    if words and words >= 2000:
        return min(settings.MAX_CONTEXT_TOKENS, 8000)
    if words and words >= 1000:
        return min(settings.MAX_CONTEXT_TOKENS, 6000)
    return settings.DEFAULT_CONTEXT_TOKENS


def augment_for_length(base_prompt: str, query: str) -> str:
    words = parse_requested_words(query)
    if not words:
        return base_prompt

    if words <= 20:
        prompt = base_prompt.replace(
            _BREVITY_RULE,
            "1. Write a very brief answer that strictly respects the user's length limit.",
        )
        return prompt.rstrip() + f"\n10. Reply in at most {words} words."

    prompt = base_prompt.replace(_BREVITY_RULE, _EXPANDED_RULE)
    prompt = prompt.rstrip() + (
        f"\n10. The user requested approximately {words} words. "
        "Provide as much relevant detail from the context as needed to approach that length."
    )
    return prompt


def augment_chat_system_prompt(base_prompt: str, query: str) -> str:
    words = parse_requested_words(query)
    if not words:
        return base_prompt

    prompt = base_prompt.replace(
        "Answer clearly and concisely.",
        "Answer thoroughly and completely.",
    )
    return prompt.rstrip() + (
        f"\nWhen appropriate, aim for approximately {words} words in your response."
    )
