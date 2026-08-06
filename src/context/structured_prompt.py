import re

from src.context.response_length import augment_for_length

STRUCTURED_TABLE_HINT = """
8. When the question asks for lists of companies, manufacturers, contacts, or records with fields like name, address, phone, or email, format the answer as a Markdown table with clear column headers (e.g. Company | Address | Phone | Email).
9. Use one row per record. Do not use bullet lists when a table is more appropriate.
"""

_STRUCTURED_PATTERNS = re.compile(
    r"\b("
    r"list|table|compare|companies|manufacturers|suppliers|contacts|"
    r"email|phone|address|directory|all\s+\w+\s+in"
    r")\b",
    re.IGNORECASE,
)


def is_structured_query(query: str) -> bool:
    return bool(_STRUCTURED_PATTERNS.search(query or ""))


_CREATIVE_PATTERNS = re.compile(
    r"\b(story|stories|poem|script|narrative|bedtime tale|children|kids|creative)\b",
    re.IGNORECASE,
)

_CREATIVE_HINT = """
10. The user asked for a creative response (e.g. a story for children). Use facts from the context as the foundation, and you may synthesize a creative narrative grounded in those facts. Do not invent facts that contradict the context.
"""


def is_creative_query(query: str) -> bool:
    return bool(_CREATIVE_PATTERNS.search(query or ""))


_CONVERSATION_GROUNDING_HINT = """
11. Base your answer only on the provided context passages and the "Conversation so far" section (if present).
12. Do not reference "previous conversations", "earlier chats", or "as we discussed before" unless that exact topic appears in the conversation section.
"""


def augment_system_prompt(base_prompt: str, query: str) -> str:
    prompt = augment_for_length(base_prompt, query)
    if is_structured_query(query):
        prompt = prompt.rstrip() + "\n" + STRUCTURED_TABLE_HINT
    if is_creative_query(query):
        prompt = prompt.rstrip() + "\n" + _CREATIVE_HINT
    prompt = prompt.rstrip() + "\n" + _CONVERSATION_GROUNDING_HINT
    return prompt
