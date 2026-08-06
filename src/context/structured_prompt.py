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


def augment_system_prompt(base_prompt: str, query: str) -> str:
    prompt = augment_for_length(base_prompt, query)
    if is_structured_query(query):
        return prompt.rstrip() + "\n" + STRUCTURED_TABLE_HINT
    return prompt
