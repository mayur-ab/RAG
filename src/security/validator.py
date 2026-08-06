import re

from config.settings import settings


class InputValidator:
    """Input sanitization and prompt injection protection."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"system\s+prompt\s+override", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+DAN", re.IGNORECASE),
        re.compile(r"reveal\s+all\s+passwords", re.IGNORECASE),
    ]

    @classmethod
    def validate_query(cls, query: str) -> str:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        clean_query = query.strip()
        max_len = settings.MAX_QUERY_LENGTH
        if len(clean_query) > max_len:
            clean_query = clean_query[:max_len].rstrip()

        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(clean_query):
                raise ValueError("Potentially malicious prompt injection detected in query.")

        return clean_query
