import re
from typing import List

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "with",
    "is", "are", "was", "were", "be", "it", "this", "that", "from", "by", "as",
    "what", "how", "why", "when", "where", "who", "please", "can", "you", "me",
    "my", "our", "your", "about", "entire", "whole", "all", "understand", "explain",
    "describe", "tell", "give", "based", "using", "into", "small", "children", "childrens",
    "india", "generate", "create", "write", "make", "story", "stories", "summary",
}

_TASK_TAIL_RE = re.compile(
    r"\s+(and|then)\s+(generate|create|write|tell|make|produce|draft)\b.+$",
    re.IGNORECASE,
)
_LEADING_VERB_RE = re.compile(
    r"^(please\s+)?(understand|explain|describe|summarize|analyze|analyse|review)\s+(the\s+)?(entire\s+|whole\s+)?",
    re.IGNORECASE,
)


def _split_camel_case(text: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")


def _tokenize(text: str) -> List[str]:
    normalized = _split_camel_case(text.lower())
    return re.findall(r"[a-z0-9]+", normalized)


def focus_retrieval_query(query: str) -> str:
    """Strip creative/task instructions and keep the document topic for retrieval."""
    text = (query or "").strip()
    text = _LEADING_VERB_RE.sub("", text)
    text = _TASK_TAIL_RE.sub("", text).strip(" ,.-")
    return text or query.strip()


def extract_keyword_query(query: str) -> str:
    tokens = _tokenize(focus_retrieval_query(query))
    keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return " ".join(keywords[:8])


def build_retrieval_queries(query: str) -> List[str]:
    """Build multiple retrieval queries to improve recall on product/doc names."""
    focused = focus_retrieval_query(query)
    keywords = extract_keyword_query(query)
    queries: List[str] = []

    for candidate in (focused, keywords, query.strip()):
        if candidate and candidate not in queries:
            queries.append(candidate)

    lower = query.lower()
    if "deviglow" in lower or "devi glow" in lower:
        for extra in ("DeviGlow Kit Research", "DeviGlow Kit", "Deviglow Kit Research"):
            if extra not in queries:
                queries.append(extra)

    return queries
