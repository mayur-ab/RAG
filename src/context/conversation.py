import re
from typing import Dict, List, Optional

from config.settings import settings

_REFINEMENT_PATTERNS = (
    re.compile(r"\b(shorter|longer|brief(ly)?|funny|humor(?:ous)?|simple|r|simpler|elaborate|expand|summarize|continue|rephrase|rewrite)\b", re.I),
    re.compile(r"\b\d+\s*words?\b", re.I),
    re.compile(r"^(okay|ok|yes|sure|go on|tell me more|and)\b", re.I),
    re.compile(r"\b(in|using)\s+\w+\s+(way|style|tone|format)\b", re.I),
    re.compile(r"\bonly\b", re.I),
)


def is_refinement_follow_up(query: str) -> bool:
    """Detect style/format/length follow-ups that should not be fully rewritten."""
    text = (query or "").strip()
    if not text:
        return False
    if len(text.split()) <= 12:
        return any(pattern.search(text) for pattern in _REFINEMENT_PATTERNS)
    return False


def last_user_message(chat_history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    if not chat_history:
        return None
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content:
                return content
    return None


def format_chat_history(chat_history: Optional[List[Dict[str, str]]], max_turns: Optional[int] = None) -> str:
    if not chat_history:
        return ""

    limit = (max_turns or settings.MAX_CHAT_HISTORY_TURNS) * 2
    recent = chat_history[-limit:]
    lines: List[str] = []
    for msg in recent:
        role = (msg.get("role") or "user").strip().lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def build_rag_user_message(
    context: str,
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    parts = [f"Context:\n{context}"]
    history_text = format_chat_history(chat_history)
    if history_text:
        parts.append(
            "Conversation so far (the current question may refer to this; "
            "honor any style or length instructions in the current question):\n"
            f"{history_text}"
        )
    parts.append(f"Current question: {query}")
    return "\n\n".join(parts)


def resolve_retrieval_query(
    clean_query: str,
    chat_history: Optional[List[Dict[str, str]]],
    rewriter,
    enable_rewriting: bool,
) -> str:
    if not chat_history:
        return clean_query

    if is_refinement_follow_up(clean_query):
        prior = last_user_message(chat_history)
        if prior and prior.lower() != clean_query.lower():
            return f"{prior} — {clean_query}"
        return clean_query

    if enable_rewriting:
        return rewriter.rewrite(clean_query, chat_history)
    return clean_query
