import re
from typing import Dict, List, Optional

_NAME_CAPTURE_RE = re.compile(
    r"(?:"
    r"my name is\s+([A-Za-z][A-Za-z'-]{0,30})"
    r"|(?:i am|i'm|call me|this is)\s+([A-Za-z][A-Za-z'-]{0,30})"
    r"|remember (?:that )?my name is\s+([A-Za-z][A-Za-z'-]{0,30})"
    r")",
    re.IGNORECASE,
)

_USER_MEMORY_PATTERNS = (
    re.compile(r"\bmy name is\b", re.I),
    re.compile(r"\b(?:i'?m|i am|call me|this is)\s+[A-Za-z]", re.I),
    re.compile(r"\bremember my\b", re.I),
    re.compile(r"\bdon'?t forget my\b", re.I),
    re.compile(r"\bplease remember\b", re.I),
    re.compile(r"\bkeep in mind\b", re.I),
)

_SKIP_REWRITE_PATTERNS = _USER_MEMORY_PATTERNS + (
    re.compile(r"\bwhat is my name\b", re.I),
    re.compile(r"\bdo you remember my name\b", re.I),
)

MEMORY_CONVERSATION_PROMPT = """You are handling a personal introduction or memory request in a chat app.

Rules:
1. If the user introduces themselves or asks you to remember their name, acknowledge warmly and confirm you will remember it.
2. Answer from the conversation and stored user profile — do not search documents or explain name etymology unless explicitly asked.
3. Keep replies brief (1-2 sentences) unless the user asks for more.
4. Do not add citation numbers or source sections."""


def is_user_memory_message(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _USER_MEMORY_PATTERNS)


def should_skip_query_rewrite(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _SKIP_REWRITE_PATTERNS)


def extract_user_name(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
    """Extract a display name from the current message or recent user turns."""
    for text in _message_texts_to_scan(query, chat_history):
        match = _NAME_CAPTURE_RE.search(text)
        if match:
            name = next((g for g in match.groups() if g), None)
            if name:
                cleaned = _clean_name(name)
                if cleaned:
                    return cleaned
    return None


def _message_texts_to_scan(query: str, chat_history: Optional[List[Dict[str, str]]]) -> List[str]:
    texts = [(query or "").strip()]
    if chat_history:
        for msg in reversed(chat_history):
            if (msg.get("role") or "").lower() == "user":
                content = (msg.get("content") or "").strip()
                if content and content not in texts:
                    texts.append(content)
    return texts


def _clean_name(raw: str) -> Optional[str]:
    name = (raw or "").strip(" .,!?:;\"'")
    if not name or len(name) < 2:
        return None
    lowered = name.lower()
    if lowered in {"the", "a", "an", "here", "there", "please", "hey", "hi", "hello"}:
        return None
    return name[0].upper() + name[1:]
