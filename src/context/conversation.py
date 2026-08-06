import re
from typing import Dict, List, Optional

from config.settings import settings
from src.context.user_memory_messages import is_user_memory_message, should_skip_query_rewrite


_REFINEMENT_PATTERNS = (
    re.compile(r"\b(shorter|longer|brief(ly)?|funny|humor(?:ous)?|simple|r|simpler|elaborate|expand|summarize|continue|rephrase|rewrite|paraphrase|translate|translation)\b", re.I),
    re.compile(r"\b\d+\s*words?\b", re.I),
    re.compile(r"^(okay|ok|yes|sure|go on|tell me more|and)\b", re.I),
    re.compile(r"\b(in|using|to|into)\s+\w+\s+(way|style|tone|format|language)\b", re.I),
    re.compile(r"\b(in|to|into)\s+(hindi|japanese|english|tamil|telugu|marathi|bengali|gujarati|kannada|malayalam|spanish|french|german|chinese|korean|arabic|punjabi|urdu)\b", re.I),
    re.compile(r"\b(give|show|write|do)\s+same\b", re.I),
    re.compile(r"\bsame\s+(text|thing|content|answer)?\s*(in|as)\b", re.I),
    re.compile(r"\bconvert\s+(it|this|that|them)?\s*(to|into)\b", re.I),
    re.compile(r"\bonly\b", re.I),
)

_CONVERSATIONAL_ONLY_RE = re.compile(
    r"^(hey|hi|hello|hii|yo|thanks|thank you|ok|okay|hmm|\?{1,3}|!{1,3}|\?\?)[\?\!\.\s]*$",
    re.IGNORECASE,
)

FOLLOWUP_SYSTEM_PROMPT = """You are continuing an ongoing conversation in a document chat app.

Rules:
1. Use the conversation summary provided with the user message — not document retrieval.
2. When the user asks to translate, rewrite, reformat, or repeat something, use topics from the summary.
3. When the user says "same", "this", or "these", they refer to the most recent relevant answer in the summary.
4. For greetings (hey, hi) or confusion (??), respond naturally and briefly.
5. Do not say you could not find information in the knowledge base for translation or reformatting tasks.
6. Do not add citation numbers or source sections."""


def is_refinement_follow_up(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    if len(text.split()) <= 16:
        return any(pattern.search(text) for pattern in _REFINEMENT_PATTERNS)
    return False


def is_conversational_only(query: str) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    if len(text) <= 3:
        return True
    return bool(_CONVERSATIONAL_ONLY_RE.match(text))


def requires_conversation_only(
    query: str,
    routing_turns: Optional[List[Dict[str, str]]] = None,
    chat_compact: Optional[str] = None,
) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    if is_user_memory_message(text):
        return True
    has_context = bool((chat_compact or "").strip()) or bool(routing_turns)
    if not has_context:
        if is_user_memory_message(text):
            return True
        return False
    if is_conversational_only(text):
        return True
    if is_refinement_follow_up(text) and (last_assistant_message(routing_turns) or (chat_compact or "").strip()):
        return True
    if re.search(r"\b(from these|from this|from the above|above text|previous (answer|response|message))\b", text, re.I):
        return True
    return False


def last_assistant_message(routing_turns: Optional[List[Dict[str, str]]]) -> Optional[str]:
    if not routing_turns:
        return None
    for msg in reversed(routing_turns):
        if msg.get("role") == "assistant":
            content = (msg.get("content") or "").strip()
            if content:
                return content
    return None


def last_user_message(routing_turns: Optional[List[Dict[str, str]]]) -> Optional[str]:
    if not routing_turns:
        return None
    for msg in reversed(routing_turns):
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content:
                return content
    return None


def format_routing_turns(routing_turns: Optional[List[Dict[str, str]]]) -> str:
    if not routing_turns:
        return ""
    lines: List[str] = []
    for msg in routing_turns[-settings.MAX_ROUTING_TURNS * 2 :]:
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
    chat_compact: Optional[str] = None,
) -> str:
    parts = [f"Context:\n{context}"]
    compact = (chat_compact or "").strip()
    if compact:
        parts.append(
            "Conversation summary so far (the current question may refer to this; "
            "honor any style or length instructions in the current question):\n"
            f"{compact}"
        )
    parts.append(f"Current question: {query}")
    return "\n\n".join(parts)


def build_rewriter_context(
    chat_compact: Optional[str] = None,
    routing_turns: Optional[List[Dict[str, str]]] = None,
) -> str:
    parts: List[str] = []
    compact = (chat_compact or "").strip()
    if compact:
        parts.append(f"Conversation summary:\n{compact}")
    recent = format_routing_turns(routing_turns)
    if recent:
        parts.append(f"Most recent exchange:\n{recent}")
    return "\n\n".join(parts)


def resolve_retrieval_query(
    clean_query: str,
    chat_compact: Optional[str],
    routing_turns: Optional[List[Dict[str, str]]],
    rewriter,
    enable_rewriting: bool,
) -> str:
    if should_skip_query_rewrite(clean_query):
        return clean_query

    has_context = bool((chat_compact or "").strip()) or bool(routing_turns)
    if not has_context:
        return clean_query

    if is_refinement_follow_up(clean_query):
        prior = last_user_message(routing_turns)
        if prior and prior.lower() != clean_query.lower():
            merged = f"{prior} — {clean_query}"
            return _inject_topic_from_compact(merged, chat_compact)
        return _inject_topic_from_compact(clean_query, chat_compact)

    rewritten = clean_query
    if enable_rewriting:
        rewritten = rewriter.rewrite(clean_query, chat_compact=chat_compact, routing_turns=routing_turns)
    return _inject_topic_from_compact(rewritten, chat_compact)


def _inject_topic_from_compact(query: str, chat_compact: Optional[str]) -> str:
    """Ensure primary topic from compact is present in retrieval query for follow-ups."""
    from src.retrieval.topic_anchor import extract_topic_terms, anchor_retrieval_query, is_contextual_topic_followup

    if not is_contextual_topic_followup(query, chat_compact):
        return query
    terms = extract_topic_terms(chat_compact, None, [])
    if not terms:
        return query
    return anchor_retrieval_query(query, terms)
