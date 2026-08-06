import json
import re
from typing import Any, Dict, List, Optional

from src.memory.models import MemoryExtraction

_SUMMARY_SYSTEM = (
    "You summarize conversations for long-term user memory. "
    "Write 2-4 sentences capturing projects, goals, preferences, and key topics discussed. "
    "Do not include greetings or filler."
)

_EXTRACT_SYSTEM = (
    "Extract durable user facts from the conversation. "
    "Return ONLY valid JSON with keys: preferences, projects, interests, facts, style. "
    "Each list item is a short phrase. style is an object with optional keys: "
    "preferred_answer_style, likes_examples, likes_flowcharts, experience_level. "
    "If nothing durable is present, return empty lists and {} for style."
)

_STOPWORDS = frozenset(
    {
        "about", "after", "again", "also", "and", "are", "ask", "asked", "being",
        "can", "could", "does", "explain", "from", "give", "have", "help", "how",
        "into", "just", "like", "make", "more", "need", "please", "same", "tell",
        "that", "the", "their", "them", "then", "there", "these", "they", "this",
        "what", "when", "where", "which", "with", "would", "your", "you",
    }
)


def format_conversation(chat_history: List[Dict[str, str]], max_chars: int = 12000) -> str:
    lines: List[str] = []
    total = 0
    for turn in chat_history:
        role = (turn.get("role") or "user").capitalize()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        line = f"{role}: {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def extract_topics_from_query(query: str, limit: int = 4) -> List[str]:
    if not query:
        return []
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", query.lower())
    topics: List[str] = []
    seen = set()
    for word in words:
        if word in _STOPWORDS or word.isdigit():
            continue
        if word in seen:
            continue
        seen.add(word)
        topics.append(word)
        if len(topics) >= limit:
            break
    return topics


def _parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


class MemoryExtractor:
    """LLM-powered conversation summarization and fact extraction."""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def summarize_conversation(self, chat_history: List[Dict[str, str]]) -> str:
        conversation = format_conversation(chat_history)
        if not conversation.strip():
            return ""

        prompt = f"Conversation:\n{conversation}\n\nSummary:"
        result = self.llm_provider.chat(
            prompt=prompt,
            system_prompt=_SUMMARY_SYSTEM,
            temperature=0.2,
            max_tokens=256,
        )
        return (result.get("answer") or "").strip()

    def extract_facts(self, chat_history: List[Dict[str, str]]) -> MemoryExtraction:
        conversation = format_conversation(chat_history)
        if not conversation.strip():
            return MemoryExtraction()

        prompt = f"Conversation:\n{conversation}\n\nJSON:"
        result = self.llm_provider.chat(
            prompt=prompt,
            system_prompt=_EXTRACT_SYSTEM,
            temperature=0.0,
            max_tokens=512,
        )
        raw = _parse_json_object(result.get("answer") or "")
        return MemoryExtraction(
            preferences=[str(x) for x in raw.get("preferences") or []],
            projects=[str(x) for x in raw.get("projects") or []],
            interests=[str(x) for x in raw.get("interests") or []],
            facts=[str(x) for x in raw.get("facts") or []],
            style=raw.get("style") if isinstance(raw.get("style"), dict) else {},
        )
