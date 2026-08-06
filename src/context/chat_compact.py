from typing import Optional

from config.settings import settings

_CHAT_COMPACT_SYSTEM = (
    "You maintain a rolling conversation summary for a RAG chat app. "
    "Merge the prior summary with the new user/assistant exchange. "
    "Keep names, topics, goals, constraints, and open questions. "
    "Write 2-6 concise sentences. No greetings or filler."
)

_SESSION_MERGE_SYSTEM = (
    "You merge chat summaries from one user session into a single session summary. "
    "Keep durable topics, projects, and preferences across chats. "
    "Write 3-8 concise sentences. No greetings."
)

_SESSION_FINALIZE_SYSTEM = (
    "You finalize a user session summary for long-term memory. "
    "Write 3-6 sentences capturing what the user worked on and learned. "
    "No greetings."
)


def _truncate(text: str, max_chars: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


class ChatCompactService:
    """Ephemeral per-chat compaction and session-level merging."""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider

    def update_chat_compact(
        self,
        prior_compact: str,
        user_message: str,
        assistant_message: str,
    ) -> str:
        user_message = (user_message or "").strip()
        assistant_message = (assistant_message or "").strip()
        if not user_message and not assistant_message:
            return prior_compact or ""

        prompt = (
            f"Prior summary:\n{prior_compact or '(none)'}\n\n"
            f"New exchange:\nUser: {user_message}\nAssistant: {assistant_message}\n\n"
            "Updated summary:"
        )
        result = self.llm_provider.chat(
            prompt=prompt,
            chat_history=None,
            system_prompt=_CHAT_COMPACT_SYSTEM,
            temperature=0.1,
            max_tokens=320,
        )
        updated = (result.get("answer") or "").strip()
        if not updated:
            parts = [p for p in [prior_compact, f"User asked: {user_message}", f"Assistant answered: {assistant_message[:200]}"] if p]
            updated = " ".join(parts)
        return _truncate(updated, settings.CHAT_COMPACT_MAX_CHARS)

    def merge_into_session(self, session_compact: str, chat_compact: str) -> str:
        chat_compact = (chat_compact or "").strip()
        if not chat_compact:
            return session_compact or ""
        if not (session_compact or "").strip():
            return _truncate(chat_compact, settings.SESSION_COMPACT_MAX_CHARS)

        prompt = (
            f"Existing session summary:\n{session_compact}\n\n"
            f"New chat summary:\n{chat_compact}\n\n"
            "Merged session summary:"
        )
        result = self.llm_provider.chat(
            prompt=prompt,
            chat_history=None,
            system_prompt=_SESSION_MERGE_SYSTEM,
            temperature=0.1,
            max_tokens=384,
        )
        merged = (result.get("answer") or "").strip()
        if not merged:
            merged = f"{session_compact}\n{chat_compact}".strip()
        return _truncate(merged, settings.SESSION_COMPACT_MAX_CHARS)

    def finalize_session_summary(self, working_compact: str) -> str:
        working_compact = (working_compact or "").strip()
        if not working_compact:
            return ""
        result = self.llm_provider.chat(
            prompt=f"Session notes:\n{working_compact}\n\nFinal session summary:",
            chat_history=None,
            system_prompt=_SESSION_FINALIZE_SYSTEM,
            temperature=0.1,
            max_tokens=384,
        )
        summary = (result.get("answer") or "").strip() or working_compact
        return _truncate(summary, settings.SESSION_COMPACT_MAX_CHARS)


def build_compact_user_message(query: str, chat_compact: Optional[str] = None) -> str:
    compact = (chat_compact or "").strip()
    if not compact:
        return query
    return (
        "Conversation summary so far (use for follow-ups; do not invent prior topics):\n"
        f"{compact}\n\n"
        f"Current message: {query}"
    )
