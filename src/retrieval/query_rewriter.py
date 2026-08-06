"""Rewrites follow-up questions into standalone retrieval queries using chat history."""

from typing import List, Dict, Optional
from config.logging_config import logger

REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone search queries for a document retrieval system.

Rules:
1. Use the conversation history to resolve pronouns and references (their, it, this, that company, etc.).
2. If the question is already fully standalone, return it unchanged.
3. Output ONLY the rewritten question — no explanation, no quotes, no prefix.
4. Keep the same intent and language as the user's question.
5. Preserve explicit formatting or length constraints (e.g. "5000 words", "bullet list", "table", "short answer").
6. Do not answer the question.

Examples:
History: User asked about Hindustan Pencils address.
Follow-up: "their email id?"
Output: What is the email id of Hindustan Pencils?

History: User asked about Shabda Brahma.
Follow-up: "who discovered it?"
Output: Who discovered Shabda Brahma?"""


class QueryRewriter:
    """History-aware query rewriter (Conversational RAG pattern)."""

    def __init__(self, llm_provider, max_history_turns: int = 5):
        self.llm_provider = llm_provider
        self.max_history_turns = max_history_turns

    def rewrite(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        if not chat_history:
            return query

        recent = chat_history[-(self.max_history_turns * 2) :]
        if not recent:
            return query

        history_lines = []
        for msg in recent:
            role = msg.get("role", "user").strip().lower()
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant"
            history_lines.append(f"{label}: {content}")

        if not history_lines:
            return query

        history_text = "\n".join(history_lines)

        try:
            result = self.llm_provider.generate(
                prompt=f"Follow-up question: {query}",
                context=f"Conversation history:\n{history_text}",
                system_prompt=REWRITE_SYSTEM_PROMPT,
                max_tokens=120,
            )
            rewritten = (result.get("answer") or "").strip().strip('"').strip("'")
            if not rewritten or len(rewritten) < 3:
                return query

            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as exc:
            logger.warning(f"Query rewrite failed, using original query: {exc}")
            return query
