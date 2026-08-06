"""Rewrites follow-up questions into standalone retrieval queries using conversation compact."""

from typing import List, Dict, Optional
from config.logging_config import logger
from src.context.conversation import build_rewriter_context
from src.context.user_memory_messages import should_skip_query_rewrite

REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone search queries for a document retrieval system.

Rules:
1. Use the conversation summary to resolve pronouns and references (their, it, this, that company, etc.).
2. Preserve named entities and product/company names from the summary (e.g. Duolingo, DeviGlow).
3. If the question is already fully standalone, return it unchanged.
4. Output ONLY the rewritten question — no explanation, no quotes, no prefix.
5. Keep the same intent and language as the user's question.
6. Preserve explicit formatting or length constraints (e.g. "5000 words", "bullet list", "table", "short answer").
7. Do not answer the question."""


class QueryRewriter:
    """History-aware query rewriter using compact conversation context."""

    def __init__(self, llm_provider, max_history_turns: int = 5):
        self.llm_provider = llm_provider
        self.max_history_turns = max_history_turns

    def rewrite(
        self,
        query: str,
        chat_compact: Optional[str] = None,
        routing_turns: Optional[List[Dict[str, str]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if should_skip_query_rewrite(query):
            return query

        context = build_rewriter_context(chat_compact, routing_turns or chat_history)
        if not context.strip():
            return query

        try:
            result = self.llm_provider.generate(
                prompt=f"Follow-up question: {query}",
                context=context,
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
