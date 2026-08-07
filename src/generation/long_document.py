"""Hierarchical long-document generation: outline → sections → merge."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from config.logging_config import logger
from config.settings import settings
from src.context.document_request import estimate_target_words, plan_section_count, words_per_section
from src.context.response_length import parse_requested_words

OUTLINE_SYSTEM_PROMPT = """You create detailed outlines for long research documents grounded in source material.

Rules:
1. Return ONLY a numbered list of section titles (one title per line).
2. Do not write section bodies or introductions.
3. Titles should be specific and cover the topic comprehensively.
4. Order sections logically (introduction first, conclusion last)."""

SECTION_SYSTEM_PROMPT = """You write ONE section of a longer research document using only the provided context passages.

Rules:
1. Write only the requested section — do not write other sections.
2. Start with a markdown heading (##) matching the section title.
3. Ground claims in the context; cite sources inline with [N] matching context labels.
4. Do NOT add a Sources or References section.
5. If context is insufficient for part of the section, omit that part rather than inventing facts.
6. Maintain continuity with the document outline and prior section summary when provided."""

SUMMARY_SYSTEM_PROMPT = """Summarize the following document section in 2-4 sentences for continuity.
Focus on key points covered. Return plain text only, no heading."""


def parse_outline(text: str, max_sections: int) -> List[str]:
    """Extract section titles from a numbered or markdown outline."""
    sections: List[str] = []
    seen: set[str] = set()

    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(
            r"^(?:"
            r"(?:\d+[\.\):\-]\s*)"
            r"|(?:#{1,3}\s+)"
            r"|(?:[-*•]\s+)"
            r")(.+)$",
            stripped,
        )
        if not match:
            continue
        title = match.group(1).strip().rstrip(".")
        key = title.lower()
        if title and key not in seen and len(title) <= 200:
            seen.add(key)
            sections.append(title)

    if not sections:
        for line in (text or "").splitlines():
            stripped = line.strip()
            if 3 < len(stripped) <= 120 and stripped[0].isupper():
                key = stripped.lower()
                if key not in seen:
                    seen.add(key)
                    sections.append(stripped)

    return sections[:max_sections]


def _default_sections(count: int) -> List[str]:
    templates = [
        "Introduction",
        "Background and Fundamentals",
        "Core Concepts",
        "Architecture and Components",
        "Implementation Strategies",
        "Best Practices",
        "Challenges and Limitations",
        "Case Studies and Examples",
        "Future Directions",
        "Conclusion",
        "References and Further Reading",
        "Appendix",
    ]
    if count <= len(templates):
        return templates[:count]
    extra = [f"Additional Topic {i}" for i in range(1, count - len(templates) + 1)]
    return templates + extra


class LongDocumentGenerator:
    """Generate long reports section-by-section within a fixed context window."""

    def __init__(self, service: Any):
        self.service = service

    def generate(
        self,
        clean_query: str,
        conv: Any,
        roles: List[str],
        filter_metadata: Optional[Dict[str, Any]],
        user_id: Optional[str],
        top_k: int,
        top_m_rerank: int,
        on_stage: Optional[Callable[[str, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        section_count = plan_section_count(clean_query)
        target_words = words_per_section(clean_query, section_count)

        if on_stage:
            on_stage("outline", "Creating document outline...")

        sections = self._build_outline(clean_query, conv, user_id, top_k, top_m_rerank, filter_metadata, roles, section_count)

        all_citations: List[Dict[str, Any]] = []
        citation_offset = 0
        section_bodies: List[str] = []
        previous_summary = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_retrieved = 0
        total_reranked = 0
        model_name = ""

        for idx, title in enumerate(sections, start=1):
            if on_stage:
                on_stage("section", f"Writing section {idx}/{len(sections)}: {title}")

            section_query = f"{clean_query} — {title}"
            ctx_str, citations, reranked, _ = self._retrieve(
                section_query,
                top_k,
                top_m_rerank,
                filter_metadata,
                roles,
                settings.LONG_DOC_SECTION_CONTEXT_TOKENS,
            )
            total_retrieved += len(reranked) + 5
            total_reranked += len(reranked)

            normalized = self._renumber_citations(citations, citation_offset)
            citation_offset += len(normalized)
            all_citations.extend(normalized)

            section_text, usage, model_name = self._generate_section(
                document_topic=clean_query,
                section_title=title,
                section_index=idx,
                section_total=len(sections),
                outline=sections,
                context_str=ctx_str,
                previous_summary=previous_summary,
                target_words=target_words,
                conv=conv,
                user_id=user_id,
                on_token=on_token,
            )
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            section_bodies.append(section_text.strip())

            if idx < len(sections):
                previous_summary = self._summarize_section(section_text)

        if on_stage:
            on_stage("merging", "Combining sections into final document...")

        final_answer = self._assemble_document(clean_query, sections, section_bodies)
        total_latency = round(time.time() - t0, 4)
        generation_res = {
            "model": model_name,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }

        from src.context.answer_formatter import AnswerFormatter
        from src.context.citation import CitationFormatter

        all_citations = AnswerFormatter.normalize_citations(all_citations)

        return {
            "query": clean_query,
            "rewritten_query": None,
            "answer": final_answer,
            "citations": all_citations,
            "formatted_citations": CitationFormatter.format_citations(all_citations),
            "retrieved_context": "",
            "model": model_name,
            "token_usage": self.service._token_usage_from_generation(generation_res),
            "latency": {
                "rewrite_seconds": 0.0,
                "retrieval_seconds": 0.0,
                "rerank_seconds": 0.0,
                "generation_seconds": total_latency,
                "total_seconds": total_latency,
            },
            "retrieved_chunks_count": total_retrieved,
            "reranked_chunks_count": total_reranked,
            "match_percent": 70.0 if all_citations else 0.0,
            "mode": "long_document",
            "cached": False,
            "grounded": bool(all_citations),
            "not_in_documents": not bool(all_citations),
            "long_document": {
                "sections": len(sections),
                "section_titles": sections,
                "target_words": estimate_target_words(clean_query),
                "generation_strategy": "hierarchical",
            },
        }

    def generate_stream(
        self,
        clean_query: str,
        conv: Any,
        roles: List[str],
        filter_metadata: Optional[Dict[str, Any]],
        user_id: Optional[str],
        top_k: int,
        top_m_rerank: int,
    ) -> Iterator[Dict[str, Any]]:
        yield {"type": "stage", "stage": "outline", "message": "Creating document outline..."}

        section_count = plan_section_count(clean_query)
        sections = self._build_outline(
            clean_query, conv, user_id, top_k, top_m_rerank, filter_metadata, roles, section_count
        )
        target_words = words_per_section(clean_query, section_count)

        section_bodies: List[str] = []
        previous_summary = ""
        all_citations: List[Dict[str, Any]] = []
        citation_offset = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_retrieved = 0
        total_reranked = 0
        model_name = ""
        t0 = time.time()

        for idx, title in enumerate(sections, start=1):
            yield {
                "type": "stage",
                "stage": "section",
                "message": f"Writing section {idx}/{len(sections)}: {title}",
            }
            section_query = f"{clean_query} — {title}"
            ctx_str, citations, reranked, _ = self._retrieve(
                section_query,
                top_k,
                top_m_rerank,
                filter_metadata,
                roles,
                settings.LONG_DOC_SECTION_CONTEXT_TOKENS,
            )
            total_retrieved += len(reranked) + 5
            total_reranked += len(reranked)
            normalized = self._renumber_citations(citations, citation_offset)
            citation_offset += len(normalized)
            all_citations.extend(normalized)

            section_text, usage, model_name = self._generate_section(
                document_topic=clean_query,
                section_title=title,
                section_index=idx,
                section_total=len(sections),
                outline=sections,
                context_str=ctx_str,
                previous_summary=previous_summary,
                target_words=target_words,
                conv=conv,
                user_id=user_id,
            )
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            body = section_text.strip()
            section_bodies.append(body)
            yield {"type": "token", "content": body + "\n\n"}
            if idx < len(sections):
                previous_summary = self._summarize_section(section_text)

        yield {"type": "stage", "stage": "merging", "message": "Combining sections..."}

        final_answer = self._assemble_document(clean_query, sections, section_bodies)
        total_latency = round(time.time() - t0, 4)
        generation_res = {
            "model": model_name,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }

        from src.context.answer_formatter import AnswerFormatter
        from src.context.citation import CitationFormatter

        all_citations = AnswerFormatter.normalize_citations(all_citations)
        result = {
            "query": clean_query,
            "rewritten_query": None,
            "answer": final_answer,
            "citations": all_citations,
            "formatted_citations": CitationFormatter.format_citations(all_citations),
            "retrieved_context": "",
            "model": model_name,
            "token_usage": self.service._token_usage_from_generation(generation_res),
            "latency": {
                "rewrite_seconds": 0.0,
                "retrieval_seconds": 0.0,
                "rerank_seconds": 0.0,
                "generation_seconds": total_latency,
                "total_seconds": total_latency,
            },
            "retrieved_chunks_count": total_retrieved,
            "reranked_chunks_count": total_reranked,
            "match_percent": 70.0 if all_citations else 0.0,
            "mode": "long_document",
            "cached": False,
            "grounded": bool(all_citations),
            "not_in_documents": not bool(all_citations),
            "long_document": {
                "sections": len(sections),
                "section_titles": sections,
                "generation_strategy": "hierarchical",
            },
        }
        yield {"type": "done", "data": result}

    def _build_outline(
        self,
        clean_query: str,
        conv: Any,
        user_id: Optional[str],
        top_k: int,
        top_m_rerank: int,
        filter_metadata: Optional[Dict[str, Any]],
        roles: List[str],
        section_count: int,
    ) -> List[str]:
        outline_context, _, _, _ = self._retrieve(
            clean_query,
            top_k,
            top_m_rerank,
            filter_metadata,
            roles,
            settings.LONG_DOC_OUTLINE_CONTEXT_TOKENS,
        )
        outline_text = self._generate_outline(clean_query, outline_context, conv, user_id, section_count)
        sections = parse_outline(outline_text, section_count)
        if len(sections) < settings.LONG_DOC_MIN_SECTIONS:
            sections = _default_sections(section_count)
            logger.info("Using fallback outline sections for long document generation")
        return sections

    def _retrieve(
        self,
        query: str,
        top_k: int,
        top_m_rerank: int,
        filter_metadata: Optional[Dict[str, Any]],
        roles: List[str],
        context_tokens: int,
    ) -> Tuple[str, List[Dict[str, Any]], List[Any], str]:
        retrieved, rerank_query = self.service._hybrid_retrieve(
            clean_query=query,
            retrieval_query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            roles=roles,
        )
        reranked = self.service.reranker.rerank(
            query=rerank_query, documents=retrieved, top_n=top_m_rerank
        )
        context_str, citations = self.service.context_builder.build_context(
            reranked, max_tokens=context_tokens
        )
        return context_str, citations, reranked, rerank_query

    def _generate_outline(
        self,
        topic: str,
        context: str,
        conv: Any,
        user_id: Optional[str],
        section_count: int,
    ) -> str:
        system = self.service._apply_user_memory(OUTLINE_SYSTEM_PROMPT, user_id, topic)
        prompt = (
            f"Topic: {topic}\n\n"
            f"Create a numbered outline with exactly {section_count} section titles.\n\n"
            f"Reference material:\n{context or '(No reference material retrieved)'}"
        )
        res = self.service.llm_provider.generate(
            prompt=prompt,
            context="",
            system_prompt=system,
            max_tokens=settings.LONG_DOC_OUTLINE_MAX_TOKENS,
            chat_compact=conv.chat_compact,
        )
        return res.get("answer", "")

    def _generate_section(
        self,
        document_topic: str,
        section_title: str,
        section_index: int,
        section_total: int,
        outline: List[str],
        context_str: str,
        previous_summary: str,
        target_words: int,
        conv: Any,
        user_id: Optional[str],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, Dict[str, int], str]:
        system = self.service._apply_user_memory(SECTION_SYSTEM_PROMPT, user_id, document_topic)
        outline_block = "\n".join(f"{i}. {t}" for i, t in enumerate(outline, start=1))
        parts = [
            f"Document topic: {document_topic}",
            f"Full outline:\n{outline_block}",
            f"Write section {section_index} of {section_total}: {section_title}",
            f"Target length: approximately {target_words} words.",
        ]
        if previous_summary:
            parts.append(f"Previous section summary:\n{previous_summary}")
        parts.append(f"Current question: Write the section titled '{section_title}'.")

        prompt = "\n\n".join(parts)
        max_tokens = min(
            settings.LONG_DOC_SECTION_OUTPUT_TOKENS,
            max(512, int(target_words * 1.35)),
        )

        stream_fn = getattr(self.service.llm_provider, "generate_stream", None)
        if on_token and stream_fn:
            chunks: List[str] = []
            for token in stream_fn(
                prompt=prompt,
                context=context_str,
                system_prompt=system,
                max_tokens=max_tokens,
                chat_compact=conv.chat_compact,
            ):
                chunks.append(token)
                on_token(token)
            answer = "".join(chunks).strip()
            usage = getattr(self.service.llm_provider, "last_stream_usage", {}) or {}
            model = getattr(self.service.llm_provider, "model", "unknown")
            if settings.LLM_PROVIDER == "ollama":
                model = f"ollama/{model}"
            return answer, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", len(answer.split())),
            }, model

        res = self.service.llm_provider.generate(
            prompt=prompt,
            context=context_str,
            system_prompt=system,
            max_tokens=max_tokens,
            chat_compact=conv.chat_compact,
        )
        return res.get("answer", ""), {
            "prompt_tokens": res.get("prompt_tokens", 0),
            "completion_tokens": res.get("completion_tokens", 0),
        }, res.get("model", "unknown")

    def _summarize_section(self, section_text: str) -> str:
        if not section_text or len(section_text) < 100:
            return section_text[: settings.LONG_DOC_SECTION_SUMMARY_MAX_CHARS]
        trimmed = section_text[: settings.LONG_DOC_SECTION_SUMMARY_MAX_CHARS * 3]
        res = self.service.llm_provider.chat(
            prompt=f"Section text:\n{trimmed}\n\nSummary:",
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            max_tokens=256,
        )
        summary = (res.get("answer") or "").strip()
        return summary[: settings.LONG_DOC_SECTION_SUMMARY_MAX_CHARS]

    @staticmethod
    def _renumber_citations(citations: List[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
        renumbered = []
        for i, cite in enumerate(citations, start=1):
            updated = dict(cite)
            updated["citation_id"] = offset + i
            renumbered.append(updated)
        return renumbered

    @staticmethod
    def _assemble_document(topic: str, sections: List[str], bodies: List[str]) -> str:
        parts = [f"# {topic.strip()}\n"]
        for title, body in zip(sections, bodies):
            if body:
                if not body.lstrip().startswith("#"):
                    parts.append(f"## {title}\n\n{body}")
                else:
                    parts.append(body)
            else:
                parts.append(f"## {title}\n\n(Section could not be generated.)")
        return "\n\n".join(parts).strip()
