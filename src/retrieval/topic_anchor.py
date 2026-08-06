"""Topic anchoring, source pinning, and retrieval relevance guards for follow-ups."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.metadata.schema import Document
from src.retrieval.source_match import find_matching_sources, fetch_chunks_for_sources

NOT_FOUND_ANSWER = "I could not find that information in the provided knowledge base."

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "with",
    "is", "are", "was", "were", "be", "it", "this", "that", "from", "by", "as",
    "what", "how", "why", "when", "where", "who", "user", "asked", "assistant",
    "answered", "discussed", "about", "strategy", "business", "company",
}

_CONTEXTUAL_FOLLOWUP_PATTERNS = (
    re.compile(r"\bsame\s+(business|company|strategy|model|approach|platform|product|idea|thing)\b", re.I),
    re.compile(r"\b(this|that)\s+(business|company|strategy|model|approach|platform|product|idea|thing)\b", re.I),
    re.compile(r"\bstart\s+(the\s+)?same\b", re.I),
    re.compile(r"\b(the\s+)?same\s+(kind|type|sort)\b", re.I),
    re.compile(r"\breplicate\s+(this|that|it|them)\b", re.I),
    re.compile(r"\bapply\s+(this|that|it)\s+(to|for)\b", re.I),
    re.compile(r"\bfrom\s+scratch\b", re.I),
)


@dataclass
class TopicAnchor:
    """Resolved topic context for retrieval and grounding."""

    topic_terms: List[str] = field(default_factory=list)
    pinned_sources: List[str] = field(default_factory=list)
    anchored_query: str = ""
    enforce_topic: bool = False
    enforce_pinning: bool = False
    is_contextual_followup: bool = False


def is_contextual_topic_followup(query: str, chat_compact: Optional[str] = None) -> bool:
    text = (query or "").strip()
    if not text:
        return False
    if not any(pattern.search(text) for pattern in _CONTEXTUAL_FOLLOWUP_PATTERNS[:6]):
        if not (chat_compact or "").strip():
            return False
        if not _CONTEXTUAL_FOLLOWUP_PATTERNS[6].search(text):
            return False
        return bool(re.search(r"\b(start|build|create|launch|replicate|copy)\b", text, re.I))
    return True


def _split_camel_case(text: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")


def _tokenize(text: str) -> List[str]:
    normalized = _split_camel_case(text.lower())
    return [
        t
        for t in re.findall(r"[a-z0-9]+", normalized)
        if t not in _STOPWORDS and len(t) > 2
    ]


_META_TERMS = {
    "user", "assistant", "prior", "rolling", "latest", "conversation", "summary",
    "discussed", "asked", "answered", "updated", "merged", "session", "chat",
}


def _is_meaningful_topic_term(term: str) -> bool:
    cleaned = (term or "").strip()
    if not cleaned:
        return False
    lower = cleaned.lower()
    if lower in _META_TERMS:
        return False
    if len(cleaned) < 3:
        return False
    if len(cleaned) <= 4 and lower not in {"rag", "ai", "llm", "kit"}:
        return False
    return True


def extract_topic_terms(
    chat_compact: Optional[str],
    routing_turns: Optional[List[Dict[str, str]]],
    indexed_sources: Sequence[Dict],
) -> List[str]:
    blob = (chat_compact or "").strip()
    if not blob:
        return []

    terms: List[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = re.sub(r"\s+", " ", (term or "").strip(" \t\n\r.,;:!?\"'"))
        if len(cleaned) < 3:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append(cleaned)

    blob_lower = blob.lower()
    for item in indexed_sources:
        title = (item.get("title") or "").strip()
        source = item.get("source") or ""
        filename = source.replace("\\", "/").split("/")[-1]
        stem = re.sub(r"\.[a-z0-9]+$", "", filename, flags=re.IGNORECASE)
        for candidate in (title, stem):
            if len(candidate) >= 4 and candidate.lower() in blob_lower:
                add(candidate)
                break

    for source_path in find_matching_sources(blob, indexed_sources):
        title = next(
            (i.get("title") for i in indexed_sources if i.get("source") == source_path),
            "",
        )
        add(title or source_path.replace("\\", "/").split("/")[-1])

    for match in re.finditer(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+|[a-z]+)*)\b", blob):
        add(match.group(1))

    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", blob):
        add(match.group(1))

    return [term for term in terms if _is_meaningful_topic_term(term)][:5]


def _normalize_source(source: str) -> str:
    return (source or "").replace("\\", "/").lower()


def anchor_retrieval_query(retrieval_query: str, topic_terms: Sequence[str]) -> str:
    query = (retrieval_query or "").strip()
    if not query or not topic_terms:
        return query
    primary = topic_terms[0]
    if primary.lower() in query.lower():
        return query
    return f"{primary} - {query}"


def build_topic_anchor(
    clean_query: str,
    retrieval_query: str,
    chat_compact: Optional[str],
    routing_turns: Optional[List[Dict[str, str]]],
    pinned_sources: Optional[List[str]],
    indexed_sources: Sequence[Dict],
) -> TopicAnchor:
    topic_terms = extract_topic_terms(chat_compact, routing_turns, indexed_sources)
    contextual = is_contextual_topic_followup(clean_query, chat_compact)
    has_conversation = bool((chat_compact or "").strip()) or bool(routing_turns)

    pinned: List[str] = []
    for src in pinned_sources or []:
        normalized = (src or "").strip()
        if normalized and normalized not in pinned:
            pinned.append(normalized)

    if topic_terms:
        for src in find_matching_sources(" ".join(topic_terms), indexed_sources):
            if src not in pinned:
                pinned.append(src)

    enforce_pinning = contextual and bool(pinned)
    enforce_topic = (contextual or bool(pinned_sources)) and bool(topic_terms)
    anchored_query = anchor_retrieval_query(retrieval_query, topic_terms) if enforce_topic else retrieval_query

    return TopicAnchor(
        topic_terms=list(topic_terms),
        pinned_sources=pinned[:3],
        anchored_query=anchored_query,
        enforce_topic=enforce_topic,
        enforce_pinning=enforce_pinning,
        is_contextual_followup=contextual,
    )


def _chunk_haystack(doc: Document) -> str:
    meta = doc.metadata
    return " ".join(
        part
        for part in (
            doc.content,
            meta.title or "",
            meta.section or "",
            meta.source or "",
        )
        if part
    ).lower()


def chunk_mentions_topic(doc: Document, topic_terms: Sequence[str]) -> bool:
    if not topic_terms:
        return True
    haystack = _chunk_haystack(doc)
    for term in topic_terms:
        term_lower = term.lower()
        if term_lower in haystack:
            return True
        tokens = _tokenize(term)
        if len(tokens) >= 2 and sum(1 for t in tokens if t in haystack) >= max(1, len(tokens) - 1):
            return True
    return False


def _source_matches_pinned(doc: Document, pinned_sources: Sequence[str]) -> bool:
    if not pinned_sources:
        return False
    doc_source = _normalize_source(doc.metadata.source or "")
    for pinned in pinned_sources:
        normalized = _normalize_source(pinned)
        if normalized and (normalized in doc_source or doc_source.endswith(normalized)):
            return True
    return False


def prioritize_retrieval_results(
    ranked: List[Tuple[Document, float]],
    topic_terms: Sequence[str],
    pinned_sources: Sequence[str],
    *,
    enforce_pinning: bool = False,
) -> List[Tuple[Document, float]]:
    if not ranked:
        return ranked

    topic_relevant: List[Tuple[Document, float]] = []
    pinned_only: List[Tuple[Document, float]] = []
    remainder: List[Tuple[Document, float]] = []

    for doc, score in ranked:
        is_pinned = _source_matches_pinned(doc, pinned_sources)
        is_relevant = chunk_mentions_topic(doc, topic_terms)
        if is_relevant:
            boosted = score + (1.0 if is_pinned else 0.0)
            topic_relevant.append((doc, boosted))
        elif is_pinned:
            pinned_only.append((doc, score))
        else:
            remainder.append((doc, score))

    topic_relevant.sort(key=lambda item: item[1], reverse=True)
    pinned_only.sort(key=lambda item: item[1], reverse=True)
    remainder.sort(key=lambda item: item[1], reverse=True)

    if enforce_pinning and pinned_sources:
        combined = topic_relevant + pinned_only
        return combined or pinned_only or remainder

    return topic_relevant + pinned_only + remainder


def apply_relevance_gate(
    ranked: List[Tuple[Document, float]],
    topic_terms: Sequence[str],
    *,
    min_relevant: int = 1,
) -> Tuple[List[Tuple[Document, float]], bool]:
    if not topic_terms:
        return ranked, True
    relevant = [(doc, score) for doc, score in ranked if chunk_mentions_topic(doc, topic_terms)]
    if len(relevant) >= min_relevant:
        irrelevant = [(doc, score) for doc, score in ranked if not chunk_mentions_topic(doc, topic_terms)]
        return relevant + irrelevant, True
    return ranked, False


def retrieve_pinned_chunks(
    all_documents: Sequence[Document],
    pinned_sources: Sequence[str],
    topic_terms: Sequence[str],
    limit: int = 20,
) -> List[Tuple[Document, float]]:
    if not pinned_sources or not all_documents:
        return []
    docs = fetch_chunks_for_sources(all_documents, pinned_sources, max_per_source=limit)
    scored: List[Tuple[Document, float]] = []
    for idx, doc in enumerate(docs):
        base = 2.5 - (idx * 0.01)
        if chunk_mentions_topic(doc, topic_terms):
            base += 1.0
        scored.append((doc, base))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def should_reject_off_topic_answer(
    answer: str,
    topic_terms: Sequence[str],
    context_str: str,
    citations: Sequence[Dict],
) -> bool:
    if not topic_terms:
        return False
    if NOT_FOUND_ANSWER.lower() in (answer or "").lower():
        return False

    context_lower = (context_str or "").lower()
    topic_in_context = any(term.lower() in context_lower for term in topic_terms)

    citation_blob = " ".join(
        f"{c.get('title', '')} {c.get('source', '')} {c.get('section', '')}"
        for c in citations
    ).lower()
    topic_in_citations = any(term.lower() in citation_blob for term in topic_terms)

    if not topic_in_context and not topic_in_citations:
        return True

    if topic_in_context and not topic_in_citations and citations:
        return True

    return False


def merge_pinned_sources(
    existing: Optional[List[str]],
    citation_sources: Sequence[str],
    topic_terms: Sequence[str],
    indexed_sources: Sequence[Dict],
    *,
    max_sources: int = 3,
) -> List[str]:
    merged: List[str] = []
    for src in list(existing or []) + list(citation_sources):
        cleaned = (src or "").strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    if topic_terms:
        for src in find_matching_sources(" ".join(topic_terms), indexed_sources):
            if src not in merged:
                merged.append(src)
    return merged[:max_sources]
