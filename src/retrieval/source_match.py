import re
from typing import Dict, List, Sequence, Set

from src.metadata.schema import Document

_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "with",
    "is", "are", "doc", "pdf", "txt", "xls", "xlsx", "what", "how", "understand",
    "explain", "generate", "story", "based", "small", "children", "childrens", "india",
}


def _split_camel_case(text: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")


def _tokenize(text: str) -> Set[str]:
    normalized = _split_camel_case(text.lower())
    return {t for t in re.findall(r"[a-z0-9]+", normalized) if t not in _STOPWORDS and len(t) > 2}


def _expand_query_tokens(tokens: Set[str]) -> Set[str]:
    expanded = set(tokens)
    compact = "".join(sorted(tokens))
    if "deviglow" in compact or ("devi" in tokens and "glow" in tokens):
        expanded.update({"devi", "glow", "deviglow", "kit", "research"})
    if "duolingo" in compact or ("duo" in tokens and "lingo" in tokens):
        expanded.update({"duo", "lingo", "duolingo", "strategy"})
    return expanded


def find_matching_sources(query: str, indexed_sources: Sequence[Dict]) -> List[str]:
    """Match user query tokens against indexed filenames and titles."""
    focused = query.strip()
    query_tokens = _expand_query_tokens(_tokenize(focused))
    if not query_tokens:
        return []

    scored: List[tuple[int, str]] = []
    for item in indexed_sources:
        source = item.get("source") or ""
        title = item.get("title") or ""
        filename = source.replace("\\", "/").split("/")[-1]
        stem = re.sub(r"\.[a-z0-9]+$", "", filename, flags=re.IGNORECASE)
        haystack = f"{title} {stem} {filename}"
        doc_tokens = _tokenize(haystack)
        if not doc_tokens:
            continue

        overlap = len(query_tokens & doc_tokens)
        compact_query = re.sub(r"\s+", "", focused.lower())
        compact_name = re.sub(r"\s+", "", haystack.lower())
        if compact_query and compact_query in compact_name:
            overlap += 8

        wants_deviglow = "deviglow" in compact_query or ("devi" in query_tokens and "glow" in query_tokens)
        if wants_deviglow and "glow" not in doc_tokens and "deviglow" not in compact_name:
            continue

        # Require at least two meaningful overlaps for generic words like "kit" / "research"
        if overlap >= 2 or (overlap >= 1 and ("devi" in doc_tokens and "glow" in doc_tokens)):
            scored.append((overlap, source))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [source for _, source in scored[:3]]


def fetch_chunks_for_sources(
    documents: Sequence[Document],
    source_paths: Sequence[str],
    max_per_source: int = 15,
) -> List[Document]:
    if not source_paths:
        return []

    matched: List[Document] = []
    normalized_paths = [p.replace("\\", "/").lower() for p in source_paths]

    for source_path in normalized_paths:
        per_source: List[Document] = []
        for doc in documents:
            doc_source = (doc.metadata.source or "").replace("\\", "/").lower()
            if source_path in doc_source or doc_source.endswith(source_path):
                per_source.append(doc)
        matched.extend(per_source[:max_per_source])

    return matched
