from typing import List, Tuple, Dict, Any, Optional
from src.metadata.schema import Document


class ContextBuilder:
    """Builds clean, token-budgeted, deduplicated context blocks for LLM consumption."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def build_context(
        self,
        retrieved_chunks: List[Tuple[Document, float]],
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        seen_contents = set()
        unique_chunks: List[Document] = []

        # 1. Deduplicate chunks
        for doc, _ in retrieved_chunks:
            normalized = doc.content.strip().lower()
            if normalized not in seen_contents:
                seen_contents.add(normalized)
                unique_chunks.append(doc)

        # 2. Preserve document/section ordering
        unique_chunks.sort(key=lambda d: (d.metadata.document_id, d.metadata.section or ""))

        # 3. Format context blocks & collect citations
        context_blocks = []
        citations = []
        estimated_chars = 0
        budget = max_tokens if max_tokens is not None else self.max_tokens
        max_chars = budget * 4  # Roughly 4 chars per token

        for idx, doc in enumerate(unique_chunks, 1):
            section_label = doc.metadata.section or doc.metadata.title or "General"
            section_label = section_label.lstrip("\ufeff").strip()
            block_str = f"[{idx}] Section: {section_label}\n{doc.content.strip()}\n"

            if estimated_chars + len(block_str) > max_chars:
                break  # Prevent token overflow

            context_blocks.append(block_str)
            estimated_chars += len(block_str)

            citations.append({
                "citation_id": idx,
                "document_id": doc.metadata.document_id,
                "chunk_id": doc.metadata.chunk_id,
                "source": doc.metadata.source,
                "title": doc.metadata.title.lstrip("\ufeff").strip(),
                "section": (doc.metadata.section or "").lstrip("\ufeff").strip() or None
            })

        formatted_context = "\n".join(context_blocks)
        return formatted_context, citations
