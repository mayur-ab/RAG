from typing import List, Dict, Any
from src.context.answer_formatter import clean_display_text


class CitationFormatter:
    """Formats source citations for standardized client responses."""

    @staticmethod
    def format_citations(citations: List[Dict[str, Any]]) -> str:
        if not citations:
            return ""

        lines = []
        for cite in citations:
            title = clean_display_text(cite.get("title", "Untitled"))
            section = clean_display_text(cite.get("section") or "")
            if section:
                lines.append(f"[{cite['citation_id']}] {section}")
            else:
                lines.append(f"[{cite['citation_id']}] {title}")

        return "\n".join(lines)
