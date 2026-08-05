import re
from typing import List, Dict, Any


def clean_display_text(text: str) -> str:
    """Remove UTF-8 BOM and extra whitespace from display strings."""
    if not text:
        return text
    return text.lstrip("\ufeff").strip()


class AnswerFormatter:
    """Cleans LLM answers so source metadata is not duplicated in the response body."""

    _SOURCES_HEADER = re.compile(
        r"^(###\s*)?(Sources?|References?|Citations?)(\s*&\s*Citations?)?\s*:?\s*$",
        re.IGNORECASE,
    )

    _METADATA_LINE_PATTERNS = [
        re.compile(r"^Source\s*:", re.IGNORECASE),
        re.compile(r"^Title\s*:", re.IGNORECASE),
        re.compile(r"^Chunk ID\s*:", re.IGNORECASE),
        re.compile(r"^---\s*\[DOCUMENT\s+\d+\]\s*---", re.IGNORECASE),
        re.compile(r"^Content\s*:\s*$", re.IGNORECASE),
    ]

    _CITATION_LIST_LINE = re.compile(r"^\[\d+\]\s+")

    @classmethod
    def sanitize_answer(cls, answer: str) -> str:
        if not answer or not answer.strip():
            return answer

        lines = answer.splitlines()

        # Truncate at "Sources:", "References:", etc.
        truncate_at = len(lines)
        for i, line in enumerate(lines):
            if cls._SOURCES_HEADER.match(line.strip()):
                truncate_at = i
                break
        lines = lines[:truncate_at]

        cleaned_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                continue

            if any(pattern.search(stripped) for pattern in cls._METADATA_LINE_PATTERNS):
                continue

            if cls._CITATION_LIST_LINE.match(stripped):
                continue

            if re.match(r"^- \[\d+\]\s+\*\*", stripped):
                continue

            cleaned_lines.append(line.rstrip())

        result = "\n".join(cleaned_lines).strip()
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result

    @staticmethod
    def normalize_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for cite in citations:
            entry = dict(cite)
            entry["title"] = clean_display_text(entry.get("title", ""))
            entry["section"] = clean_display_text(entry.get("section") or "") or None
            entry["source"] = clean_display_text(entry.get("source", ""))
            normalized.append(entry)
        return normalized
