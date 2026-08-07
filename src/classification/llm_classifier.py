import json
import re
from typing import Any, List, Optional

from config.settings import settings
from src.classification.models import DocumentClassification

CLASSIFY_SYSTEM_PROMPT = """You classify enterprise documents for a knowledge management system.

Return ONLY valid JSON with this exact shape:
{
  "category": "one of the allowed categories",
  "subcategory": "specific topic area",
  "tags": ["tag1", "tag2", "tag3"],
  "summary": "one sentence summary"
}

Rules:
- category MUST be one of the allowed categories listed in the user message
- subcategory should be specific (e.g. RAG, Payroll, Contracts, API)
- tags: 3-6 lowercase relevant keywords
- summary: max 25 words
- Return JSON only, no markdown fences"""


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _normalize_category(raw: str, allowed: List[str]) -> str:
    if not raw:
        return "Uncategorized"
    cleaned = raw.strip()
    for cat in allowed:
        if cleaned.lower() == cat.lower():
            return cat
    return "Uncategorized"


class LLMDocumentClassifier:
    """LLM-based document classification during ingestion."""

    def __init__(self, llm_provider, categories: Optional[List[str]] = None):
        self.llm = llm_provider
        self.categories = categories or list(settings.DOC_CATEGORIES)

    def classify(self, title: str, text_sample: str) -> DocumentClassification:
        sample = (text_sample or "")[: settings.DOC_CLASSIFY_SAMPLE_CHARS]
        categories_block = ", ".join(self.categories)
        prompt = (
            f"Allowed categories: {categories_block}\n\n"
            f"File name: {title}\n\n"
            f"Document excerpt:\n{sample or '(empty document)'}\n\n"
            "Return JSON only."
        )
        res = self.llm.chat(
            prompt=prompt,
            system_prompt=CLASSIFY_SYSTEM_PROMPT,
            max_tokens=384,
            temperature=0.0,
        )
        parsed = _extract_json(res.get("answer", ""))
        if not parsed:
            return DocumentClassification(
                category="Uncategorized",
                subcategory="General",
                tags=[],
                summary=f"Document: {title}",
            )

        tags = parsed.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        tags = [str(t).strip().lower() for t in tags if str(t).strip()][:8]

        return DocumentClassification(
            category=_normalize_category(str(parsed.get("category", "")), self.categories),
            subcategory=str(parsed.get("subcategory") or "General").strip()[:120],
            tags=tags,
            summary=str(parsed.get("summary") or "").strip()[:500],
        )


CLUSTER_NAME_SYSTEM = """You name groups of similar documents for a knowledge library.
Return ONLY a short collection name (2-5 words), no quotes, no JSON."""


class LLMClusterNamer:
    def __init__(self, llm_provider):
        self.llm = llm_provider

    def name_cluster(self, titles: List[str]) -> str:
        sample = titles[:12]
        listing = "\n".join(f"- {t}" for t in sample)
        prompt = (
            "These documents belong together:\n\n"
            f"{listing}\n\n"
            "Give a short collection name."
        )
        res = self.llm.chat(
            prompt=prompt,
            system_prompt=CLUSTER_NAME_SYSTEM,
            max_tokens=32,
            temperature=0.0,
        )
        name = (res.get("answer") or "Related Documents").strip().strip('"').strip("'")
        return name[:80] or "Related Documents"
