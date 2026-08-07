"""Detect and plan long-document generation requests (multi-page reports)."""

import re
from typing import Optional

from config.settings import settings
from src.context.response_length import parse_requested_words

_PAGE_RE = re.compile(
    r"(?:"
    r"(\d{1,3})\s*[-]?\s*(?:page|pages)\b"
    r"|"
    r"(?:page|pages)\s*(?:count|length)?\s*[:=]?\s*(\d{1,3})"
    r")",
    re.IGNORECASE,
)

_LONG_DOC_VERB_RE = re.compile(
    r"\b("
    r"generate|write|create|produce|draft|prepare|compose|build"
    r")\b",
    re.IGNORECASE,
)

_LONG_DOC_NOUN_RE = re.compile(
    r"\b("
    r"research\s+report|full\s+report|detailed\s+report|comprehensive\s+report|"
    r"white\s*paper|whitepaper|thesis|dissertation|documentation|"
    r"long\s+form|multi[- ]page|document|essay|paper"
    r")\b",
    re.IGNORECASE,
)


def parse_requested_pages(query: str) -> Optional[int]:
    if not query:
        return None
    target = None
    for match in _PAGE_RE.finditer(query):
        raw = next((g for g in match.groups() if g), None)
        if raw:
            target = max(target or 0, int(raw))
    return target


def estimate_target_words(query: str) -> Optional[int]:
    """Estimate desired word count from explicit words or page count."""
    words = parse_requested_words(query)
    if words:
        return words
    pages = parse_requested_pages(query)
    if pages:
        return pages * settings.LONG_DOC_WORDS_PER_PAGE
    return None


def is_long_document_request(query: str) -> bool:
    """True when the user asks for a multi-section report that exceeds one-shot output."""
    if not query or not settings.ENABLE_HIERARCHICAL_GENERATION:
        return False

    pages = parse_requested_pages(query)
    if pages and pages >= settings.LONG_DOC_MIN_PAGES:
        return True

    words = parse_requested_words(query)
    if words and words >= settings.LONG_DOC_MIN_WORDS:
        return True

    if _LONG_DOC_VERB_RE.search(query) and _LONG_DOC_NOUN_RE.search(query):
        if pages and pages >= 2:
            return True
        if words and words >= settings.LONG_DOC_MIN_WORDS:
            return True
        # "Generate a research report on X" without explicit length — use default sections
        if _LONG_DOC_NOUN_RE.search(query):
            return True

    return False


def plan_section_count(query: str) -> int:
    """Choose how many sections to generate based on requested length."""
    pages = parse_requested_pages(query)
    words = estimate_target_words(query)

    if pages:
        count = max(settings.LONG_DOC_MIN_SECTIONS, (pages + 1) // 2)
    elif words:
        count = max(settings.LONG_DOC_MIN_SECTIONS, words // settings.LONG_DOC_WORDS_PER_SECTION)
    else:
        count = settings.LONG_DOC_DEFAULT_SECTIONS

    return min(settings.LONG_DOC_MAX_SECTIONS, count)


def words_per_section(query: str, section_count: int) -> int:
    target = estimate_target_words(query)
    if target and section_count > 0:
        return max(200, target // section_count)
    return settings.LONG_DOC_WORDS_PER_SECTION
