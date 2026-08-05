import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context.answer_formatter import AnswerFormatter


def test_sanitize_removes_source_metadata_lines():
    raw = (
        "According to [2], Shabda Brahma refers to the Sound Universe.\n\n"
        "Source: Temp-Doc\\file.txt | Title: Test Doc | Section: Case Study 3\n"
    )
    cleaned = AnswerFormatter.sanitize_answer(raw)
    assert "Source:" not in cleaned
    assert "Shabda Brahma" in cleaned
    assert "[2]" in cleaned


def test_sanitize_strips_bom_from_citations():
    citations = [{"citation_id": 1, "title": "\ufeffThe Title", "section": None, "source": "s.txt", "chunk_id": "c1", "document_id": "d1"}]
    normalized = AnswerFormatter.normalize_citations(citations)
    assert normalized[0]["title"] == "The Title"


def test_sanitize_removes_trailing_sources_block():
    raw = (
        "Shabda Brahma refers to the idea that the universe began with vibration (Om) "
        "and that sound shapes matter [2]. It is also mentioned as \"the Sound Universe\" "
        "in Case Study 3.\n"
        "Sources:\n"
        "[1] Case Study 2: Vedic Astronomy\n"
        "[2] Case Study 3: Acoustic Rishis (Shabda Brahma - The Sound Universe)\n"
        "[3] Part 1: The Core Objectives\n"
    )
    cleaned = AnswerFormatter.sanitize_answer(raw)
    assert "Sources:" not in cleaned
    assert "Case Study 2" not in cleaned
    assert "Shabda Brahma" in cleaned
    assert "[2]" in cleaned
