import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.query_focus import focus_retrieval_query, build_retrieval_queries
from src.retrieval.source_match import find_matching_sources


def test_focus_retrieval_query_strips_story_instruction():
    query = (
        "understand entire deviglow research kit and generate a story "
        "based on it for small childrens in india."
    )
    focused = focus_retrieval_query(query)
    assert "generate" not in focused.lower()
    assert "story" not in focused.lower()
    assert "deviglow" in focused.lower()


def test_build_retrieval_queries_includes_deviglow_doc_hint():
    query = "understand entire deviglow research kit and generate a story"
    queries = build_retrieval_queries(query)
    assert any("DeviGlow Kit Research" in q for q in queries)


def test_find_matching_sources_deviglow():
    indexed = [
        {"source": "C:/Docs/DeviGlow Kit Research.doc", "title": "Deviglow Kit Research"},
        {"source": "C:/Docs/Dinosaur Kit Market and Material Research.doc", "title": "Dinosaur Kit"},
    ]
    matches = find_matching_sources("deviglow research kit", indexed)
    assert any("DeviGlow Kit Research.doc" in path for path in matches)


def test_find_matching_sources_duolingo():
    indexed = [
        {"source": "C:/Docs/Duo Lingo Strategy.doc", "title": "Duo Lingo Strategy"},
        {"source": "C:/Docs/Other.doc", "title": "Other"},
    ]
    matches = find_matching_sources("Duolingo Strategy", indexed)
    assert any("Duo Lingo Strategy.doc" in path for path in matches)
