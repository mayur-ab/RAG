import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metadata.schema import ChunkMetadata, Document
from src.retrieval.topic_anchor import (
    NOT_FOUND_ANSWER,
    anchor_retrieval_query,
    apply_relevance_gate,
    build_topic_anchor,
    chunk_mentions_topic,
    extract_topic_terms,
    is_contextual_topic_followup,
    merge_pinned_sources,
    prioritize_retrieval_results,
    should_reject_off_topic_answer,
)


def _doc(content: str, source: str, title: str) -> Document:
    return Document(
        content=content,
        metadata=ChunkMetadata(
            document_id="doc-1",
            source=source,
            title=title,
            section=title,
        ),
    )


INDEXED = [
    {"source": "C:/Docs/Duo Lingo Strategy.doc", "title": "Duo Lingo Strategy"},
    {"source": "C:/Docs/Dosha Transcript.doc", "title": "Dosha Transcript"},
]


def test_is_contextual_topic_followup_detects_same_business():
    assert is_contextual_topic_followup("how do i start the same business from scratch", "User asked about Duolingo.")
    assert is_contextual_topic_followup("what is this strategy?", "Duolingo strategy discussion.")


def test_extract_topic_terms_from_compact():
    terms = extract_topic_terms(
        "User asked about Duolingo Strategy and gamification.",
        None,
        INDEXED,
    )
    assert any("duolingo" in t.lower() for t in terms)


def test_anchor_retrieval_query_injects_topic():
    anchored = anchor_retrieval_query("how to start a business from scratch", ["Duolingo"])
    assert "Duolingo" in anchored


def test_build_topic_anchor_pins_duolingo_source():
    anchor = build_topic_anchor(
        "how do i start the same business from scratch",
        "how to start a business from scratch",
        "User discussed Duolingo Strategy and gamification.",
        None,
        None,
        INDEXED,
    )
    assert anchor.is_contextual_followup
    assert any("duolingo" in t.lower() for t in anchor.topic_terms)
    assert any("Duo Lingo Strategy.doc" in src for src in anchor.pinned_sources)
    assert "lingo" in anchor.anchored_query.lower()


def test_prioritize_retrieval_results_prefers_topic_chunks():
    duo_doc = _doc("Duolingo uses gamification.", "C:/Docs/Duo Lingo Strategy.doc", "Duo Lingo Strategy")
    dosha_doc = _doc("Vata Pitta Kapha business teams.", "C:/Docs/Dosha Transcript.doc", "Dosha Transcript")
    ranked = [(dosha_doc, 0.9), (duo_doc, 0.8)]
    ordered = prioritize_retrieval_results(
        ranked,
        ["Duolingo"],
        ["C:/Docs/Duo Lingo Strategy.doc"],
        enforce_pinning=True,
    )
    assert "Duolingo" in ordered[0][0].content


def test_apply_relevance_gate_fails_without_topic_match():
    dosha_doc = _doc("Vata Pitta Kapha business teams.", "C:/Docs/Dosha Transcript.doc", "Dosha")
    ranked = [(dosha_doc, 0.9)]
    _, passed = apply_relevance_gate(ranked, ["Duolingo"])
    assert passed is False


def test_should_reject_off_topic_answer():
    answer = "You should balance Vata, Pitta, and Kapha on your team."
    context = "Vata Pitta Kapha business teams need balance."
    citations = [{"title": "Dosha Transcript", "source": "C:/Docs/Dosha Transcript.doc"}]
    assert should_reject_off_topic_answer(answer, ["Duolingo"], context, citations)


def test_merge_pinned_sources_keeps_prior_and_adds_citations():
    merged = merge_pinned_sources(
        ["C:/Docs/Duo Lingo Strategy.doc"],
        ["C:/Docs/Duo Lingo Strategy.doc"],
        ["Duolingo"],
        INDEXED,
    )
    assert merged
    assert any("Duo Lingo Strategy.doc" in src for src in merged)


def test_not_found_constant():
    assert "could not find" in NOT_FOUND_ANSWER.lower()
