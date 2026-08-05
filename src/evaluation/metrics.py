import re
from typing import List, Dict, Any


class EvaluationMetrics:
    """Automated RAG Quality and Groundedness Metrics."""

    @staticmethod
    def retrieval_precision(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> float:
        if not retrieved_doc_ids:
            return 0.0
        relevant_set = set(relevant_doc_ids)
        hits = sum(1 for d_id in retrieved_doc_ids if d_id in relevant_set)
        return hits / len(retrieved_doc_ids)

    @staticmethod
    def retrieval_recall(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> float:
        if not relevant_doc_ids:
            return 1.0
        relevant_set = set(relevant_doc_ids)
        hits = sum(1 for d_id in retrieved_doc_ids if d_id in relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def context_relevance(query: str, context: str) -> float:
        query_terms = set(re.findall(r"\w+", query.lower())) - {"what", "is", "the", "a", "an", "in", "of", "to", "for"}
        if not query_terms or not context.strip():
            return 0.0
        context_terms = set(re.findall(r"\w+", context.lower()))
        matches = query_terms & context_terms
        return len(matches) / len(query_terms)

    @staticmethod
    def faithfulness(answer: str, context: str) -> float:
        """Percentage of answer sentences grounded in context."""
        if "could not find that information" in answer.lower():
            return 1.0
        sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0
        
        context_lower = context.lower()
        grounded_count = 0
        for sent in sentences:
            keywords = set(re.findall(r"\w+", sent.lower())) - {"the", "is", "a", "an", "and", "or", "to", "of", "in"}
            if not keywords:
                grounded_count += 1
                continue
            matched = sum(1 for kw in keywords if kw in context_lower)
            if matched / len(keywords) >= 0.5:
                grounded_count += 1

        return grounded_count / len(sentences)

    @staticmethod
    def hallucination_rate(answer: str, context: str) -> float:
        return 1.0 - EvaluationMetrics.faithfulness(answer, context)
