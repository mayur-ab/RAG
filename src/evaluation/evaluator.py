from typing import List, Dict, Any
from src.evaluation.metrics import EvaluationMetrics


class RAGEvaluator:
    """Automated evaluation engine for testing retrieval and generation quality."""

    def evaluate_query_response(
        self,
        query: str,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        context: str,
        answer: str
    ) -> Dict[str, float]:

        precision = EvaluationMetrics.retrieval_precision(retrieved_doc_ids, relevant_doc_ids)
        recall = EvaluationMetrics.retrieval_recall(retrieved_doc_ids, relevant_doc_ids)
        ctx_relevance = EvaluationMetrics.context_relevance(query, context)
        faithfulness = EvaluationMetrics.faithfulness(answer, context)
        hallucination_rate = EvaluationMetrics.hallucination_rate(answer, context)

        return {
            "retrieval_precision": round(precision, 4),
            "retrieval_recall": round(recall, 4),
            "context_relevance": round(ctx_relevance, 4),
            "faithfulness": round(faithfulness, 4),
            "hallucination_rate": round(hallucination_rate, 4)
        }
