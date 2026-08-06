import re
from typing import List, Dict, Any, Optional

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.llm_judge import LLMJudge


class RAGEvaluator:
    """Automated evaluation engine for retrieval and generation quality."""

    def evaluate_query_response(
        self,
        query: str,
        retrieved_doc_ids: List[str],
        context: str,
        answer: str,
        relevant_doc_ids: Optional[List[str]] = None,
        llm_judge: Optional[LLMJudge] = None,
    ) -> Dict[str, Any]:
        heuristic = {
            "context_relevance": round(EvaluationMetrics.context_relevance(query, context), 4),
            "faithfulness": round(EvaluationMetrics.faithfulness(answer, context), 4),
            "hallucination_rate": round(EvaluationMetrics.hallucination_rate(answer, context), 4),
        }

        if relevant_doc_ids:
            heuristic["retrieval_precision"] = round(
                EvaluationMetrics.retrieval_precision(retrieved_doc_ids, relevant_doc_ids), 4
            )
            heuristic["retrieval_recall"] = round(
                EvaluationMetrics.retrieval_recall(retrieved_doc_ids, relevant_doc_ids), 4
            )
            heuristic["hit_at_5"] = round(
                EvaluationMetrics.hit_at_k(retrieved_doc_ids, relevant_doc_ids, k=5), 4
            )
            heuristic["mrr"] = round(
                EvaluationMetrics.mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids), 4
            )
            heuristic["ndcg"] = round(
                EvaluationMetrics.ndcg(retrieved_doc_ids, relevant_doc_ids, k=5), 4
            )

        result: Dict[str, Any] = {"heuristic": heuristic, "metrics": dict(heuristic)}

        if llm_judge is not None and context.strip() and answer.strip():
            judge_scores = llm_judge.evaluate_all(query, answer, context)
            result["llm_judge"] = judge_scores
            result["metrics"]["llm_answer_relevance"] = judge_scores["answer_relevance"]["score"]
            result["metrics"]["llm_groundedness"] = judge_scores["groundedness"]["score"]
            result["metrics"]["llm_retrieval_relevance"] = judge_scores["retrieval_relevance"]["score"]
            result["metrics"]["llm_aggregate_score"] = judge_scores["aggregate_score"]

        return result

    @staticmethod
    def flatten_metrics(eval_result: Dict[str, Any]) -> Dict[str, float]:
        return dict(eval_result.get("metrics", {}))
