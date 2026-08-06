import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.llm_judge import LLMJudge


def test_eval_metrics():
    prec = EvaluationMetrics.retrieval_precision(["doc1", "doc2"], ["doc1", "doc3"])
    rec = EvaluationMetrics.retrieval_recall(["doc1", "doc2"], ["doc1", "doc3"])
    assert prec == 0.5
    assert rec == 0.5

    ctx_rel = EvaluationMetrics.context_relevance("Rishi Kanada atom theory", "Rishi Kanada deduced the atom parmanu.")
    assert ctx_rel > 0.0

    assert EvaluationMetrics.hit_at_k(["doc2", "doc1"], ["doc1"], k=2) == 1.0
    assert EvaluationMetrics.mean_reciprocal_rank(["doc2", "doc1"], ["doc1"]) == 0.5
    assert EvaluationMetrics.ndcg(["doc1", "doc2"], ["doc1"], k=2) == 1.0

    evaluator = RAGEvaluator()
    res = evaluator.evaluate_query_response(
        query="What is Parmanu?",
        retrieved_doc_ids=["doc1"],
        relevant_doc_ids=["doc1"],
        context="Parmanu is the fundamental particle.",
        answer="Parmanu is the fundamental particle.",
        llm_judge=LLMJudge(provider="mock"),
    )
    assert res["heuristic"]["retrieval_precision"] == 1.0
    assert res["heuristic"]["retrieval_recall"] == 1.0
    assert res["heuristic"]["faithfulness"] == 1.0
    assert res["heuristic"]["hit_at_5"] == 1.0
    assert res["metrics"]["llm_aggregate_score"] == 1.0


def test_llm_judge_mock():
    judge = LLMJudge(provider="mock")
    result = judge.evaluate_all(
        question="What is AI?",
        answer="AI is artificial intelligence.",
        context="AI means artificial intelligence.",
    )
    assert result["aggregate_score"] == 1.0
