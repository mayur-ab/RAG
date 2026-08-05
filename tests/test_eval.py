import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.evaluator import RAGEvaluator


def test_eval_metrics():
    prec = EvaluationMetrics.retrieval_precision(["doc1", "doc2"], ["doc1", "doc3"])
    rec = EvaluationMetrics.retrieval_recall(["doc1", "doc2"], ["doc1", "doc3"])
    assert prec == 0.5
    assert rec == 0.5

    ctx_rel = EvaluationMetrics.context_relevance("Rishi Kanada atom theory", "Rishi Kanada deduced the atom parmanu.")
    assert ctx_rel > 0.0

    evaluator = RAGEvaluator()
    res = evaluator.evaluate_query_response(
        query="What is Parmanu?",
        retrieved_doc_ids=["doc1"],
        relevant_doc_ids=["doc1"],
        context="Parmanu is the fundamental particle.",
        answer="Parmanu is the fundamental particle."
    )
    assert res["retrieval_precision"] == 1.0
    assert res["retrieval_recall"] == 1.0
    assert res["faithfulness"] == 1.0
    assert res["hallucination_rate"] == 0.0
