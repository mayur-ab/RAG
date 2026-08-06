import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.settings import settings
from src.service import RAGPipelineService
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.llm_judge import LLMJudge


def load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("questions", payload if isinstance(payload, list) else [])


def aggregate_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = set()
    for row in rows:
        keys.update(row.get("metrics", {}).keys())
    summary = {}
    for key in sorted(keys):
        values = [row["metrics"][key] for row in rows if key in row.get("metrics", {})]
        if values:
            summary[key] = round(sum(values) / len(values), 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch RAG evaluation without reference answers.")
    parser.add_argument(
        "--questions",
        default="eval/questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        default="eval/reports",
        help="Directory for JSON report output",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        default=settings.EVAL_USE_LLM_JUDGE,
        help="Enable LLM-as-judge evaluators (slower; uses Ollama)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of questions (0 = all)",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise SystemExit(f"Questions file not found: {questions_path}")

    items = load_questions(questions_path)
    if args.limit > 0:
        items = items[: args.limit]

    service = RAGPipelineService()
    evaluator = RAGEvaluator()
    judge = LLMJudge(model=settings.EVAL_JUDGE_MODEL or settings.OLLAMA_LLM_MODEL) if args.use_llm_judge else None

    print(f"=== RAG Eval Suite ({len(items)} questions) ===")
    print(f"Index vectors: {service.vector_store.get_stats().get('total_vectors', 0)}")
    print(f"LLM judge: {'enabled' if judge else 'disabled'}")

    rows = []
    for idx, item in enumerate(items, 1):
        query = item["query"]
        relevant_doc_ids = item.get("relevant_doc_ids") or None
        if relevant_doc_ids == []:
            relevant_doc_ids = None

        print(f"\n[{idx}/{len(items)}] {query}")
        res = service.query(query=query, use_cache=False)
        retrieved_ids = [c["document_id"] for c in res.get("citations", [])]
        context = res.get("retrieved_context") or ""

        eval_result = evaluator.evaluate_query_response(
            query=query,
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=relevant_doc_ids,
            context=context,
            answer=res["answer"],
            llm_judge=judge,
        )

        row = {
            "query": query,
            "answer_preview": res["answer"][:240],
            "metrics": eval_result["metrics"],
            "heuristic": eval_result.get("heuristic", {}),
            "llm_judge": eval_result.get("llm_judge"),
            "retrieved_chunks_count": res.get("reranked_chunks_count", 0),
            "match_percent": res.get("match_percent"),
        }
        rows.append(row)
        print(f"  metrics: {eval_result['metrics']}")

    summary = aggregate_metrics(rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(rows),
        "llm_judge_enabled": bool(judge),
        "summary": summary,
        "results": rows,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"eval_report_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
