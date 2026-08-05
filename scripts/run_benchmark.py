import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reconfigure stdout for Windows unicode support
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.settings import settings
from src.service import RAGPipelineService
from src.evaluation.evaluator import RAGEvaluator


def main():
    service = RAGPipelineService()
    temp_doc_path = os.path.join("Temp-Doc", "The Rishi Cognitive Framework_ Ancient Logic for Modern Minds.txt")

    stats = service.vector_store.get_stats()
    if stats.get("total_vectors", 0) == 0 and os.path.exists(temp_doc_path):
        print(f"No vectors found — ingesting with {settings.CHUNKING_STRATEGY} chunking...")
        service.ingest_source(source=temp_doc_path, chunking_strategy=settings.CHUNKING_STRATEGY)
    else:
        print(f"Using existing index ({stats.get('total_vectors', 0)} vectors).")

    benchmark_queries = [
        {
            "query": "What are the four mental muscles in the Rishi Cognitive Framework?",
            "relevant_doc_ids": ["The_Rishi_Cognitive_Framework__Ancient_Logic_for_Modern_Minds.txt"]
        },
        {
            "query": "How did Rishi Kanada deduce atomic theory?",
            "relevant_doc_ids": ["The_Rishi_Cognitive_Framework__Ancient_Logic_for_Modern_Minds.txt"]
        },
        {
            "query": "What is Shabda Brahma and Chladni figures?",
            "relevant_doc_ids": ["The_Rishi_Cognitive_Framework__Ancient_Logic_for_Modern_Minds.txt"]
        }
    ]

    evaluator = RAGEvaluator()
    print("=== Running RAG Evaluation Benchmark Suite ===")
    for idx, test in enumerate(benchmark_queries, 1):
        q = test["query"]
        res = service.query(query=q)
        retrieved_ids = [c["document_id"] for c in res["citations"]]

        metrics = evaluator.evaluate_query_response(
            query=q,
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=test["relevant_doc_ids"],
            context=res["formatted_citations"],
            answer=res["answer"]
        )

        print(f"\n--- Test Question #{idx}: '{q}' ---")
        print(f"Answer:\n{res['answer']}\n")
        print(f"Metrics: {metrics}")

    print("\n=== Benchmark Completed ===")


if __name__ == "__main__":
    main()
