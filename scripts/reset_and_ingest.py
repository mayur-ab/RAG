import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.settings import settings
from src.service import RAGPipelineService


def main():
    temp_doc_path = os.path.join(
        "Temp-Doc",
        "The Rishi Cognitive Framework_ Ancient Logic for Modern Minds.txt",
    )

    if not os.path.exists(temp_doc_path):
        print(f"Error: File not found at {temp_doc_path}")
        sys.exit(1)

    print("=== Resetting Chroma vector store ===")
    service = RAGPipelineService()

    if hasattr(service.vector_store, "reset"):
        service.vector_store.reset()
        print("Chroma collection cleared.")
    else:
        print("Warning: Vector store does not support reset; skipping.")

    print(f"\n=== Re-ingesting with {settings.EMBEDDING_PROVIDER} embeddings ===")
    print(f"    Chunking: {settings.CHUNKING_STRATEGY}")
    print(f"    LLM: {settings.LLM_PROVIDER}")
    print(f"    Source: {temp_doc_path}\n")

    res = service.ingest_source(
        source=temp_doc_path,
        chunking_strategy=settings.CHUNKING_STRATEGY,
    )

    print("Ingestion Result:")
    print(res)
    print("\nVector Store Stats:")
    print(service.vector_store.get_stats())
    print("\n=== Re-ingestion completed ===")


if __name__ == "__main__":
    main()
