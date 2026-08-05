import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.service import RAGPipelineService


def main():
    service = RAGPipelineService()
    temp_doc_path = os.path.join("Temp-Doc", "The Rishi Cognitive Framework_ Ancient Logic for Modern Minds.txt")
    
    if not os.path.exists(temp_doc_path):
        print(f"Error: File not found at {temp_doc_path}")
        sys.exit(1)

    print(f"=== Starting Ingestion of Temp-Doc: {temp_doc_path} ===")
    res = service.ingest_source(source=temp_doc_path, chunking_strategy=settings.CHUNKING_STRATEGY)
    print("Ingestion Result:")
    print(res)
    print("\nVector Store Stats:")
    print(service.vector_store.get_stats())
    print("=== Temp-Doc Ingestion Completed Successfully ===")


if __name__ == "__main__":
    main()
