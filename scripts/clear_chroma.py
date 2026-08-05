"""Clear the Chroma vector store (fresh start). Stop uvicorn first if running."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.service import RAGPipelineService


def main():
    print(f"=== Clearing Chroma at {settings.VECTOR_DB_PATH} ===")
    service = RAGPipelineService()

    if hasattr(service.vector_store, "reset"):
        before = service.vector_store.get_stats().get("total_vectors", 0)
        service.vector_store.reset()
        after = service.vector_store.get_stats().get("total_vectors", 0)
        print(f"Removed {before} vectors. Current count: {after}")
    else:
        print("Vector store does not support reset.")
        sys.exit(1)

    print("=== Chroma cleared successfully ===")
    print("Restart uvicorn before querying or ingesting again.")


if __name__ == "__main__":
    main()
