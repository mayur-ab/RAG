"""Batch-classify all indexed documents (run once for existing corpus)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.dependencies import get_rag_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify all indexed documents")
    parser.add_argument("--force", action="store_true", help="Re-classify even if already labeled")
    parser.add_argument("--cluster", action="store_true", help="Run clustering after classification")
    args = parser.parse_args()

    service = get_rag_service()
    if not service.doc_classification:
        print("Document classification is disabled (ENABLE_DOC_CLASSIFICATION=false)")
        sys.exit(1)

    indexed = service.vector_store.list_indexed_sources()
    print(f"Indexed documents: {len(indexed)}")

    result = service.doc_classification.classify_indexed_documents(
        indexed,
        only_missing=not args.force,
        chunk_lookup=service._chunk_texts_by_source(),
    )
    print(
        f"Classified: {result['classified']}, "
        f"skipped: {result['skipped']}, failed: {result['failed']}"
    )

    if args.cluster:
        cluster_result = service.doc_classification.run_clustering()
        if cluster_result.get("clustered"):
            print(f"Clusters: {cluster_result['clusters']}")
            for coll in cluster_result.get("collections", [])[:10]:
                print(f"  - {coll['label']} ({coll['count']} docs)")
        else:
            print(cluster_result.get("reason", "Clustering skipped"))


if __name__ == "__main__":
    main()
