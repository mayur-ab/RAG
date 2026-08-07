"""Verify document classification + clustering state."""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.vector_store.chroma_store import ChromaVectorStore


def main() -> None:
    db = settings.DOC_CLASSIFICATION_DB_PATH
    print("=== SQLite:", db, "exists:", os.path.exists(db))

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) c FROM document_metadata").fetchone()["c"]
    print(f"Total metadata records: {total}")

    print("\n--- Categories ---")
    for row in conn.execute(
        "SELECT category, COUNT(*) cnt FROM document_metadata GROUP BY category ORDER BY cnt DESC"
    ):
        print(f"  {row['category']}: {row['cnt']}")

    clustered = conn.execute(
        "SELECT COUNT(*) c FROM document_metadata WHERE cluster_id IS NOT NULL"
    ).fetchone()["c"]
    print(f"\nClustered: {clustered}/{total}")

    print("\n--- Cluster labels ---")
    for row in conn.execute(
        """
        SELECT cluster_label, COUNT(*) cnt FROM document_metadata
        WHERE cluster_label != '' GROUP BY cluster_label ORDER BY cnt DESC
        """
    ):
        print(f"  {row['cluster_label']}: {row['cnt']}")

    with_tags = conn.execute(
        "SELECT COUNT(*) c FROM document_metadata WHERE tags != '[]'"
    ).fetchone()["c"]
    with_summary = conn.execute(
        "SELECT COUNT(*) c FROM document_metadata WHERE summary != ''"
    ).fetchone()["c"]
    with_emb = conn.execute(
        "SELECT COUNT(*) c FROM document_metadata WHERE doc_embedding IS NOT NULL AND doc_embedding != ''"
    ).fetchone()["c"]
    print(f"\nWith tags: {with_tags}, summary: {with_summary}, embedding: {with_emb}")

    uncategorized = conn.execute(
        "SELECT COUNT(*) c FROM document_metadata WHERE category = 'Uncategorized'"
    ).fetchone()["c"]
    print(f"Uncategorized in SQLite: {uncategorized}")

    print("\n--- Sample classified doc ---")
    row = conn.execute(
        """
        SELECT title, category, subcategory, tags, summary, cluster_label, source
        FROM document_metadata WHERE category != 'Uncategorized' LIMIT 1
        """
    ).fetchone()
    if row:
        print(json.dumps(dict(row), indent=2))

    conn.close()

    print("\n=== ChromaDB ===")
    store = ChromaVectorStore(settings.VECTOR_DB_PATH)
    stats = store.get_stats()
    indexed = store.list_indexed_sources()
    print(f"Vectors: {stats.get('total_vectors')}")
    print(f"Indexed sources: {len(indexed)}")

    if store.collection.count() > 0:
        sample = store.collection.get(limit=1, include=["metadatas"])
        meta = sample["metadatas"][0] if sample.get("metadatas") else {}
        print("Chunk metadata keys:", sorted(meta.keys()))
        print("Category stored in Chroma?", "category" in meta)

    # Cross-check: indexed in Chroma but missing from SQLite
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    meta_sources = {
        r["source"] for r in conn.execute("SELECT source FROM document_metadata").fetchall()
    }
    conn.close()
    chroma_sources = {d["source"] for d in indexed}
    missing_meta = chroma_sources - meta_sources
    extra_meta = meta_sources - chroma_sources
    print(f"\nIn Chroma but NOT in SQLite metadata: {len(missing_meta)}")
    if missing_meta:
        for s in list(missing_meta)[:5]:
            print(f"  - {s}")
    print(f"In SQLite but NOT in Chroma: {len(extra_meta)}")


if __name__ == "__main__":
    main()
