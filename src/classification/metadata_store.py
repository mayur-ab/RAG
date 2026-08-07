import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.classification.models import DocumentClassification, DocumentMetadataRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentMetadataStore:
    """SQLite store for document classification metadata (separate from Chroma embeddings)."""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS document_metadata (
                        document_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL UNIQUE,
                        title TEXT NOT NULL DEFAULT '',
                        category TEXT NOT NULL DEFAULT 'Uncategorized',
                        subcategory TEXT NOT NULL DEFAULT '',
                        tags TEXT NOT NULL DEFAULT '[]',
                        summary TEXT NOT NULL DEFAULT '',
                        cluster_id INTEGER,
                        cluster_label TEXT NOT NULL DEFAULT '',
                        doc_embedding TEXT,
                        classified_at TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_doc_meta_category
                        ON document_metadata(category);
                    CREATE INDEX IF NOT EXISTS idx_doc_meta_subcategory
                        ON document_metadata(category, subcategory);
                    CREATE INDEX IF NOT EXISTS idx_doc_meta_cluster
                        ON document_metadata(cluster_id);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentMetadataRecord:
        tags = json.loads(row["tags"] or "[]")
        return DocumentMetadataRecord(
            document_id=row["document_id"],
            source=row["source"],
            title=row["title"] or "",
            category=row["category"] or "Uncategorized",
            subcategory=row["subcategory"] or "",
            tags=tags if isinstance(tags, list) else [],
            summary=row["summary"] or "",
            cluster_id=row["cluster_id"],
            cluster_label=row["cluster_label"] or "",
            classified_at=row["classified_at"],
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )

    def get_by_source(self, source: str) -> Optional[DocumentMetadataRecord]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM document_metadata WHERE source = ?",
                    (source,),
                ).fetchone()
                return self._row_to_record(row) if row else None
            finally:
                conn.close()

    def upsert_classification(
        self,
        document_id: str,
        source: str,
        title: str,
        classification: DocumentClassification,
        doc_embedding: Optional[List[float]] = None,
    ) -> DocumentMetadataRecord:
        now = _utc_now()
        tags_json = json.dumps(classification.tags or [])
        embedding_json = json.dumps(doc_embedding) if doc_embedding else None

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT document_id FROM document_metadata WHERE source = ?",
                    (source,),
                ).fetchone()
                if existing:
                    conn.execute(
                        """
                        UPDATE document_metadata SET
                            document_id = ?,
                            title = ?,
                            category = ?,
                            subcategory = ?,
                            tags = ?,
                            summary = ?,
                            doc_embedding = COALESCE(?, doc_embedding),
                            classified_at = ?,
                            updated_at = ?
                        WHERE source = ?
                        """,
                        (
                            document_id,
                            title,
                            classification.category or "Uncategorized",
                            classification.subcategory or "",
                            tags_json,
                            classification.summary or "",
                            embedding_json,
                            now,
                            now,
                            source,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO document_metadata (
                            document_id, source, title, category, subcategory,
                            tags, summary, doc_embedding, classified_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            source,
                            title,
                            classification.category or "Uncategorized",
                            classification.subcategory or "",
                            tags_json,
                            classification.summary or "",
                            embedding_json,
                            now,
                            now,
                            now,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()

        record = self.get_by_source(source)
        assert record is not None
        return record

    def update_cluster(
        self,
        document_id: str,
        cluster_id: int,
        cluster_label: str,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE document_metadata
                    SET cluster_id = ?, cluster_label = ?, updated_at = ?
                    WHERE document_id = ?
                    """,
                    (cluster_id, cluster_label, _utc_now(), document_id),
                )
                conn.commit()
            finally:
                conn.close()

    def list_all(self) -> List[DocumentMetadataRecord]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM document_metadata ORDER BY category, subcategory, title"
                ).fetchall()
                return [self._row_to_record(row) for row in rows]
            finally:
                conn.close()

    def get_category_tree(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT category, subcategory, COUNT(*) AS cnt
                    FROM document_metadata
                    GROUP BY category, subcategory
                    ORDER BY category, subcategory
                    """
                ).fetchall()
                tag_rows = conn.execute(
                    "SELECT tags FROM document_metadata WHERE tags != '[]'"
                ).fetchall()
            finally:
                conn.close()

        tree: Dict[str, Dict[str, int]] = {}
        category_totals: Dict[str, int] = {}
        for row in rows:
            cat = row["category"] or "Uncategorized"
            sub = row["subcategory"] or "General"
            cnt = row["cnt"]
            tree.setdefault(cat, {})
            tree[cat][sub] = tree[cat].get(sub, 0) + cnt
            category_totals[cat] = category_totals.get(cat, 0) + cnt

        tag_counts: Dict[str, int] = {}
        for row in tag_rows:
            try:
                tags = json.loads(row["tags"] or "[]")
            except json.JSONDecodeError:
                continue
            for tag in tags:
                key = str(tag).strip().lower()
                if key:
                    tag_counts[key] = tag_counts.get(key, 0) + 1

        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))[:50]

        return {
            "categories": [
                {
                    "name": cat,
                    "count": category_totals[cat],
                    "subcategories": [
                        {"name": sub, "count": cnt}
                        for sub, cnt in sorted(subs.items(), key=lambda x: x[0].lower())
                    ],
                }
                for cat, subs in sorted(tree.items(), key=lambda x: x[0].lower())
            ],
            "tags": [{"name": t, "count": c} for t, c in sorted_tags],
            "total_classified": sum(category_totals.values()),
        }

    def get_embeddings_for_clustering(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT document_id, source, title, doc_embedding
                    FROM document_metadata
                    WHERE doc_embedding IS NOT NULL AND doc_embedding != ''
                    """
                ).fetchall()
            finally:
                conn.close()

        items = []
        for row in rows:
            try:
                emb = json.loads(row["doc_embedding"])
            except (json.JSONDecodeError, TypeError):
                continue
            if emb:
                items.append(
                    {
                        "document_id": row["document_id"],
                        "source": row["source"],
                        "title": row["title"] or "",
                        "embedding": emb,
                    }
                )
        return items

    def filter_records(
        self,
        *,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        tag: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[DocumentMetadataRecord]:
        records = self.list_all()
        if category:
            records = [r for r in records if r.category.lower() == category.lower()]
        if subcategory:
            records = [r for r in records if r.subcategory.lower() == subcategory.lower()]
        if tag:
            tag_l = tag.lower().lstrip("#")
            records = [r for r in records if any(t.lower() == tag_l for t in r.tags)]
        if search:
            q = search.lower()
            records = [
                r
                for r in records
                if q in (r.title or "").lower()
                or q in r.source.lower()
                or q in r.summary.lower()
                or any(q in t.lower() for t in r.tags)
            ]
        return records

    def delete_by_source(self, source: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("DELETE FROM document_metadata WHERE source = ?", (source,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()
