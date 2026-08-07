"""Hybrid document classification: LLM labels on ingest + periodic clustering."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from config.logging_config import logger
from config.settings import settings
from src.classification.clustering import cluster_documents
from src.classification.llm_classifier import LLMClusterNamer, LLMDocumentClassifier
from src.classification.metadata_store import DocumentMetadataStore
from src.classification.models import DocumentClassification
from src.ingestion.fingerprint import normalize_source_path


class DocumentClassificationService:
    def __init__(
        self,
        metadata_store: DocumentMetadataStore,
        llm_provider,
        embedding_provider,
    ):
        self.store = metadata_store
        self.classifier = LLMDocumentClassifier(llm_provider)
        self.cluster_namer = LLMClusterNamer(llm_provider)
        self.embedder = embedding_provider

    def classify_document(
        self,
        document_id: str,
        source: str,
        title: str,
        text_sample: str,
    ) -> DocumentClassification:
        classification = self.classifier.classify(title, text_sample)
        embed_text = f"{title}\n{classification.summary}\n{' '.join(classification.tags)}"
        try:
            doc_embedding = self.embedder.embed_documents([embed_text[:2000]])[0]
        except Exception as exc:
            logger.warning("Doc embedding failed for %s: %s", source, exc)
            doc_embedding = None

        self.store.upsert_classification(
            document_id=document_id,
            source=source,
            title=title,
            classification=classification,
            doc_embedding=doc_embedding,
        )
        return classification

    def classify_from_chunks(
        self,
        document_id: str,
        source: str,
        title: str,
        chunk_texts: List[str],
    ) -> DocumentClassification:
        sample = "\n\n".join(chunk_texts[:8])[: settings.DOC_CLASSIFY_SAMPLE_CHARS]
        return self.classify_document(document_id, source, title, sample)

    def classify_indexed_documents(
        self,
        indexed_docs: List[Dict[str, Any]],
        *,
        only_missing: bool = True,
        chunk_lookup: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        classified = 0
        skipped = 0
        failed = 0
        results = []

        for doc in indexed_docs:
            source = doc.get("source") or ""
            if not source:
                continue
            existing = self.store.get_by_source(source)
            if only_missing and existing and existing.category != "Uncategorized":
                skipped += 1
                continue

            document_id = doc.get("document_id") or source
            title = doc.get("title") or os.path.basename(source)
            chunks = (chunk_lookup or {}).get(normalize_source_path(source), [])
            if not chunks:
                chunks = (chunk_lookup or {}).get(source, [])

            try:
                self.classify_from_chunks(document_id, source, title, chunks)
                classified += 1
                results.append({"source": source, "status": "classified"})
            except Exception as exc:
                failed += 1
                results.append({"source": source, "status": "failed", "error": str(exc)})
                logger.warning("Classification failed for %s: %s", source, exc)

        return {
            "classified": classified,
            "skipped": skipped,
            "failed": failed,
            "total": len(indexed_docs),
            "results": results[:50],
        }

    def run_clustering(self, k: Optional[int] = None) -> Dict[str, Any]:
        items = self.store.get_embeddings_for_clustering()
        if len(items) < settings.DOC_CLUSTER_MIN_DOCS:
            return {
                "clustered": False,
                "reason": f"Need at least {settings.DOC_CLUSTER_MIN_DOCS} embedded documents",
                "available": len(items),
            }

        groups = cluster_documents(items, k=k)
        collections = []

        for cluster_id, members in sorted(groups.items()):
            titles = [m["title"] or os.path.basename(m["source"]) for m in members]
            try:
                label = self.cluster_namer.name_cluster(titles)
            except Exception as exc:
                logger.warning("Cluster naming failed for cluster %s: %s", cluster_id, exc)
                label = f"Collection {cluster_id + 1}"

            for member in members:
                self.store.update_cluster(member["document_id"], cluster_id, label)

            collections.append(
                {
                    "cluster_id": cluster_id,
                    "label": label,
                    "count": len(members),
                    "sample_titles": titles[:5],
                }
            )

        return {
            "clustered": True,
            "clusters": len(collections),
            "documents": len(items),
            "collections": collections,
        }

    def semantic_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        items = self.store.get_embeddings_for_clustering()
        if not items or not query.strip():
            return []

        query_vec = np.array(self.embedder.embed_documents([query])[0], dtype=float)
        q_norm = np.linalg.norm(query_vec) or 1.0
        query_vec = query_vec / q_norm

        scored = []
        for item in items:
            vec = np.array(item["embedding"], dtype=float)
            v_norm = np.linalg.norm(vec) or 1.0
            sim = float(np.dot(query_vec, vec / v_norm))
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, item in scored[:limit]:
            meta = self.store.get_by_source(item["source"])
            results.append(
                {
                    "source": item["source"],
                    "title": item["title"],
                    "similarity": round(sim, 4),
                    "category": meta.category if meta else "Uncategorized",
                    "subcategory": meta.subcategory if meta else "",
                    "tags": meta.tags if meta else [],
                    "summary": meta.summary if meta else "",
                    "cluster_label": meta.cluster_label if meta else "",
                }
            )
        return results

    def enrich_indexed_documents(
        self,
        indexed_docs: List[Dict[str, Any]],
        *,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        tag: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        meta_by_source = {r.source: r for r in self.store.list_all()}
        enriched = []

        for doc in indexed_docs:
            source = doc.get("source") or ""
            meta = meta_by_source.get(source)
            row = dict(doc)
            if meta:
                row.update(
                    {
                        "category": meta.category,
                        "subcategory": meta.subcategory,
                        "tags": meta.tags,
                        "summary": meta.summary,
                        "cluster_id": meta.cluster_id,
                        "cluster_label": meta.cluster_label,
                        "classified_at": meta.classified_at,
                    }
                )
            else:
                row.update(
                    {
                        "category": "Uncategorized",
                        "subcategory": "",
                        "tags": [],
                        "summary": "",
                        "cluster_id": None,
                        "cluster_label": "",
                        "classified_at": None,
                    }
                )
            enriched.append(row)

        if category:
            enriched = [d for d in enriched if (d.get("category") or "").lower() == category.lower()]
        if subcategory:
            enriched = [d for d in enriched if (d.get("subcategory") or "").lower() == subcategory.lower()]
        if tag:
            tag_l = tag.lower().lstrip("#")
            enriched = [
                d
                for d in enriched
                if any(str(t).lower() == tag_l for t in (d.get("tags") or []))
            ]
        if search:
            q = search.lower()
            enriched = [
                d
                for d in enriched
                if q in (d.get("title") or "").lower()
                or q in (d.get("source") or "").lower()
                or q in (d.get("summary") or "").lower()
                or any(q in str(t).lower() for t in (d.get("tags") or []))
            ]

        return enriched

    def get_browse_tree(self, indexed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        tree = self.store.get_category_tree()
        meta_by_source = {r.source: r for r in self.store.list_all()}
        unclassified = 0
        for doc in indexed_docs:
            meta = meta_by_source.get(doc.get("source") or "")
            if not meta or meta.category == "Uncategorized":
                unclassified += 1
        tree["total_indexed"] = len(indexed_docs)
        tree["unclassified"] = unclassified
        return tree

    def on_document_deleted(self, source: str) -> None:
        self.store.delete_by_source(source)
