from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_rag_service
from src.service import RAGPipelineService

router = APIRouter(tags=["Documents"])


@router.get("/documents/categories")
def get_document_categories(service: RAGPipelineService = Depends(get_rag_service)):
    if not service.doc_classification:
        raise HTTPException(status_code=503, detail="Document classification is disabled")
    indexed = service.vector_store.list_indexed_sources() if hasattr(service.vector_store, "list_indexed_sources") else []
    return service.doc_classification.get_browse_tree(indexed)


@router.get("/documents/indexed")
def list_indexed_documents(
    category: Optional[str] = Query(None),
    subcategory: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    service: RAGPipelineService = Depends(get_rag_service),
):
    indexed = service.list_indexed_documents(
        category=category,
        subcategory=subcategory,
        tag=tag,
        search=search,
    )
    stats = service.vector_store.get_stats()
    return {
        "total_vectors": stats.get("total_vectors", 0),
        "document_count": len(indexed),
        "documents": indexed,
        "filters": {
            "category": category,
            "subcategory": subcategory,
            "tag": tag,
            "search": search,
        },
    }


@router.get("/documents/search")
def semantic_document_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    service: RAGPipelineService = Depends(get_rag_service),
):
    if not service.doc_classification:
        raise HTTPException(status_code=503, detail="Document classification is disabled")
    results = service.doc_classification.semantic_search(q, limit=limit)
    return {"query": q, "count": len(results), "results": results}


@router.post("/documents/classify")
def classify_all_documents(
    only_missing: bool = Query(True, description="Skip already classified documents"),
    service: RAGPipelineService = Depends(get_rag_service),
):
    if not service.doc_classification:
        raise HTTPException(status_code=503, detail="Document classification is disabled")
    indexed = service.vector_store.list_indexed_sources() if hasattr(service.vector_store, "list_indexed_sources") else []
    return service.doc_classification.classify_indexed_documents(
        indexed,
        only_missing=only_missing,
        chunk_lookup=service._chunk_texts_by_source(),
    )


@router.post("/documents/cluster")
def cluster_documents(
    k: Optional[int] = Query(None, ge=2, le=50),
    service: RAGPipelineService = Depends(get_rag_service),
):
    if not service.doc_classification:
        raise HTTPException(status_code=503, detail="Document classification is disabled")
    return service.doc_classification.run_clustering(k=k)
