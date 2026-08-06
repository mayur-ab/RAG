from fastapi import APIRouter, Depends
from src.monitoring.metrics import metrics_collector
from src.api.dependencies import get_rag_service
from src.service import RAGPipelineService
from src.utils.ollama_health import get_ollama_status
from config.settings import settings

router = APIRouter(tags=["Monitoring"])


@router.get("/health")
def health_check(service: RAGPipelineService = Depends(get_rag_service)):
    vector_stats = service.vector_store.get_stats()
    ollama_status = get_ollama_status() if settings.EMBEDDING_PROVIDER == "ollama" or settings.LLM_PROVIDER == "ollama" else None

    overall_status = "healthy"
    if ollama_status and not ollama_status.get("reachable"):
        overall_status = "degraded"

    return {
        "status": overall_status,
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "providers": {
            "vector_store": settings.VECTOR_STORE_PROVIDER,
            "embeddings": settings.EMBEDDING_PROVIDER,
            "llm": settings.LLM_PROVIDER,
            "llm_model": service.get_llm_model_name(),
            "reranker": settings.RERANKER_PROVIDER,
            "chunking": settings.CHUNKING_STRATEGY,
        },
        "vector_store_stats": vector_stats,
        "bm25_chunks_indexed": len(service._all_chunk_documents),
        "ollama": ollama_status,
        "nvidia_api_key_configured": bool(settings.NVIDIA_API_KEY),
        "limits": {
            "max_query_chars": settings.MAX_QUERY_LENGTH,
            "max_query_warn_chars": settings.MAX_QUERY_WARN_LENGTH,
        },
    }


@router.get("/metrics")
def get_metrics():
    return metrics_collector.get_metrics_summary()
