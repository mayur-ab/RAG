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
        "context": {
            "ollama_num_ctx": settings.OLLAMA_NUM_CTX,
            "default_context_tokens": settings.DEFAULT_CONTEXT_TOKENS,
            "max_context_tokens": settings.MAX_CONTEXT_TOKENS,
            "default_max_output_tokens": settings.DEFAULT_MAX_OUTPUT_TOKENS,
            "max_output_tokens_cap": settings.MAX_OUTPUT_TOKENS_CAP,
            "chat_compact_max_chars": settings.CHAT_COMPACT_MAX_CHARS,
            "session_compact_max_chars": settings.SESSION_COMPACT_MAX_CHARS,
            "max_routing_turns": settings.MAX_ROUTING_TURNS,
            "chunk_size_chars": settings.CHUNK_SIZE,
            "chunk_overlap_chars": settings.CHUNK_OVERLAP,
        },
        "retrieval": {
            "top_k_retrieval": settings.TOP_K_RETRIEVAL,
            "top_k_rerank": settings.TOP_K_RERANK,
            "hybrid_alpha": settings.HYBRID_ALPHA,
            "enable_query_rewriting": settings.ENABLE_QUERY_REWRITING,
            "user_memory_top_k": settings.USER_MEMORY_TOP_K,
            "user_memory_max_chars": settings.USER_MEMORY_MAX_CHARS,
        },
        "long_document": {
            "enabled": settings.ENABLE_HIERARCHICAL_GENERATION,
            "min_pages": settings.LONG_DOC_MIN_PAGES,
            "min_words": settings.LONG_DOC_MIN_WORDS,
            "max_sections": settings.LONG_DOC_MAX_SECTIONS,
            "section_output_tokens": settings.LONG_DOC_SECTION_OUTPUT_TOKENS,
        },
        "document_classification": {
            "enabled": settings.ENABLE_DOC_CLASSIFICATION,
            "categories": settings.DOC_CATEGORIES,
            "cluster_min_docs": settings.DOC_CLUSTER_MIN_DOCS,
        },
        "user_memory": {
            "enabled": settings.ENABLE_USER_MEMORY,
            "profile_store": "sqlite",
            "episodic_store": "chroma",
        },
    }


@router.get("/metrics")
def get_metrics():
    return metrics_collector.get_metrics_summary()
