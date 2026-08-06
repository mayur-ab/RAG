import threading
from src.service import RAGPipelineService

_service_lock = threading.Lock()
_rag_service_instance: RAGPipelineService | None = None
_runtime_llm_model: str | None = None


def get_runtime_llm_model() -> str | None:
    return _runtime_llm_model


def set_runtime_llm_model(model: str) -> None:
    global _runtime_llm_model
    _runtime_llm_model = model


def get_rag_service() -> RAGPipelineService:
    global _rag_service_instance
    if _rag_service_instance is not None:
        return _rag_service_instance

    with _service_lock:
        if _rag_service_instance is None:
            _rag_service_instance = RAGPipelineService()
            if _runtime_llm_model and hasattr(_rag_service_instance.llm_provider, "set_model"):
                _rag_service_instance.llm_provider.set_model(_runtime_llm_model)
        return _rag_service_instance


def reset_rag_service() -> None:
    """Reset singleton — useful after reload or failed Chroma init."""
    global _rag_service_instance
    with _service_lock:
        _rag_service_instance = None
