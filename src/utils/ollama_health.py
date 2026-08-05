import httpx
from config.settings import settings


def get_ollama_status() -> dict:
    """Check whether Ollama is reachable and required models are installed."""
    base = settings.OLLAMA_BASE_URL.rstrip("/")
    status = {
        "reachable": False,
        "base_url": base,
        "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
        "llm_model": settings.OLLAMA_LLM_MODEL,
        "embedding_model_available": False,
        "llm_model_available": False,
        "installed_models": [],
    }

    try:
        response = httpx.get(f"{base}/api/tags", timeout=5.0)
        response.raise_for_status()
        status["reachable"] = True
        models = [m.get("name", "") for m in response.json().get("models", [])]
        status["installed_models"] = models
        status["embedding_model_available"] = any(
            settings.OLLAMA_EMBEDDING_MODEL in name for name in models
        )
        status["llm_model_available"] = any(
            settings.OLLAMA_LLM_MODEL in name for name in models
        )
    except Exception as exc:
        status["error"] = str(exc)

    return status
