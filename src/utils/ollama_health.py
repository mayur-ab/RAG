import os
from typing import Any, Dict, Optional

import httpx

from config.settings import settings


def _extract_model_context_length(model_info: Dict[str, Any]) -> Optional[int]:
    """Read max context length from Ollama model_info (key varies by architecture)."""
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def get_ollama_status() -> dict:
    """Check whether Ollama is reachable and required models are installed."""
    base = settings.OLLAMA_BASE_URL.rstrip("/")
    env_num_ctx = os.environ.get("OLLAMA_NUM_CTX")
    status: Dict[str, Any] = {
        "reachable": False,
        "base_url": base,
        "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
        "llm_model": settings.OLLAMA_LLM_MODEL,
        "embedding_model_available": False,
        "llm_model_available": False,
        "installed_models": [],
        "num_ctx_app": settings.OLLAMA_NUM_CTX,
        "num_ctx_env": int(env_num_ctx) if env_num_ctx and env_num_ctx.isdigit() else env_num_ctx,
        "model_max_context_length": None,
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

        if status["llm_model_available"]:
            show_resp = httpx.post(
                f"{base}/api/show",
                json={"name": settings.OLLAMA_LLM_MODEL},
                timeout=10.0,
            )
            show_resp.raise_for_status()
            show_data = show_resp.json()
            model_info = show_data.get("model_info") or {}
            status["model_max_context_length"] = _extract_model_context_length(model_info)
    except Exception as exc:
        status["error"] = str(exc)

    effective = settings.OLLAMA_NUM_CTX
    if env_num_ctx and str(env_num_ctx).isdigit():
        effective = min(effective, int(env_num_ctx))
    status["num_ctx_effective_hint"] = effective

    return status
