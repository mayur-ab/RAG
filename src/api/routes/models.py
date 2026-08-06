import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from config.settings import settings
from src.api.dependencies import get_rag_service, get_runtime_llm_model, set_runtime_llm_model
from src.service import RAGPipelineService

router = APIRouter(tags=["Models"])


class ModelSelectRequest(BaseModel):
    model: str = Field(..., description="Ollama model name, e.g. llama3.1:8b")


def _current_model_name() -> str:
    runtime = get_runtime_llm_model()
    if runtime:
        return runtime
    try:
        return get_rag_service().get_llm_model_name()
    except Exception:
        return settings.OLLAMA_LLM_MODEL if settings.LLM_PROVIDER == "ollama" else settings.LLM_PROVIDER


@router.get("/models")
def list_models():
    current = _current_model_name()
    if settings.LLM_PROVIDER != "ollama":
        return {
            "provider": settings.LLM_PROVIDER,
            "current_model": current,
            "models": [current],
            "switchable": False,
        }

    base = settings.OLLAMA_BASE_URL.rstrip("/")
    try:
        response = httpx.get(f"{base}/api/tags", timeout=10.0)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not list Ollama models: {exc}")

    return {
        "provider": "ollama",
        "current_model": current,
        "models": sorted(models),
        "switchable": True,
    }


@router.post("/models/select")
def select_model(req: ModelSelectRequest, service: RAGPipelineService = Depends(get_rag_service)):
    if settings.LLM_PROVIDER != "ollama":
        raise HTTPException(status_code=400, detail="Model switching is only supported with Ollama.")
    try:
        service.set_llm_model(req.model)
        set_runtime_llm_model(req.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"current_model": service.get_llm_model_name(), "status": "ok"}
