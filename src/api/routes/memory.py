from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_rag_service
from src.api.schemas import (
    ArchiveSessionRequest,
    ArchiveSessionResponse,
    DeleteUserMemoryResponse,
    EndChatRequest,
    EndChatResponse,
    EndSessionRequest,
    EndSessionResponse,
    SessionStartResponse,
    UserProfileResponse,
)
from src.service import RAGPipelineService

router = APIRouter(tags=["Memory"])


def _history_payload(chat_history):
    if not chat_history:
        return []
    return [{"role": m.role, "content": m.content} for m in chat_history]


@router.post("/memory/session/start", response_model=SessionStartResponse)
def start_session(
    user_id: str = Query(...),
    session_id: str | None = Query(default=None),
    service: RAGPipelineService = Depends(get_rag_service),
):
    try:
        result = service.start_user_session(user_id, session_id)
        return SessionStartResponse(
            session_id=result["session_id"],
            user_id=result["user_id"],
            status=result.get("status", "active"),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))


@router.post("/memory/session/chat-end", response_model=EndChatResponse)
def end_chat(req: EndChatRequest, service: RAGPipelineService = Depends(get_rag_service)):
    try:
        result = service.end_user_chat(req.user_id, req.session_id, req.chat_id, req.chat_compact)
        return EndChatResponse(
            merged=bool(result.get("merged")),
            session_id=result.get("session_id") or req.session_id,
            chat_id=req.chat_id,
            reason=result.get("reason"),
            chat_count=int(result.get("chat_count") or 0),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))


@router.post("/memory/session/end", response_model=EndSessionResponse)
def end_session(req: EndSessionRequest, service: RAGPipelineService = Depends(get_rag_service)):
    try:
        result = service.end_user_session(req.user_id, req.session_id, req.chat_compact)
        return EndSessionResponse(
            archived=bool(result.get("archived")),
            session_id=result.get("session_id") or req.session_id,
            summary=result.get("summary"),
            chat_count=int(result.get("chat_count") or 0),
            reason=result.get("reason"),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))


@router.post("/memory/archive", response_model=ArchiveSessionResponse)
def archive_session(req: ArchiveSessionRequest, service: RAGPipelineService = Depends(get_rag_service)):
    """Legacy archive — prefer /memory/session/chat-end and /memory/session/end."""
    try:
        result = service.archive_user_session(req.user_id, _history_payload(req.chat_history))
        return ArchiveSessionResponse(
            archived=bool(result.get("archived")),
            reason=result.get("reason"),
            archive_id=result.get("archive_id"),
            summary=result.get("summary"),
            message_count=int(result.get("message_count") or 0),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory archive failed: {exc}")


@router.get("/memory/profile", response_model=UserProfileResponse)
def get_user_profile(
    user_id: str = Query(..., description="Stable user identifier"),
    service: RAGPipelineService = Depends(get_rag_service),
):
    try:
        profile = service.get_user_profile(user_id)
        return UserProfileResponse(**profile)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Profile lookup failed: {exc}")


@router.delete("/memory/user", response_model=DeleteUserMemoryResponse)
def delete_user_memory(
    user_id: str = Query(..., description="Stable user identifier"),
    service: RAGPipelineService = Depends(get_rag_service),
):
    """Delete all user memory (profile, topics, archives, episodic summaries). RAG docs are untouched."""
    try:
        result = service.delete_user_memory(user_id)
        sqlite = result.get("sqlite") or {}
        return DeleteUserMemoryResponse(
            user_id=result["user_id"],
            deleted=True,
            profiles_deleted=int(sqlite.get("profiles_deleted") or 0),
            topics_deleted=int(sqlite.get("topics_deleted") or 0),
            archives_deleted=int(sqlite.get("archives_deleted") or 0),
            episodic_memories_deleted=int(result.get("episodic_memories_deleted") or 0),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Memory delete failed: {exc}")
