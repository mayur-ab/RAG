import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from src.api.schemas import QueryRequest, QueryResponse
from src.api.dependencies import get_rag_service, set_runtime_llm_model
from src.service import RAGPipelineService

router = APIRouter(tags=["Query"])


def _history_payload(req: QueryRequest):
    if not req.chat_history:
        return None
    return [{"role": m.role, "content": m.content} for m in req.chat_history]


@router.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest, service: RAGPipelineService = Depends(get_rag_service)):
    try:
        res = service.query(
            query=req.query,
            top_k=req.top_k,
            top_m_rerank=req.top_m_rerank,
            user_roles=req.user_roles,
            filter_metadata=req.filter_metadata,
            chat_history=_history_payload(req),
            use_rag=req.use_rag,
            use_cache=req.use_cache,
            model=req.model,
        )
        if req.model:
            set_runtime_llm_model(req.model.strip())
        return QueryResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.post("/query/stream")
def query_rag_stream(req: QueryRequest, service: RAGPipelineService = Depends(get_rag_service)):
    if req.model:
        set_runtime_llm_model(req.model.strip())

    def event_generator():
        try:
            for event in service.query_stream(
                query=req.query,
                top_k=req.top_k,
                top_m_rerank=req.top_m_rerank,
                user_roles=req.user_roles,
                filter_metadata=req.filter_metadata,
                chat_history=_history_payload(req),
                use_rag=req.use_rag,
                use_cache=req.use_cache,
                model=req.model,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except ValueError as ve:
            yield f"data: {json.dumps({'type': 'error', 'message': str(ve)}, ensure_ascii=False)}\n\n"
        except RuntimeError as re:
            yield f"data: {json.dumps({'type': 'error', 'message': str(re)}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Query execution failed: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
