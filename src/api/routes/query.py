from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import QueryRequest, QueryResponse
from src.api.dependencies import get_rag_service
from src.service import RAGPipelineService

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest, service: RAGPipelineService = Depends(get_rag_service)):
    try:
        res = service.query(
            query=req.query,
            top_k=req.top_k,
            top_m_rerank=req.top_m_rerank,
            user_roles=req.user_roles,
            filter_metadata=req.filter_metadata
        )
        return QueryResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=503, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
