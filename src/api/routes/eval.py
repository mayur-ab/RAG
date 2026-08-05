from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import EvaluateRequest, EvaluateResponse
from src.api.dependencies import get_rag_service
from src.service import RAGPipelineService
from src.evaluation.evaluator import RAGEvaluator

router = APIRouter(tags=["Evaluation"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_rag(req: EvaluateRequest, service: RAGPipelineService = Depends(get_rag_service)):
    try:
        query_res = service.query(query=req.query, user_roles=req.allowed_roles)
        retrieved_ids = [c["document_id"] for c in query_res["citations"]]

        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_query_response(
            query=req.query,
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=req.relevant_doc_ids,
            context=query_res["formatted_citations"],
            answer=query_res["answer"]
        )

        return EvaluateResponse(
            query=req.query,
            metrics=metrics,
            answer=query_res["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
