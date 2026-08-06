from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from config.settings import settings
from src.api.schemas import EvaluateRequest, EvaluateResponse
from src.api.dependencies import get_rag_service
from src.service import RAGPipelineService
from src.evaluation.evaluator import RAGEvaluator
from src.evaluation.llm_judge import LLMJudge

router = APIRouter(tags=["Evaluation"])


def _build_llm_judge(use_llm_judge: bool) -> Optional[LLMJudge]:
    if not use_llm_judge:
        return None
    return LLMJudge(model=settings.EVAL_JUDGE_MODEL or settings.OLLAMA_LLM_MODEL)


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_rag(req: EvaluateRequest, service: RAGPipelineService = Depends(get_rag_service)):
    try:
        query_res = service.query(
            query=req.query,
            user_roles=req.allowed_roles,
            use_cache=req.use_cache,
        )
        retrieved_ids = [c["document_id"] for c in query_res.get("citations", [])]
        context = query_res.get("retrieved_context") or ""

        evaluator = RAGEvaluator()
        eval_result = evaluator.evaluate_query_response(
            query=req.query,
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=req.relevant_doc_ids,
            context=context,
            answer=query_res["answer"],
            llm_judge=_build_llm_judge(req.use_llm_judge),
        )

        return EvaluateResponse(
            query=req.query,
            metrics=eval_result["metrics"],
            answer=query_res["answer"],
            heuristic=eval_result.get("heuristic", {}),
            llm_judge=eval_result.get("llm_judge"),
            retrieved_chunks_count=query_res.get("reranked_chunks_count", 0),
            retrieved_context_preview=context[:500],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
