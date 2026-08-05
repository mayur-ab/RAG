from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source: str = Field(..., description="File path (e.g. Temp-Doc/doc.txt, Docs/file.pdf) or Web URL")
    document_id: Optional[str] = Field(None, description="Optional custom document ID")
    chunking_strategy: Optional[str] = Field("recursive", description="fixed, recursive, semantic, or header")
    chunk_size: Optional[int] = Field(500, description="Max characters per chunk")
    chunk_overlap: Optional[int] = Field(100, description="Overlap characters")
    allowed_roles: Optional[List[str]] = Field(["user", "admin"], description="RBAC allowed roles")


class IngestResponse(BaseModel):
    source: str
    document_id: str
    total_chunks: int
    chunking_strategy: str
    elapsed_seconds: float


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or query string")
    top_k: Optional[int] = Field(10, description="Number of vector/BM25 candidates")
    top_m_rerank: Optional[int] = Field(5, description="Number of reranked candidates passed to LLM")
    user_roles: Optional[List[str]] = Field(["user"], description="Roles of user issuing query")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata key-value filters")


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    formatted_citations: str
    model: Optional[str]
    token_usage: Dict[str, int]
    latency: Dict[str, float]
    retrieved_chunks_count: int
    reranked_chunks_count: int
    match_percent: float = Field(..., description="Percentage of answer grounded in retrieved documents")


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    folder: str
    ingested: bool
    ingestion: Optional[IngestResponse] = None


class EvaluateRequest(BaseModel):
    query: str
    relevant_doc_ids: List[str]
    allowed_roles: Optional[List[str]] = ["user"]


class EvaluateResponse(BaseModel):
    query: str
    metrics: Dict[str, float]
    answer: str
