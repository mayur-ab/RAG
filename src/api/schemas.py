from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message text")


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
    status: str = Field(default="ingested", description="ingested, skipped, or replaced")
    skipped: bool = Field(default=False, description="True when source was unchanged and not re-indexed")


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or query string")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Prior conversation turns for follow-up question rewriting",
    )
    top_k: Optional[int] = Field(10, description="Number of vector/BM25 candidates")
    top_m_rerank: Optional[int] = Field(5, description="Number of reranked candidates passed to LLM")
    user_roles: Optional[List[str]] = Field(["user"], description="Roles of user issuing query")
    filter_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata key-value filters")
    use_rag: bool = Field(default=True, description="When false, skip retrieval and answer with the LLM only")
    use_cache: bool = Field(default=True, description="Return cached response for identical queries")
    model: Optional[str] = Field(default=None, description="Optional Ollama model override for this request")
    user_id: Optional[str] = Field(
        default=None,
        description="Stable user identifier for long-term memory personalization",
    )
    session_id: Optional[str] = Field(default=None, description="Active user session identifier")
    chat_id: Optional[str] = Field(default=None, description="Current chat thread within the session")
    chat_compact: Optional[str] = Field(
        default=None,
        description="Rolling compact summary for this chat (ephemeral, not stored server-side)",
    )
    routing_turns: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Last 1-2 turns for follow-up routing only — never sent verbatim to the LLM",
    )
    pinned_sources: Optional[List[str]] = Field(
        default=None,
        description="Source paths pinned from earlier turns in this chat for follow-up retrieval",
    )


class EndChatRequest(BaseModel):
    user_id: str
    session_id: str
    chat_id: str
    chat_compact: str = ""


class EndSessionRequest(BaseModel):
    user_id: str
    session_id: str
    chat_compact: Optional[str] = ""


class SessionStartResponse(BaseModel):
    session_id: str
    user_id: str
    status: str = "active"


class EndChatResponse(BaseModel):
    merged: bool
    session_id: str
    chat_id: str
    reason: Optional[str] = None
    chat_count: int = 0


class EndSessionResponse(BaseModel):
    archived: bool
    session_id: str
    summary: Optional[str] = None
    chat_count: int = 0
    reason: Optional[str] = None


class ArchiveSessionRequest(BaseModel):
    user_id: str = Field(..., description="Stable user identifier")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Conversation to archive")


class ArchiveSessionResponse(BaseModel):
    archived: bool
    reason: Optional[str] = None
    archive_id: Optional[str] = None
    summary: Optional[str] = None
    message_count: int = 0


class UserProfileResponse(BaseModel):
    user_id: str
    display_name: str = ""
    preferences: Dict[str, Any]
    interests: List[str]
    projects: List[str]
    facts: List[str]
    style: Dict[str, Any]
    frequent_topics: List[str] = Field(default_factory=list)
    recent_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    active_session: Optional[Dict[str, Any]] = None


class DeleteUserMemoryResponse(BaseModel):
    user_id: str
    deleted: bool = True
    profiles_deleted: int = 0
    topics_deleted: int = 0
    archives_deleted: int = 0
    episodic_memories_deleted: int = 0


class QueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    citations: List[Dict[str, Any]]
    formatted_citations: str
    model: Optional[str]
    token_usage: Dict[str, int]
    latency: Dict[str, float]
    retrieved_chunks_count: int
    reranked_chunks_count: int
    match_percent: float = Field(..., description="Percentage of answer grounded in retrieved documents")
    mode: str = Field(default="rag", description="rag or direct")
    cached: bool = Field(default=False, description="True when served from response cache")
    grounded: bool = Field(default=True, description="True when answer is grounded in documents")
    not_in_documents: bool = Field(default=False, description="True when answer is not from indexed documents")
    retrieved_context: str = Field(default="", description="Retrieved chunk text passed to the LLM (for eval/debug)")
    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_compact: str = Field(default="", description="Updated rolling compact for the next turn in this chat")
    pinned_sources: List[str] = Field(
        default_factory=list,
        description="Pinned document sources to carry into the next turn",
    )


class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    folder: str
    ingested: bool
    ingestion: Optional[IngestResponse] = None


class EvaluateRequest(BaseModel):
    query: str
    relevant_doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional known relevant document IDs for retrieval ranking metrics",
    )
    allowed_roles: Optional[List[str]] = ["user"]
    use_llm_judge: bool = Field(
        default=False,
        description="Run LLM-as-judge evaluators (relevance, groundedness, retrieval relevance)",
    )
    use_cache: bool = Field(default=False, description="Use cached query responses during evaluation")


class EvaluateResponse(BaseModel):
    query: str
    metrics: Dict[str, float]
    answer: str
    heuristic: Dict[str, float] = Field(default_factory=dict)
    llm_judge: Optional[Dict[str, Any]] = None
    retrieved_chunks_count: int = 0
    retrieved_context_preview: str = ""
