from typing import Optional, Literal, List
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_project_path(value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(PROJECT_ROOT, value.lstrip("./\\")))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # System Information
    PROJECT_NAME: str = "Scalable Enterprise RAG"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Storage Paths
    DOCS_DIR: str = Field(default="Docs", description="Primary directory for document storage")
    UPLOADED_DOCS_DIR: str = Field(default="Uploaded-Docs", description="Directory for user-uploaded documents")
    TEMP_DOC_DIR: str = Field(default="Temp-Doc", description="Temporary directory for single-document mode")
    VECTOR_DB_PATH: str = Field(default="vector_db_data", description="Persistence directory for local vector store")

    # Layer Providers
    VECTOR_STORE_PROVIDER: Literal["memory", "qdrant", "chroma", "faiss"] = "chroma"
    EMBEDDING_PROVIDER: Literal["mock", "ollama", "openai", "sentence_transformers"] = "ollama"
    LLM_PROVIDER: Literal["mock", "ollama", "nvidia_nim", "openai", "gemini"] = "ollama"
    RERANKER_PROVIDER: Literal["identity", "cross_encoder"] = "cross_encoder"

    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_LLM_MODEL: str = "llama3.1:8b"
    OLLAMA_NUM_CTX: int = Field(
        default=32768,
        description="Context window passed to Ollama per request (num_ctx). Also set OLLAMA_NUM_CTX env on the Ollama server.",
    )

    # NVIDIA NIM Settings
    NVIDIA_API_KEY: Optional[str] = None
    NVIDIA_NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    # NVIDIA_NIM_MODEL: str = "meta/llama-3.3-70b-instruct"
    NVIDIA_NIM_MODEL: str = "meta/llama-3.1-8b-instruct"

    # Cloud Provider Keys
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_LLM_MODEL: str = "gemini-1.5-flash"

    # Chunking Configuration
    CHUNKING_STRATEGY: Literal["recursive", "fixed", "semantic", "header"] = "header"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    HYBRID_ALPHA: float = 0.5  # Weight balance: 1.0 = purely vector, 0.0 = purely BM25
    RRF_K: int = 60           # Reciprocal Rank Fusion constant
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    ENABLE_QUERY_REWRITING: bool = True
    MAX_CHAT_HISTORY_TURNS: int = 5

    # Generation limits
    DEFAULT_MAX_OUTPUT_TOKENS: int = 1024
    MAX_OUTPUT_TOKENS_CAP: int = 8192
    DEFAULT_CONTEXT_TOKENS: int = 4000
    MAX_CONTEXT_TOKENS: int = 8000

    # Evaluation
    EVAL_USE_LLM_JUDGE: bool = False
    EVAL_JUDGE_MODEL: Optional[str] = None

    # Input limits
    MAX_QUERY_LENGTH: int = 4000
    MAX_QUERY_WARN_LENGTH: int = 3600

    # Long-term user memory (separate from RAG knowledge base)
    ENABLE_USER_MEMORY: bool = True
    USER_MEMORY_DB_PATH: str = Field(default="user_memory.db", description="SQLite path for user profiles")
    USER_MEMORY_CHROMA_PATH: str = Field(
        default="user_memory_db",
        description="Chroma path for episodic conversation summaries",
    )
    USER_MEMORY_TOP_K: int = 5
    USER_MEMORY_MIN_MESSAGES_ARCHIVE: int = 2
    USER_MEMORY_MAX_PROFILE_ITEMS: int = 20
    USER_MEMORY_MAX_CHARS: int = Field(
        default=4000,
        description="Max characters for user memory block injected into system prompt (~1K tokens)",
    )

    # Conversation hierarchy: user → session → chat
    CHAT_COMPACT_MAX_CHARS: int = 1200
    SESSION_COMPACT_MAX_CHARS: int = 2400
    MAX_ROUTING_TURNS: int = 2

    # Hierarchical long-document generation (outline → sections → merge)
    ENABLE_HIERARCHICAL_GENERATION: bool = True
    LONG_DOC_MIN_PAGES: int = 3
    LONG_DOC_MIN_WORDS: int = 1500
    LONG_DOC_MIN_SECTIONS: int = 4
    LONG_DOC_MAX_SECTIONS: int = 12
    LONG_DOC_DEFAULT_SECTIONS: int = 6
    LONG_DOC_WORDS_PER_PAGE: int = 500
    LONG_DOC_WORDS_PER_SECTION: int = 800
    LONG_DOC_OUTLINE_MAX_TOKENS: int = 1024
    LONG_DOC_OUTLINE_CONTEXT_TOKENS: int = 3000
    LONG_DOC_SECTION_CONTEXT_TOKENS: int = 4000
    LONG_DOC_SECTION_OUTPUT_TOKENS: int = 2048
    LONG_DOC_SECTION_SUMMARY_MAX_CHARS: int = 800

    # Document classification (hybrid: LLM on ingest + periodic clustering)
    ENABLE_DOC_CLASSIFICATION: bool = True
    DOC_CLASSIFICATION_DB_PATH: str = Field(
        default="document_metadata.db",
        description="SQLite path for document categories, tags, summaries",
    )
    DOC_CLASSIFY_SAMPLE_CHARS: int = 3000
    DOC_CATEGORIES: List[str] = Field(
        default_factory=lambda: [
            "HR",
            "Finance",
            "Legal",
            "Technical",
            "Operations",
            "Sales",
            "Marketing",
            "Uncategorized",
        ]
    )
    DOC_CLUSTER_MIN_DOCS: int = 10
    DOC_CLUSTER_MIN_K: int = 5
    DOC_CLUSTER_MAX_K: int = 20

    # Security & Access Control
    ENABLE_RBAC: bool = True
    DEFAULT_USER_ROLES: List[str] = ["user"]
    SECRET_KEY: str = "production-secret-key-change-in-env"

    @field_validator("VECTOR_DB_PATH", "DOCS_DIR", "UPLOADED_DOCS_DIR", "TEMP_DOC_DIR", "USER_MEMORY_DB_PATH", "USER_MEMORY_CHROMA_PATH", "DOC_CLASSIFICATION_DB_PATH")
    @classmethod
    def resolve_relative_paths(cls, value: str) -> str:
        return _resolve_project_path(value)


settings = Settings()
