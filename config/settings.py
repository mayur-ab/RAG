from typing import Optional, Literal, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    HYBRID_ALPHA: float = 0.5  # Weight balance: 1.0 = purely vector, 0.0 = purely BM25
    RRF_K: int = 60           # Reciprocal Rank Fusion constant
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # Security & Access Control
    ENABLE_RBAC: bool = True
    DEFAULT_USER_ROLES: List[str] = ["user"]
    SECRET_KEY: str = "production-secret-key-change-in-env"


settings = Settings()
