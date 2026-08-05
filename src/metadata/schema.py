from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid


class ChunkMetadata(BaseModel):
    document_id: str = Field(..., description="Unique document ID")
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk ID")
    source: str = Field(..., description="File path, URL, or origin name")
    title: str = Field(default="Untitled", description="Document title")
    section: Optional[str] = Field(default=None, description="Section or heading title")
    author: Optional[str] = Field(default="Unknown", description="Author of document")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Ingestion creation timestamp")
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Ingestion update timestamp")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    version: str = Field(default="1.0", description="Document version")
    allowed_roles: List[str] = Field(default_factory=lambda: ["user", "admin"], description="RBAC allowed roles for access filtering")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata fields")


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: ChunkMetadata
