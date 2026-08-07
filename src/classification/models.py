from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentClassification(BaseModel):
    category: str = "Uncategorized"
    subcategory: str = ""
    tags: List[str] = Field(default_factory=list)
    summary: str = ""


class DocumentMetadataRecord(BaseModel):
    document_id: str
    source: str
    title: str = ""
    category: str = "Uncategorized"
    subcategory: str = ""
    tags: List[str] = Field(default_factory=list)
    summary: str = ""
    cluster_id: Optional[int] = None
    cluster_label: str = ""
    classified_at: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
