from abc import ABC, abstractmethod
from typing import List, Optional
from src.metadata.schema import Document, ChunkMetadata


class BaseDocumentLoader(ABC):
    """Abstract Base Class for pluggable document loaders."""

    @abstractmethod
    def load(self, source: str) -> str:
        """Load raw content from source file or URL."""
        pass

    @abstractmethod
    def parse(self, raw_content: str, source: str) -> List[Document]:
        """Parse raw content into standardized Document objects."""
        pass

    @abstractmethod
    def extract_metadata(self, source: str, raw_content: str) -> ChunkMetadata:
        """Extract metadata from source and raw content."""
        pass

    def process(self, source: str, document_id: Optional[str] = None) -> List[Document]:
        """Complete ingestion flow: load -> parse -> extract metadata."""
        raw_content = self.load(source)
        documents = self.parse(raw_content, source)
        metadata = self.extract_metadata(source, raw_content)
        
        if document_id:
            metadata.document_id = document_id

        for doc in documents:
            doc.metadata = metadata.model_copy()
        
        return documents
