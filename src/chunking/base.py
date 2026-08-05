from abc import ABC, abstractmethod
from typing import List
from src.metadata.schema import Document


class BaseChunker(ABC):
    """Abstract Base Class for Document Chunking strategies."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller semantic chunks with metadata attached."""
        pass
