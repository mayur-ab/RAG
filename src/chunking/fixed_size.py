import uuid
from typing import List
from src.chunking.base import BaseChunker
from src.metadata.schema import Document, ChunkMetadata


class FixedSizeChunker(BaseChunker):
    """Fixed-size character chunker with overlap."""

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []

        for doc in documents:
            text = doc.content
            start = 0
            text_len = len(text)

            if text_len <= self.chunk_size:
                chunked_docs.append(doc)
                continue

            chunk_idx = 0
            while start < text_len:
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:
                    meta: ChunkMetadata = doc.metadata.model_copy()
                    meta.chunk_id = f"{doc.metadata.document_id}_chk_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                    chunked_docs.append(Document(id=meta.chunk_id, content=chunk_text, metadata=meta))
                    chunk_idx += 1

                start += (self.chunk_size - self.chunk_overlap)
                if start >= text_len or self.chunk_size <= self.chunk_overlap:
                    break

        return chunked_docs
