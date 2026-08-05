import re
import uuid
from typing import List
from src.chunking.base import BaseChunker
from src.metadata.schema import Document, ChunkMetadata


class SemanticChunker(BaseChunker):
    """Splits document content into sentence-bounded semantic chunks."""

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []

        for doc in documents:
            sentences = re.split(r"(?<=[.!?])\s+", doc.content)
            current_chunk_sentences = []
            current_length = 0
            chunk_idx = 0

            for sentence in sentences:
                sentence_clean = sentence.strip()
                if not sentence_clean:
                    continue

                if current_length + len(sentence_clean) <= self.chunk_size:
                    current_chunk_sentences.append(sentence_clean)
                    current_length += len(sentence_clean) + 1
                else:
                    if current_chunk_sentences:
                        chunk_text = " ".join(current_chunk_sentences)
                        meta: ChunkMetadata = doc.metadata.model_copy()
                        meta.chunk_id = f"{doc.metadata.document_id}_sem_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                        chunked_docs.append(Document(id=meta.chunk_id, content=chunk_text, metadata=meta))
                        chunk_idx += 1

                    current_chunk_sentences = [sentence_clean]
                    current_length = len(sentence_clean)

            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                meta: ChunkMetadata = doc.metadata.model_copy()
                meta.chunk_id = f"{doc.metadata.document_id}_sem_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                chunked_docs.append(Document(id=meta.chunk_id, content=chunk_text, metadata=meta))

        return chunked_docs
