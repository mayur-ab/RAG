import re
import uuid
from typing import List
from src.chunking.base import BaseChunker
from src.metadata.schema import Document, ChunkMetadata


class HeaderAwareChunker(BaseChunker):
    """Splits documents based on headers (Markdown headers, Part/Chapter lines) preserving hierarchy."""

    HEADER_PATTERN = re.compile(r"^(#{1,6}\s+.+|Part\s+\d+:?.*|Case Study\s+\d+:?.*)$", re.IGNORECASE | re.MULTILINE)

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []

        for doc in documents:
            lines = doc.content.split("\n")
            current_section = doc.metadata.title or "General"
            current_lines = []
            chunk_idx = 0

            for line in lines:
                match = self.HEADER_PATTERN.match(line.strip())
                if match:
                    # Flush previous section chunk
                    if current_lines:
                        text_block = "\n".join(current_lines).strip()
                        if text_block:
                            meta: ChunkMetadata = doc.metadata.model_copy()
                            meta.chunk_id = f"{doc.metadata.document_id}_hdr_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                            meta.section = current_section
                            chunked_docs.append(Document(id=meta.chunk_id, content=text_block, metadata=meta))
                            chunk_idx += 1
                        current_lines = []
                    current_section = match.group(0).lstrip("#").strip()

                current_lines.append(line)

            if current_lines:
                text_block = "\n".join(current_lines).strip()
                if text_block:
                    meta: ChunkMetadata = doc.metadata.model_copy()
                    meta.chunk_id = f"{doc.metadata.document_id}_hdr_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                    meta.section = current_section
                    chunked_docs.append(Document(id=meta.chunk_id, content=text_block, metadata=meta))

        return chunked_docs
