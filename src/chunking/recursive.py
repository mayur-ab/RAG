import uuid
from typing import List
from src.chunking.base import BaseChunker
from src.metadata.schema import Document, ChunkMetadata
from src.metadata.extractor import MetadataExtractor


class RecursiveCharacterChunker(BaseChunker):
    """Recursive character chunker using hierarchical separators."""

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def chunk(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []

        for doc in documents:
            sub_texts = self._split_text(doc.content, self.SEPARATORS)
            chunk_idx = 0
            for text_piece in sub_texts:
                clean_piece = text_piece.strip()
                if not clean_piece:
                    continue

                meta: ChunkMetadata = doc.metadata.model_copy()
                meta.chunk_id = f"{doc.metadata.document_id}_rec_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                inferred_sec = MetadataExtractor.infer_section(clean_piece)
                if inferred_sec:
                    meta.section = inferred_sec

                chunked_docs.append(Document(id=meta.chunk_id, content=clean_piece, metadata=meta))
                chunk_idx += 1

        return chunked_docs

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break

        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        current_chunk = []
        current_length = 0

        for s in splits:
            item = s if separator == "" else s + separator
            item_len = len(item)

            if current_length + item_len <= self.chunk_size:
                current_chunk.append(item)
                current_length += item_len
            else:
                if current_chunk:
                    merged = "".join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)
                
                # Reset chunk
                if item_len > self.chunk_size and len(separators) > 1:
                    # Recursively split large piece with next separator
                    sub_splits = self._split_text(item, separators[separators.index(separator)+1:])
                    final_chunks.extend(sub_splits)
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [item]
                    current_length = item_len

        if current_chunk:
            merged = "".join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks
