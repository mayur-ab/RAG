import os
from typing import List
from src.ingestion.base import BaseDocumentLoader
from src.metadata.schema import Document, ChunkMetadata
from src.metadata.extractor import MetadataExtractor


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for Plain Text and Markdown documents."""

    def load(self, source: str) -> str:
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def parse(self, raw_content: str, source: str) -> List[Document]:
        metadata = self.extract_metadata(source, raw_content)
        return [Document(content=raw_content.strip(), metadata=metadata)]

    def extract_metadata(self, source: str, raw_content: str) -> ChunkMetadata:
        metadata = MetadataExtractor.extract_from_file(source)
        # Check if first line contains title header
        lines = [line.strip() for line in raw_content.split("\n") if line.strip()]
        if lines and lines[0].startswith("#"):
            metadata.title = lines[0].lstrip("#").strip()
        elif lines and len(lines[0]) < 80:
            metadata.title = lines[0]
        return metadata
