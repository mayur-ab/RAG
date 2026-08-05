import os
import re
from typing import List
from src.ingestion.base import BaseDocumentLoader
from src.metadata.schema import Document, ChunkMetadata
from src.metadata.extractor import MetadataExtractor


class HTMLDocumentLoader(BaseDocumentLoader):
    """Loader for HTML documents."""

    def load(self, source: str) -> str:
        if not os.path.exists(source):
            raise FileNotFoundError(f"HTML file not found: {source}")
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def parse(self, raw_content: str, source: str) -> List[Document]:
        # Strip HTML tags
        clean_text = re.sub(r"<[^>]+>", " ", raw_content)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        metadata = self.extract_metadata(source, raw_content)
        return [Document(content=clean_text, metadata=metadata)]

    def extract_metadata(self, source: str, raw_content: str) -> ChunkMetadata:
        metadata = MetadataExtractor.extract_from_file(source)
        title_match = re.search(r"<title>(.*?)</title>", raw_content, re.IGNORECASE)
        if title_match:
            metadata.title = title_match.group(1).strip()
        metadata.tags.append("html")
        return metadata
