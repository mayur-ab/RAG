import re
import httpx
from typing import List
from src.ingestion.base import BaseDocumentLoader
from src.metadata.schema import Document, ChunkMetadata
from src.metadata.extractor import MetadataExtractor


class WebPageLoader(BaseDocumentLoader):
    """Loader for remote web pages via HTTP."""

    def load(self, source: str) -> str:
        response = httpx.get(source, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        return response.text

    def parse(self, raw_content: str, source: str) -> List[Document]:
        clean_text = re.sub(r"<script.*?>.*?</script>", "", raw_content, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r"<style.*?>.*?</style>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
        clean_text = re.sub(r"<[^>]+>", " ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        
        metadata = self.extract_metadata(source, raw_content)
        return [Document(content=clean_text, metadata=metadata)]

    def extract_metadata(self, source: str, raw_content: str) -> ChunkMetadata:
        metadata = MetadataExtractor.extract_from_file(source)
        title_match = re.search(r"<title>(.*?)</title>", raw_content, re.IGNORECASE)
        if title_match:
            metadata.title = title_match.group(1).strip()
        metadata.source = source
        metadata.tags.extend(["web", "http"])
        return metadata
