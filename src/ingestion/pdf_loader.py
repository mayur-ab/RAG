import os
from typing import List
from src.ingestion.base import BaseDocumentLoader
from src.metadata.schema import Document, ChunkMetadata
from src.metadata.extractor import MetadataExtractor


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF documents with pypdf and fallback handling."""

    def load(self, source: str) -> str:
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")
        
        extracted_text = []
        try:
            import pypdf
            reader = pypdf.PdfReader(source)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
        except Exception as e:
            # Fallback to basic binary text filter if pypdf encounters issues
            with open(source, "rb") as f:
                content = f.read()
                text_clean = "".join(chr(b) for b in content if 32 <= b <= 126 or b in (10, 13))
                extracted_text.append(text_clean)

        full_text = "\n\n".join(extracted_text).strip()
        if not full_text:
            raise ValueError(
                f"No text could be extracted from PDF '{source}'. "
                "The file may be scanned or image-based. "
                "Convert it to searchable PDF (OCR) or upload a .txt/.docx version instead."
            )
        return full_text

    def parse(self, raw_content: str, source: str) -> List[Document]:
        metadata = self.extract_metadata(source, raw_content)
        return [Document(content=raw_content.strip(), metadata=metadata)]

    def extract_metadata(self, source: str, raw_content: str) -> ChunkMetadata:
        metadata = MetadataExtractor.extract_from_file(source)
        metadata.tags.append("pdf")
        return metadata
