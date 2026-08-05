import os
import subprocess
import tempfile
from pathlib import Path
from typing import List
from src.ingestion.base import BaseDocumentLoader
from src.metadata.schema import Document, ChunkMetadata
from src.metadata.extractor import MetadataExtractor


class DocDocumentLoader(BaseDocumentLoader):
    """Loader for legacy Microsoft Word (.doc) documents."""

    def load(self, source: str) -> str:
        if not os.path.exists(source):
            raise FileNotFoundError(f"DOC file not found: {source}")

        abs_path = os.path.abspath(source)
        errors: List[str] = []

        # Strategy 1: Microsoft Word via COM (Windows + Word installed)
        try:
            text = self._load_via_word_com(abs_path)
            if text.strip():
                return text
            errors.append("Word COM returned empty text")
        except Exception as exc:
            errors.append(f"Word COM: {exc}")

        # Strategy 2: LibreOffice headless conversion
        try:
            text = self._load_via_libreoffice(abs_path)
            if text.strip():
                return text
            errors.append("LibreOffice returned empty text")
        except Exception as exc:
            errors.append(f"LibreOffice: {exc}")

        raise ValueError(
            f"Could not read .doc file '{source}'. "
            "Install Microsoft Word or LibreOffice, or convert to .docx/.txt. "
            f"Details: {'; '.join(errors)}"
        )

    def _load_via_word_com(self, abs_path: str) -> str:
        import win32com.client  # type: ignore

        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        doc = None
        try:
            doc = word.Documents.Open(abs_path, ReadOnly=True)
            return doc.Content.Text or ""
        finally:
            if doc is not None:
                doc.Close(False)
            word.Quit()

    def _load_via_libreoffice(self, abs_path: str) -> str:
        soffice = self._find_soffice()
        if not soffice:
            raise RuntimeError("LibreOffice (soffice) not found on PATH")

        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", tmpdir, abs_path],
                check=True,
                capture_output=True,
                timeout=120,
            )
            out_file = Path(tmpdir) / (Path(abs_path).stem + ".txt")
            if not out_file.exists():
                candidates = list(Path(tmpdir).glob("*.txt"))
                if not candidates:
                    raise RuntimeError("LibreOffice produced no output file")
                out_file = candidates[0]
            return out_file.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _find_soffice() -> str | None:
        import shutil

        found = shutil.which("soffice")
        if found:
            return found
        for candidate in (
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ):
            if os.path.exists(candidate):
                return candidate
        return None

    def parse(self, raw_content: str, source: str) -> List[Document]:
        metadata = self.extract_metadata(source, raw_content)
        return [Document(content=raw_content.strip(), metadata=metadata)]

    def extract_metadata(self, source: str, raw_content: str) -> ChunkMetadata:
        metadata = MetadataExtractor.extract_from_file(source)
        metadata.tags.append("doc")
        return metadata
