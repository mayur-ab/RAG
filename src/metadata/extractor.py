import os
import re
from datetime import datetime
from typing import Dict, Any, Optional
from src.metadata.schema import ChunkMetadata


class MetadataExtractor:
    """Extracts rich metadata from document content and file attributes."""

    @staticmethod
    def extract_from_file(file_path: str, document_id: Optional[str] = None, allowed_roles: Optional[list[str]] = None) -> ChunkMetadata:
        doc_id = document_id or os.path.basename(file_path).replace(" ", "_")
        source_name = file_path
        filename = os.path.basename(file_path)
        
        # Derive title from filename
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()

        created_at = datetime.utcnow().isoformat()
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            created_at = datetime.utcfromtimestamp(stat.st_ctime).isoformat()
            updated_at = datetime.utcfromtimestamp(stat.st_mtime).isoformat()
        else:
            updated_at = created_at

        roles = allowed_roles or ["user", "admin"]

        return ChunkMetadata(
            document_id=doc_id,
            source=source_name,
            title=title,
            section=None,
            author="Unknown",
            created_at=created_at,
            updated_at=updated_at,
            tags=[os.path.splitext(filename)[1].lstrip(".").lower()],
            version="1.0",
            allowed_roles=roles
        )

    @staticmethod
    def infer_section(text: str) -> Optional[str]:
        """Find the nearest preceding markdown heading or uppercase section marker."""
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line_str = line.strip()
            if line_str.startswith("#"):
                return line_str.lstrip("#").strip()
            if line_str.isupper() and len(line_str) < 60:
                return line_str
        return None
