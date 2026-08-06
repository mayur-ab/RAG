import hashlib
import os
from typing import Optional


def normalize_source_path(source: str) -> str:
    """Canonical path for deduplication lookups (local files only)."""
    if source.startswith("http://") or source.startswith("https://"):
        return source.rstrip("/")
    return os.path.normpath(source).replace("\\", "/")


def stable_document_id(source: str) -> str:
    """Stable document id from normalized source path (avoids basename collisions)."""
    normalized = normalize_source_path(source)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"doc_{digest[:16]}"


def compute_source_fingerprint(source: str) -> Optional[str]:
    """Fast change detector: mtime + size. None for URLs."""
    if source.startswith("http://") or source.startswith("https://"):
        return None
    if not os.path.isfile(source):
        return None
    stat = os.stat(source)
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def file_updated_at_iso(source: str) -> Optional[str]:
    """File mtime as ISO string (legacy dedup fallback)."""
    if not os.path.isfile(source):
        return None
    from datetime import datetime

    stat = os.stat(source)
    return datetime.utcfromtimestamp(stat.st_mtime).isoformat()
