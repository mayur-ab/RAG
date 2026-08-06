import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, Optional


class QueryCache:
    """In-memory LRU cache for identical query responses."""

    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _make_key(
        self,
        query: str,
        use_rag: bool,
        model: str,
        chat_history: Optional[list] = None,
    ) -> str:
        history_slice = (chat_history or [])[-4:]
        payload = {
            "query": query.strip().lower(),
            "use_rag": use_rag,
            "model": model,
            "history": history_slice,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        query: str,
        use_rag: bool,
        model: str,
        chat_history: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        key = self._make_key(query, use_rag, model, chat_history)
        entry = self._store.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self.ttl_seconds:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        result = dict(entry["data"])
        result["cached"] = True
        return result

    def set(
        self,
        query: str,
        use_rag: bool,
        model: str,
        chat_history: Optional[list],
        data: Dict[str, Any],
    ) -> None:
        key = self._make_key(query, use_rag, model, chat_history)
        stored = dict(data)
        stored.pop("cached", None)
        self._store[key] = {"ts": time.time(), "data": stored}
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
