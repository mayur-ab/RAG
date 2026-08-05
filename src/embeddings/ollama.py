import httpx
from typing import List
from src.embeddings.base import BaseEmbeddingProvider
from config.settings import settings
from config.logging_config import logger

BATCH_SIZE = 16


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Local Ollama embedding provider (e.g. nomic-embed-text)."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_EMBEDDING_MODEL,
        dim: int = 768,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _parse_embeddings_response(self, data: dict, expected_count: int) -> List[List[float]]:
        embeddings = data.get("embeddings")
        if embeddings:
            if len(embeddings) != expected_count:
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} embeddings, expected {expected_count}."
                )
            if any(not emb for emb in embeddings):
                raise RuntimeError("Ollama returned one or more empty embedding vectors.")
            if embeddings[0]:
                self._dim = len(embeddings[0])
            return embeddings

        embedding = data.get("embedding")
        if embedding and expected_count == 1:
            if not embedding:
                raise RuntimeError("Ollama returned an empty embedding vector.")
            self._dim = len(embedding)
            return [embedding]

        raise RuntimeError(f"Unexpected Ollama embedding response: {list(data.keys())}")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts if len(texts) > 1 else texts[0]}

        try:
            response = httpx.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
            return self._parse_embeddings_response(response.json(), len(texts))
        except httpx.HTTPError as exc:
            logger.warning(f"Ollama /api/embed failed ({exc}); trying legacy /api/embeddings endpoint.")
            return [self._embed_text_legacy(text) for text in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def _embed_text_legacy(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        response = httpx.post(
            url,
            json={"model": self.model, "prompt": text},
            timeout=60.0,
        )
        response.raise_for_status()
        embedding = response.json().get("embedding", [])
        if not embedding:
            raise RuntimeError("Ollama legacy embeddings endpoint returned an empty vector.")
        self._dim = len(embedding)
        return embedding

    def embed_text(self, text: str) -> List[float]:
        try:
            return self.embed_documents([text])[0]
        except Exception as exc:
            raise RuntimeError(
                f"Ollama embedding failed for model '{self.model}' at {self.base_url}. "
                f"Ensure Ollama is running and the model is pulled (`ollama pull {self.model}`). "
                f"Original error: {exc}"
            ) from exc
