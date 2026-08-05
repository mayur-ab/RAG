import httpx
from typing import List, Optional
from src.embeddings.base import BaseEmbeddingProvider
from config.settings import settings


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI API Embedding Provider."""

    def __init__(self, api_key: Optional[str] = settings.OPENAI_API_KEY, model: str = settings.OPENAI_EMBEDDING_MODEL, dim: int = 1536):
        self.api_key = api_key
        self.model = model
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = httpx.post(url, json={"input": texts, "model": self.model}, headers=headers, timeout=20.0)
        response.raise_for_status()
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        if embeddings:
            self._dim = len(embeddings[0])
        return embeddings
