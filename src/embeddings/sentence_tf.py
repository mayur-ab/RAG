from typing import List
from src.embeddings.base import BaseEmbeddingProvider


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Local SentenceTransformer / HuggingFace embedding provider (BGE, E5, etc.)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._dim = 384
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()
