import httpx
from typing import Dict, Any
from src.llm.base import BaseLLMProvider, SYSTEM_PROMPT
from config.settings import settings


class OllamaLLMProvider(BaseLLMProvider):
    """Ollama Local LLM Provider."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: str = settings.OLLAMA_LLM_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        prompt: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        try:
            response = httpx.post(url, json=payload, timeout=180.0)
            response.raise_for_status()
            data = response.json()
            answer = data.get("message", {}).get("content", "").strip()
            if not answer:
                answer = data.get("response", "").strip()
            return {
                "answer": answer or "I could not find that information in the provided knowledge base.",
                "model": f"ollama/{self.model}",
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
        except Exception as exc:
            raise RuntimeError(
                f"Ollama LLM request failed for model '{self.model}' at {self.base_url}. "
                f"Ensure Ollama is running and the model is pulled (`ollama pull {self.model}`). "
                f"Original error: {exc}"
            ) from exc
