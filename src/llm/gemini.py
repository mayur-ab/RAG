import httpx
from typing import Dict, Any, Optional
from src.llm.base import BaseLLMProvider, SYSTEM_PROMPT
from config.settings import settings


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini API LLM Provider via REST API."""

    def __init__(self, api_key: Optional[str] = settings.GEMINI_API_KEY, model: str = settings.GEMINI_LLM_MODEL):
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        full_text = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {prompt}\nAnswer:"
        payload = {
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
        }

        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {
            "answer": answer,
            "model": f"gemini/{self.model}",
            "prompt_tokens": len(full_text.split()),
            "completion_tokens": len(answer.split())
        }
