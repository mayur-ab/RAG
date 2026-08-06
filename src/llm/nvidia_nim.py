import httpx
from typing import Dict, Any, Optional, List
from src.llm.base import BaseLLMProvider, SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
from config.settings import settings
from config.logging_config import logger


class NvidiaNIMLLMProvider(BaseLLMProvider):
    """NVIDIA NIM Free API LLM Provider (OpenAI Compatible API)."""

    def __init__(
        self,
        api_key: Optional[str] = settings.NVIDIA_API_KEY,
        base_url: str = settings.NVIDIA_NIM_BASE_URL,
        model: str = settings.NVIDIA_NIM_MODEL
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
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
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Add it to .env or switch LLM_PROVIDER=ollama for local inference."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=180.0)
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            return {
                "answer": answer,
                "model": f"nvidia/{self.model}",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)
            }
        except Exception as exc:
            logger.error(f"NVIDIA NIM request failed: {exc}")
            raise RuntimeError(
                f"NVIDIA NIM request failed for model '{self.model}'. "
                f"Check your API key and network connectivity. Original error: {exc}"
            ) from exc

    def chat(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Add it to .env or switch LLM_PROVIDER=ollama for local inference."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for turn in chat_history[-20:]:
                role = turn.get("role")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=180.0)
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            return {
                "answer": answer,
                "model": f"nvidia/{self.model}",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        except Exception as exc:
            logger.error(f"NVIDIA NIM chat request failed: {exc}")
            raise RuntimeError(
                f"NVIDIA NIM chat request failed for model '{self.model}'. "
                f"Check your API key and network connectivity. Original error: {exc}"
            ) from exc
