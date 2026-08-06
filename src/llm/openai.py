import httpx
from typing import Dict, Any, Optional, List
from src.llm.base import BaseLLMProvider, SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
from config.settings import settings


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI API LLM Provider."""

    def __init__(self, api_key: Optional[str] = settings.OPENAI_API_KEY, model: str = settings.OPENAI_LLM_MODEL):
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
            raise RuntimeError("OPENAI_API_KEY is not set.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
        ]
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return {
            "answer": answer,
            "model": f"openai/{self.model}",
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)
        }

    def chat(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for turn in chat_history[-20:]:
                role = turn.get("role")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return {
            "answer": answer,
            "model": f"openai/{self.model}",
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
