import json
import httpx
from typing import Dict, Any, List, Optional, Iterator
from src.llm.base import BaseLLMProvider, SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
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

    def set_model(self, model: str) -> None:
        if not model or not model.strip():
            raise ValueError("Model name cannot be empty.")
        self.model = model.strip()

    def _stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with httpx.stream("POST", url, json=payload, timeout=180.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break

    def generate(
        self,
        prompt: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        chat_history: Optional[List[Dict[str, str]]] = None,
        chat_compact: Optional[str] = None,
    ) -> Dict[str, Any]:
        from src.context.conversation import build_rag_user_message

        url = f"{self.base_url}/api/chat"
        user_content = build_rag_user_message(context, prompt, chat_compact=chat_compact)
        if not chat_compact and chat_history:
            from src.context.conversation import format_routing_turns
            legacy = format_routing_turns(chat_history[-4:])
            if legacy:
                user_content = f"Context:\n{context}\n\nPrior exchange:\n{legacy}\n\nCurrent question: {prompt}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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

    def chat(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
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
                "answer": answer or "I could not generate a response.",
                "model": f"ollama/{self.model}",
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
        except Exception as exc:
            raise RuntimeError(
                f"Ollama chat request failed for model '{self.model}' at {self.base_url}. "
                f"Ensure Ollama is running and the model is pulled (`ollama pull {self.model}`). "
                f"Original error: {exc}"
            ) from exc

    def generate_stream(
        self,
        prompt: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        chat_history: Optional[List[Dict[str, str]]] = None,
        chat_compact: Optional[str] = None,
    ) -> Iterator[str]:
        from src.context.conversation import build_rag_user_message

        user_content = build_rag_user_message(context, prompt, chat_compact=chat_compact)
        if not chat_compact and chat_history:
            from src.context.conversation import format_routing_turns
            legacy = format_routing_turns(chat_history[-4:])
            if legacy:
                user_content = f"Context:\n{context}\n\nPrior exchange:\n{legacy}\n\nCurrent question: {prompt}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        yield from self._stream_chat(messages, temperature, max_tokens)

    def chat_stream(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Iterator[str]:
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for turn in chat_history[-20:]:
                role = turn.get("role")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": turn.get("content", "")})
        messages.append({"role": "user", "content": prompt})
        yield from self._stream_chat(messages, temperature, max_tokens)
