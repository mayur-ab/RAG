import json
import math
import re
from typing import Any, Dict, List, Optional

import httpx

from config.settings import settings


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMJudge:
    """LLM-as-judge evaluators (LangSmith-style) without requiring reference answers."""

    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        model: Optional[str] = None,
        provider: str = settings.LLM_PROVIDER,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model or settings.OLLAMA_LLM_MODEL
        self.provider = provider

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        }
        response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        return (data.get("message", {}).get("content") or data.get("response") or "").strip()

    def _grade(self, system_prompt: str, user_prompt: str, result_key: str) -> Dict[str, Any]:
        if self.provider == "mock":
            return {"score": 1.0, "pass": True, "explanation": "mock judge", result_key: True}

        raw = self._call_ollama(system_prompt, user_prompt)
        parsed = _extract_json_object(raw) or {}
        passed = bool(parsed.get(result_key, parsed.get("pass", False)))
        explanation = str(parsed.get("explanation", raw[:240]))
        return {
            "score": 1.0 if passed else 0.0,
            "pass": passed,
            "explanation": explanation,
            result_key: passed,
        }

    def score_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        system_prompt = (
            "You grade whether an answer addresses the user's question. "
            'Respond with JSON only: {"explanation": "...", "relevant": true|false}'
        )
        user_prompt = f"QUESTION: {question}\nANSWER: {answer}"
        result = self._grade(system_prompt, user_prompt, "relevant")
        result["metric"] = "answer_relevance"
        return result

    def score_groundedness(self, context: str, answer: str) -> Dict[str, Any]:
        system_prompt = (
            "You grade whether an answer is grounded in the provided facts and does not hallucinate. "
            'Respond with JSON only: {"explanation": "...", "grounded": true|false}'
        )
        user_prompt = f"FACTS:\n{context}\n\nANSWER: {answer}"
        result = self._grade(system_prompt, user_prompt, "grounded")
        result["metric"] = "groundedness"
        return result

    def score_retrieval_relevance(self, question: str, context: str) -> Dict[str, Any]:
        system_prompt = (
            "You grade whether retrieved facts are relevant to the question. "
            "Facts are relevant if they contain keywords or semantic meaning related to the question. "
            'Respond with JSON only: {"explanation": "...", "relevant": true|false}'
        )
        user_prompt = f"QUESTION: {question}\nFACTS:\n{context}"
        result = self._grade(system_prompt, user_prompt, "relevant")
        result["metric"] = "retrieval_relevance"
        return result

    def evaluate_all(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        relevance = self.score_relevance(question, answer)
        groundedness = self.score_groundedness(context, answer)
        retrieval = self.score_retrieval_relevance(question, context)
        scores = [relevance["score"], groundedness["score"], retrieval["score"]]
        return {
            "answer_relevance": relevance,
            "groundedness": groundedness,
            "retrieval_relevance": retrieval,
            "aggregate_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }
