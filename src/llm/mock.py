import re
from typing import Dict, Any, List, Optional
from src.llm.base import BaseLLMProvider, SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT


class MockLLMProvider(BaseLLMProvider):
    """Deterministic grounded mock LLM provider for zero-dependency offline operation."""

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

        user_block = build_rag_user_message(context, prompt, chat_compact=chat_compact)
        if not chat_compact and chat_history:
            from src.context.conversation import format_routing_turns
            legacy = format_routing_turns(chat_history[-4:])
            if legacy:
                user_block = f"Context:\n{context}\n\nPrior exchange:\n{legacy}\n\nCurrent question: {prompt}"

        if not context or not context.strip():
            return {
                "answer": "I could not find that information in the provided knowledge base.",
                "model": "mock-grounded-llm",
                "prompt_tokens": len(prompt.split()) + len(context.split()),
                "completion_tokens": 12
            }

        # Extract sentences from context that match prompt keywords
        query_words = set(re.findall(r"\w+", prompt.lower())) - {"what", "is", "the", "a", "an", "and", "of", "in", "to", "for", "how"}
        lines = [line.strip() for line in context.split("\n") if line.strip() and not line.startswith("---")]

        matching_lines = []
        for line in lines:
            line_words = set(re.findall(r"\w+", line.lower()))
            if query_words & line_words:
                matching_lines.append(line)

        if matching_lines:
            answer = "Based on the knowledge base:\n\n" + "\n".join(matching_lines[:5])
        else:
            # Return first snippet if query words are generic
            snippet = lines[0] if lines else ""
            answer = f"Based on the provided context: {snippet}" if snippet else "I could not find that information in the provided knowledge base."

        return {
            "answer": answer,
            "model": "mock-grounded-llm",
            "prompt_tokens": len(prompt.split()) + len(context.split()),
            "completion_tokens": len(answer.split())
        }

    def chat(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        if "Return ONLY valid JSON" in (system_prompt or ""):
            return {
                "answer": (
                    '{"preferences":["likes detailed explanations"],'
                    '"projects":["building local RAG"],'
                    '"interests":["chromadb","memory architecture"],'
                    '"facts":["uses Ollama locally"],'
                    '"style":{"preferred_answer_style":"detailed","likes_examples":true}}'
                ),
                "model": "mock-chat-llm",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 24,
            }
        if "Updated summary:" in prompt:
            user_line = next((line for line in prompt.splitlines() if line.startswith("User: ")), "")
            return {
                "answer": f"Rolling chat summary. {user_line}".strip(),
                "model": "mock-chat-llm",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 12,
            }
        if "Merged session summary:" in prompt:
            return {
                "answer": "Session summary covering prior chats and the latest chat topics.",
                "model": "mock-chat-llm",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 10,
            }
        if "Final session summary:" in prompt:
            return {
                "answer": "User discussed RAG architecture, ChromaDB, and long-term memory design.",
                "model": "mock-chat-llm",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 16,
            }
        if "summarize conversations" in (system_prompt or "").lower() or prompt.strip().endswith("Summary:"):
            return {
                "answer": "User discussed RAG architecture, ChromaDB, and long-term memory design.",
                "model": "mock-chat-llm",
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 16,
            }
        answer = f"[Direct mode] {prompt}"
        if chat_history:
            answer = f"[Direct mode, {len(chat_history)} prior turns] {prompt}"
        return {
            "answer": answer,
            "model": "mock-chat-llm",
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(answer.split()),
        }

    def generate_stream(
        self,
        prompt: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        chat_history: Optional[List[Dict[str, str]]] = None,
        chat_compact: Optional[str] = None,
    ):
        result = self.generate(
            prompt,
            context,
            system_prompt,
            temperature,
            max_tokens,
            chat_history,
            chat_compact,
        )
        for word in result["answer"].split():
            yield word + " "

    def chat_stream(
        self,
        prompt: str,
        chat_history=None,
        system_prompt: str = CHAT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        result = self.chat(prompt, chat_history, system_prompt, temperature, max_tokens)
        for word in result["answer"].split():
            yield word + " "
