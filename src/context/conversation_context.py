from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ConversationContext:
    """User → session → chat hierarchy with ephemeral chat compact."""

    session_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_compact: str = ""
    routing_turns: List[Dict[str, str]] = field(default_factory=list)
    pinned_sources: List[str] = field(default_factory=list)

    @classmethod
    def from_request(
        cls,
        *,
        session_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_compact: Optional[str] = None,
        routing_turns: Optional[List[Dict[str, str]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        pinned_sources: Optional[List[str]] = None,
    ) -> "ConversationContext":
        turns = list(routing_turns or [])
        if not turns and chat_history:
            turns = chat_history[-2:]
        pins = [s.strip() for s in (pinned_sources or []) if (s or "").strip()]
        return cls(
            session_id=(session_id or "").strip() or None,
            chat_id=(chat_id or "").strip() or None,
            chat_compact=(chat_compact or "").strip(),
            routing_turns=turns[-2:],
            pinned_sources=pins[:3],
        )

    def has_conversation(self) -> bool:
        return bool(self.chat_compact.strip()) or bool(self.routing_turns)
