import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.logging_config import logger
from src.context.chat_compact import ChatCompactService
from src.memory.episodic_store import EpisodicMemoryStore
from src.memory.extractor import MemoryExtractor, extract_topics_from_query
from src.memory.models import EpisodicMemoryHit, UserMemoryContext
from src.context.user_memory_messages import extract_user_name, is_user_memory_message
from src.memory.profile_store import ProfileStore
from src.memory.prompt import format_memory_context
from src.memory.session_store import SessionStore


_USER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{8,64}$")


def validate_user_id(user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    cleaned = user_id.strip()
    if not _USER_ID_RE.match(cleaned):
        raise ValueError("Invalid user_id: use 8-64 alphanumeric characters, hyphens, or underscores.")
    return cleaned


def validate_session_id(session_id: Optional[str]) -> Optional[str]:
    if not session_id:
        return None
    cleaned = session_id.strip()
    if not _SESSION_ID_RE.match(cleaned):
        raise ValueError("Invalid session_id.")
    return cleaned


class MemoryManager:
    """Orchestrates user profile, sessions, and episodic memory."""

    def __init__(
        self,
        profile_store: ProfileStore,
        session_store: SessionStore,
        episodic_store: EpisodicMemoryStore,
        embedding_provider,
        llm_provider,
        *,
        memory_top_k: int = 5,
        max_profile_items: int = 20,
    ):
        self.profile_store = profile_store
        self.session_store = session_store
        self.episodic_store = episodic_store
        self.embedding_provider = embedding_provider
        self.extractor = MemoryExtractor(llm_provider)
        self.compact_service = ChatCompactService(llm_provider)
        self.memory_top_k = memory_top_k
        self.max_profile_items = max_profile_items

    def build_memory_context(self, user_id: str, query: str) -> UserMemoryContext:
        profile = self.profile_store.get_profile(user_id)
        frequent_topics = self.profile_store.top_topics(user_id)

        episodic: List[EpisodicMemoryHit] = []
        if query.strip():
            try:
                embedding = self.embedding_provider.embed_text(query)
                if embedding:
                    for hit in self.episodic_store.search(embedding, user_id, top_k=self.memory_top_k):
                        episodic.append(
                            EpisodicMemoryHit(
                                text=hit.get("text", ""),
                                memory_type=hit.get("memory_type", "summary"),
                                score=float(hit.get("score", 0.0)),
                                created_at=hit.get("created_at", ""),
                            )
                        )
            except Exception as exc:
                logger.warning(f"Episodic memory retrieval failed for user {user_id}: {exc}")

        return UserMemoryContext(
            profile=profile,
            episodic_memories=episodic,
            frequent_topics=frequent_topics,
        )

    def record_query(self, user_id: str, query: str) -> None:
        topics = extract_topics_from_query(query)
        if topics:
            self.profile_store.record_topics(user_id, topics)

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        active = self.session_store.get_active_session(user_id)
        if active and not session_id:
            return active
        return self.session_store.start_session(user_id, session_id)

    def end_chat(
        self,
        user_id: str,
        session_id: str,
        chat_id: str,
        chat_compact: str,
    ) -> Dict[str, Any]:
        session = self.session_store.get_session(session_id, user_id)
        if not session:
            return {"merged": False, "reason": "session_not_found"}
        if session.get("status") != "active":
            return {"merged": False, "reason": "session_not_active"}

        compact = (chat_compact or "").strip()
        working = session.get("working_compact") or ""
        if compact:
            working = self.compact_service.merge_into_session(working, compact)
        self.session_store.update_working_compact(
            session_id,
            user_id,
            working,
            increment_chat_count=bool(compact),
        )
        return {
            "merged": bool(compact),
            "session_id": session_id,
            "chat_id": chat_id,
            "chat_count": (session.get("chat_count") or 0) + (1 if compact else 0),
        }

    def end_session(
        self,
        user_id: str,
        session_id: str,
        final_chat_compact: Optional[str] = None,
    ) -> Dict[str, Any]:
        session = self.session_store.get_session(session_id, user_id)
        if not session:
            return {"archived": False, "reason": "session_not_found"}
        if session.get("status") != "active":
            return {"archived": False, "reason": "session_not_active"}

        working = session.get("working_compact") or ""
        if (final_chat_compact or "").strip():
            working = self.compact_service.merge_into_session(working, final_chat_compact.strip())

        summary = self.compact_service.finalize_session_summary(working)
        if not summary and working.strip():
            summary = working.strip()
        if not summary and (final_chat_compact or "").strip():
            summary = final_chat_compact.strip()
        if not summary:
            chat_count = int(session.get("chat_count") or 0)
            if chat_count > 0 or (final_chat_compact or "").strip():
                ended = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                summary = f"Session ended {ended} · {max(chat_count, 1)} chat(s)."
            else:
                self.session_store.archive_session(session_id, user_id, "")
                return {
                    "archived": True,
                    "reason": "empty_session",
                    "session_id": session_id,
                    "summary": "",
                    "chat_count": 0,
                }

        pseudo_history = [{"role": "assistant", "content": summary}]
        extraction = self.extractor.extract_facts(pseudo_history)
        self.profile_store.merge_profile_updates(
            user_id,
            preferences=extraction.preferences,
            interests=extraction.interests,
            projects=extraction.projects,
            facts=extraction.facts,
            style_updates=extraction.style,
            max_items=self.max_profile_items,
        )

        created_at = datetime.now(timezone.utc).isoformat()
        memory_id = str(uuid.uuid4())
        try:
            embedding = self.embedding_provider.embed_text(summary)
            if embedding:
                self.episodic_store.add_memory(
                    text=summary,
                    embedding=embedding,
                    user_id=user_id,
                    memory_type="session",
                    memory_id=memory_id,
                    created_at=created_at,
                )
        except Exception as exc:
            logger.warning(f"Failed to store session episodic memory for {user_id}: {exc}")

        archived = self.session_store.archive_session(session_id, user_id, summary)
        return {
            "archived": True,
            "session_id": session_id,
            "summary": summary,
            "chat_count": archived.get("chat_count") if archived else session.get("chat_count", 0),
        }

    def update_chat_compact(self, prior_compact: str, user_message: str, assistant_message: str) -> str:
        return self.compact_service.update_chat_compact(prior_compact, user_message, assistant_message)

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        profile = self.profile_store.get_profile(user_id)
        active = self.session_store.get_active_session(user_id)
        archived = self.session_store.list_archived_sessions(user_id)
        recent_sessions = [
            {
                "session_id": item["session_id"],
                "summary": item["summary"],
                "chat_count": item.get("chat_count", 0),
                "created_at": item.get("ended_at") or item.get("started_at", ""),
            }
            for item in archived
        ]
        return {
            **profile.model_dump(),
            "frequent_topics": self.profile_store.top_topics(user_id),
            "recent_sessions": recent_sessions,
            "active_session": active,
        }

    def capture_user_statement(
        self,
        user_id: str,
        query: str,
        routing_turns: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not is_user_memory_message(query):
            return {}

        saved: Dict[str, Any] = {}
        name = extract_user_name(query, routing_turns)
        if name:
            profile = self.profile_store.set_display_name(user_id, name)
            saved["display_name"] = profile.display_name
        return saved

    def delete_all_user_data(self, user_id: str) -> Dict[str, Any]:
        sqlite_counts = self.profile_store.delete_user_data(user_id)
        sessions_deleted = self.session_store.delete_user_sessions(user_id)
        episodic_deleted = self.episodic_store.delete_user_memories(user_id)
        sqlite_counts["sessions_deleted"] = sessions_deleted
        return {
            "user_id": user_id,
            "deleted": True,
            "sqlite": sqlite_counts,
            "episodic_memories_deleted": episodic_deleted,
        }

    # Legacy path — prefer end_chat/end_session with chat_compact
    def archive_session(self, user_id: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        if len(chat_history) < 2:
            return {"archived": False, "reason": "not_enough_messages", "message_count": len(chat_history)}
        summary = self.extractor.summarize_conversation(chat_history)
        if not summary:
            return {"archived": False, "reason": "empty_summary", "message_count": len(chat_history)}
        active = self.session_store.get_active_session(user_id) or self.session_store.start_session(user_id)
        result = self.end_chat(user_id, active["session_id"], str(uuid.uuid4()), summary)
        return {
            "archived": bool(result.get("merged")),
            "summary": summary,
            "message_count": len(chat_history),
            "session_id": result.get("session_id"),
            "reason": result.get("reason"),
        }
