import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.memory.models import UserPreferences, UserProfile, UserStyle


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProfileStore:
    """SQLite-backed long-term user profile and topic frequency tracking."""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL DEFAULT '',
                        preferences TEXT NOT NULL DEFAULT '{}',
                        interests TEXT NOT NULL DEFAULT '[]',
                        projects TEXT NOT NULL DEFAULT '[]',
                        facts TEXT NOT NULL DEFAULT '[]',
                        style TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS topic_frequency (
                        user_id TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        count INTEGER NOT NULL DEFAULT 1,
                        last_seen TEXT NOT NULL,
                        PRIMARY KEY (user_id, topic)
                    );

                    CREATE TABLE IF NOT EXISTS session_archives (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        message_count INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_topic_user_count
                        ON topic_frequency(user_id, count DESC);
                    CREATE INDEX IF NOT EXISTS idx_session_user
                        ON session_archives(user_id, created_at DESC);
                    """
                )
                self._ensure_column(conn, "user_profiles", "display_name", "TEXT NOT NULL DEFAULT ''")
                conn.commit()
            finally:
                conn.close()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row["name"] for row in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def get_profile(self, user_id: str) -> UserProfile:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM user_profiles WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                if row is None:
                    now = _utc_now()
                    conn.execute(
                        """
                        INSERT INTO user_profiles
                            (user_id, display_name, preferences, interests, projects, facts, style, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user_id,
                            "",
                            json.dumps(UserPreferences().model_dump()),
                            "[]",
                            "[]",
                            "[]",
                            json.dumps(UserStyle().model_dump()),
                            now,
                            now,
                        ),
                    )
                    conn.commit()
                    return UserProfile(user_id=user_id)
                return UserProfile(
                    user_id=row["user_id"],
                    display_name=(row["display_name"] if "display_name" in row.keys() else "") or "",
                    preferences=UserPreferences(**json.loads(row["preferences"] or "{}")),
                    interests=json.loads(row["interests"] or "[]"),
                    projects=json.loads(row["projects"] or "[]"),
                    facts=json.loads(row["facts"] or "[]"),
                    style=UserStyle(**json.loads(row["style"] or "{}")),
                )
            finally:
                conn.close()

    def save_profile(self, profile: UserProfile) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE user_profiles
                    SET display_name = ?, preferences = ?, interests = ?, projects = ?, facts = ?, style = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (
                        profile.display_name or "",
                        json.dumps(profile.preferences.model_dump()),
                        json.dumps(profile.interests),
                        json.dumps(profile.projects),
                        json.dumps(profile.facts),
                        json.dumps(profile.style.model_dump()),
                        _utc_now(),
                        profile.user_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _merge_unique(existing: List[str], incoming: List[str], max_items: int) -> List[str]:
        merged: List[str] = []
        seen = set()
        for item in existing + incoming:
            normalized = (item or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
        return merged[:max_items]

    def merge_profile_updates(
        self,
        user_id: str,
        *,
        preferences: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
        projects: Optional[List[str]] = None,
        facts: Optional[List[str]] = None,
        style_updates: Optional[Dict[str, Any]] = None,
        max_items: int = 20,
    ) -> UserProfile:
        profile = self.get_profile(user_id)

        if preferences:
            for pref in preferences:
                lowered = pref.lower()
                if "detailed" in lowered or "thorough" in lowered:
                    profile.preferences.response_style = "detailed"
                elif "simple" in lowered or "concise" in lowered or "brief" in lowered:
                    profile.preferences.response_style = "simple"
                if "beginner" in lowered:
                    profile.preferences.technical_level = "beginner"
                elif "advanced" in lowered or "expert" in lowered:
                    profile.preferences.technical_level = "advanced"

        profile.interests = self._merge_unique(profile.interests, interests or [], max_items)
        profile.projects = self._merge_unique(profile.projects, projects or [], max_items)
        profile.facts = self._merge_unique(profile.facts, facts or [], max_items)

        if style_updates:
            for key, value in style_updates.items():
                if hasattr(profile.style, key) and value is not None:
                    setattr(profile.style, key, value)
            if style_updates.get("preferred_answer_style"):
                profile.style.preferred_answer_style = str(style_updates["preferred_answer_style"])

        self.save_profile(profile)
        return profile

    def set_display_name(self, user_id: str, display_name: str) -> UserProfile:
        profile = self.get_profile(user_id)
        profile.display_name = display_name.strip()
        profile.facts = self._merge_unique(
            profile.facts,
            [f"User's name is {profile.display_name}"],
            max_items=20,
        )
        self.save_profile(profile)
        return profile

    def list_recent_archives(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT summary, message_count, created_at
                    FROM session_archives
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
                return [
                    {
                        "summary": row["summary"],
                        "message_count": row["message_count"],
                        "created_at": row["created_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def record_topics(self, user_id: str, topics: List[str]) -> None:
        if not topics:
            return
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                for topic in topics:
                    conn.execute(
                        """
                        INSERT INTO topic_frequency (user_id, topic, count, last_seen)
                        VALUES (?, ?, 1, ?)
                        ON CONFLICT(user_id, topic) DO UPDATE SET
                            count = count + 1,
                            last_seen = excluded.last_seen
                        """,
                        (user_id, topic, now),
                    )
                conn.commit()
            finally:
                conn.close()

    def top_topics(self, user_id: str, limit: int = 8) -> List[str]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT topic FROM topic_frequency
                    WHERE user_id = ?
                    ORDER BY count DESC, last_seen DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
                return [row["topic"] for row in rows]
            finally:
                conn.close()

    def record_session_archive(
        self,
        archive_id: str,
        user_id: str,
        summary: str,
        message_count: int,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO session_archives (id, user_id, summary, message_count, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (archive_id, user_id, summary, message_count, _utc_now()),
                )
                conn.commit()
            finally:
                conn.close()

    def delete_user_data(self, user_id: str) -> Dict[str, int]:
        """Delete all SQLite user-memory rows. Does not touch RAG document indexes."""
        with self._lock:
            conn = self._connect()
            try:
                topics = conn.execute(
                    "DELETE FROM topic_frequency WHERE user_id = ?",
                    (user_id,),
                ).rowcount
                archives = conn.execute(
                    "DELETE FROM session_archives WHERE user_id = ?",
                    (user_id,),
                ).rowcount
                profiles = conn.execute(
                    "DELETE FROM user_profiles WHERE user_id = ?",
                    (user_id,),
                ).rowcount
                conn.commit()
                return {
                    "profiles_deleted": profiles or 0,
                    "topics_deleted": topics or 0,
                    "archives_deleted": archives or 0,
                }
            finally:
                conn.close()
