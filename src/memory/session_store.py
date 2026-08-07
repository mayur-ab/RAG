import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """SQLite store for user sessions (multiple chats → one session compact)."""

    def __init__(self, db_path: str):
        import os

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
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active',
                        working_compact TEXT NOT NULL DEFAULT '',
                        compact_summary TEXT NOT NULL DEFAULT '',
                        chat_count INTEGER NOT NULL DEFAULT 0,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_user_sessions_user_status
                        ON user_sessions(user_id, status, updated_at DESC);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM user_sessions WHERE session_id = ? AND user_id = ?",
                    (sid, user_id),
                ).fetchone()
                if row and row["status"] == "active":
                    return dict(row)
                conn.execute(
                    """
                    INSERT INTO user_sessions
                        (session_id, user_id, status, working_compact, compact_summary,
                         chat_count, started_at, updated_at)
                    VALUES (?, ?, 'active', '', '', 0, ?, ?)
                    """,
                    (sid, user_id, now, now),
                )
                conn.commit()
                return {
                    "session_id": sid,
                    "user_id": user_id,
                    "status": "active",
                    "working_compact": "",
                    "compact_summary": "",
                    "chat_count": 0,
                    "started_at": now,
                    "updated_at": now,
                }
            finally:
                conn.close()

    def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                if user_id:
                    row = conn.execute(
                        "SELECT * FROM user_sessions WHERE session_id = ? AND user_id = ?",
                        (session_id, user_id),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT * FROM user_sessions WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_active_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT * FROM user_sessions
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (user_id,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def update_working_compact(
        self,
        session_id: str,
        user_id: str,
        working_compact: str,
        *,
        increment_chat_count: bool = True,
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                if increment_chat_count:
                    conn.execute(
                        """
                        UPDATE user_sessions
                        SET working_compact = ?, chat_count = chat_count + 1, updated_at = ?
                        WHERE session_id = ? AND user_id = ? AND status = 'active'
                        """,
                        (working_compact, _utc_now(), session_id, user_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE user_sessions
                        SET working_compact = ?, updated_at = ?
                        WHERE session_id = ? AND user_id = ? AND status = 'active'
                        """,
                        (working_compact, _utc_now(), session_id, user_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def archive_session(
        self,
        session_id: str,
        user_id: str,
        compact_summary: str,
    ) -> Optional[Dict[str, Any]]:
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE user_sessions
                    SET status = 'archived',
                        compact_summary = ?,
                        working_compact = '',
                        ended_at = ?,
                        updated_at = ?
                    WHERE session_id = ? AND user_id = ? AND status = 'active'
                    """,
                    (compact_summary, now, now, session_id, user_id),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM user_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def list_archived_sessions(self, user_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT session_id, compact_summary, working_compact, chat_count, started_at, ended_at
                    FROM user_sessions
                    WHERE user_id = ? AND status = 'archived'
                    ORDER BY ended_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
                results: List[Dict[str, Any]] = []
                for row in rows:
                    summary = (row["compact_summary"] or row["working_compact"] or "").strip()
                    if not summary:
                        ended = (row["ended_at"] or row["started_at"] or "")[:10]
                        chat_count = int(row["chat_count"] or 0)
                        summary = f"Session - {chat_count} chat(s)" + (f" - {ended}" if ended else "")
                    results.append(
                        {
                            "session_id": row["session_id"],
                            "summary": summary,
                            "chat_count": row["chat_count"],
                            "started_at": row["started_at"],
                            "ended_at": row["ended_at"],
                        }
                    )
                return results
            finally:
                conn.close()

    def delete_user_sessions(self, user_id: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                count = conn.execute(
                    "DELETE FROM user_sessions WHERE user_id = ?",
                    (user_id,),
                ).rowcount
                conn.commit()
                return count or 0
            finally:
                conn.close()
