"""Messages database for admin-managed announcements on the public page."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class Message:
    """A single message record."""

    def __init__(
        self,
        id: str,
        title: str,
        body: str,
        message_type: str,
        link_url: str | None,
        is_active: bool,
        sort_order: int,
        created_at: datetime,
        expires_at: datetime | None,
    ) -> None:
        self.id = id
        self.title = title
        self.body = body
        self.message_type = message_type
        self.link_url = link_url
        self.is_active = is_active
        self.sort_order = sort_order
        self.created_at = created_at
        self.expires_at = expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "message_type": self.message_type,
            "link_url": self.link_url,
            "is_active": self.is_active,
            "sort_order": self.sort_order,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


class MessagesDatabase:
    """SQLite database for admin-managed messages."""

    VALID_TYPES = {"info", "warning", "feature", "fix"}

    def __init__(self, db_path: str = "data/analytics.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    message_type TEXT NOT NULL
                        CHECK(message_type IN ('info', 'warning', 'feature', 'fix')),
                    link_url TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_active
                ON messages(is_active, sort_order)
            """)
            conn.commit()

    def _row_to_message(self, row: tuple[Any, ...]) -> Message:
        return Message(
            id=row[0],
            title=row[1],
            body=row[2],
            message_type=row[3],
            link_url=row[4],
            is_active=bool(row[5]),
            sort_order=row[6],
            created_at=datetime.fromisoformat(row[7]) if row[7] else datetime.now(),
            expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
        )

    def get_active_messages(self) -> list[Message]:
        """Get active, non-expired messages ordered by sort_order."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, title, body, message_type, link_url,
                       is_active, sort_order, created_at, expires_at
                FROM messages
                WHERE is_active = 1
                  AND (expires_at IS NULL OR expires_at > datetime('now'))
                ORDER BY sort_order ASC, created_at DESC
                """,
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def get_all_messages(self) -> list[Message]:
        """Get all messages for admin view."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, title, body, message_type, link_url,
                       is_active, sort_order, created_at, expires_at
                FROM messages
                ORDER BY sort_order ASC, created_at DESC
                """,
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def get_message(self, message_id: str) -> Message | None:
        """Get a single message by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, title, body, message_type, link_url,
                       is_active, sort_order, created_at, expires_at
                FROM messages WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        return self._row_to_message(row) if row else None

    def create_message(
        self,
        title: str,
        body: str,
        message_type: str = "info",
        link_url: str | None = None,
        is_active: bool = True,
        sort_order: int = 0,
        expires_at: str | None = None,
    ) -> Message:
        """Create a new message."""
        if message_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid message_type: {message_type}")
        message_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages (id, title, body, message_type, link_url,
                                      is_active, sort_order, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, title, body, message_type, link_url,
                 int(is_active), sort_order, expires_at),
            )
            conn.commit()
        return self.get_message(message_id)  # type: ignore[return-value]

    def update_message(
        self,
        message_id: str,
        title: str | None = None,
        body: str | None = None,
        message_type: str | None = None,
        link_url: str | None = ...,  # type: ignore[assignment]
        is_active: bool | None = None,
        sort_order: int | None = None,
        expires_at: str | None = ...,  # type: ignore[assignment]
    ) -> Message | None:
        """Update a message. Only provided fields are changed."""
        if message_type is not None and message_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid message_type: {message_type}")
        existing = self.get_message(message_id)
        if not existing:
            return None
        fields: list[str] = []
        values: list[Any] = []
        if title is not None:
            fields.append("title = ?")
            values.append(title)
        if body is not None:
            fields.append("body = ?")
            values.append(body)
        if message_type is not None:
            fields.append("message_type = ?")
            values.append(message_type)
        if link_url is not ...:  # type: ignore[comparison-overlap]
            fields.append("link_url = ?")
            values.append(link_url)
        if is_active is not None:
            fields.append("is_active = ?")
            values.append(int(is_active))
        if sort_order is not None:
            fields.append("sort_order = ?")
            values.append(sort_order)
        if expires_at is not ...:  # type: ignore[comparison-overlap]
            fields.append("expires_at = ?")
            values.append(expires_at)
        if not fields:
            return existing
        values.append(message_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"UPDATE messages SET {', '.join(fields)} WHERE id = ?",
                values,
            )
            conn.commit()
        return self.get_message(message_id)

    def delete_message(self, message_id: str) -> bool:
        """Delete a message. Returns True if deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
            conn.commit()
        return cursor.rowcount > 0


# Singleton
_messages_db: MessagesDatabase | None = None


def get_messages_database() -> MessagesDatabase:
    """Get or create the singleton MessagesDatabase instance."""
    global _messages_db
    if _messages_db is None:
        from clorag.config import get_settings
        settings = get_settings()
        _messages_db = MessagesDatabase(db_path=settings.analytics_database_path)
    return _messages_db
