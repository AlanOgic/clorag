"""Session management for conversation and admin sessions.

This module provides session storage and management for the web application,
including conversation session handling and admin session serialization.
"""

import time
import uuid
from collections import OrderedDict

import structlog
from fastapi import HTTPException
from itsdangerous import URLSafeTimedSerializer

from clorag.config import get_settings
from clorag.web.schemas import ConversationSession

logger = structlog.get_logger()

# Session constants
MAX_SESSIONS = 1000  # Maximum concurrent sessions
ADMIN_SESSION_COOKIE = "admin_session"
ADMIN_SESSION_MAX_AGE = 4 * 60 * 60  # 4 hours (reduced from 24 for security)


class SessionStore:
    """Thread-safe session store with LRU eviction and TTL."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, ConversationSession] = OrderedDict()
        self._max_sessions = max_sessions

    def create_session(self) -> ConversationSession:
        """Create a new conversation session."""
        self._cleanup_expired()
        session_id = str(uuid.uuid4())
        session = ConversationSession(session_id=session_id)
        self._sessions[session_id] = session
        self._sessions.move_to_end(session_id)
        # Evict oldest if over limit
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return session

    def get_session(self, session_id: str) -> ConversationSession | None:
        """Get a session by ID, returns None if not found or expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            del self._sessions[session_id]
            return None
        session.last_accessed = time.time()
        self._sessions.move_to_end(session_id)
        return session

    def get_or_create_session(self, session_id: str | None) -> ConversationSession:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()

    def _cleanup_expired(self) -> None:
        """Remove expired sessions (called periodically)."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]


# Global session store singleton
_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Get or create session store singleton."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store


def get_session_serializer() -> URLSafeTimedSerializer:
    """Get session serializer using admin password as secret key."""
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")
    return URLSafeTimedSerializer(settings.admin_password.get_secret_value())
