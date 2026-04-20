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


# Cache of the PBKDF2-derived HMAC key when SESSION_SECRET is not set.
# Derivation costs ~250-400 ms at 600K iterations, which would pin a CPU
# core if recomputed on every admin request. The cache is keyed by the
# admin password so rotating it (via restart or reload-env) naturally
# invalidates the entry. reset_session_serializer_cache() forces refresh.
_session_secret_cache: dict[str, str] = {}


def reset_session_serializer_cache() -> None:
    """Clear the cached derived session secret.

    Called by /api/admin/reload-env after get_settings.cache_clear(), so
    rotating ADMIN_PASSWORD or SESSION_SECRET without a restart forces
    re-derivation on the next request.
    """
    _session_secret_cache.clear()


def get_session_serializer() -> URLSafeTimedSerializer:
    """Get session serializer using dedicated session secret.

    Falls back to a cached PBKDF2-derived key from admin_password if
    session_secret is not set.
    """
    import hashlib

    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")

    if settings.session_secret:
        secret = settings.session_secret.get_secret_value()
    else:
        raw = settings.admin_password.get_secret_value()
        cached = _session_secret_cache.get(raw)
        if cached is None:
            # OWASP 2023 minimum for PBKDF2-HMAC-SHA256 is 600K iterations.
            cached = hashlib.pbkdf2_hmac(
                "sha256", raw.encode(), b"clorag-session-signing", 600_000
            ).hex()
            _session_secret_cache.clear()
            _session_secret_cache[raw] = cached
        secret = cached

    return URLSafeTimedSerializer(secret)
