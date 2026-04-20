"""Admin authentication and brute force protection.

This module provides admin authentication via session cookies,
along with brute force protection through login attempt tracking.
"""

import time
from typing import TYPE_CHECKING, Annotated

import structlog
from fastapi import Cookie, HTTPException
from itsdangerous import BadSignature, SignatureExpired

from clorag.config import get_settings
from clorag.web.auth.sessions import (
    ADMIN_SESSION_MAX_AGE,
    get_session_serializer,
)

if TYPE_CHECKING:
    from clorag.core.analytics_db import AnalyticsDatabase

logger = structlog.get_logger()

# Brute force protection settings
LOGIN_LOCKOUT_THRESHOLD = 5  # Failed attempts before lockout
LOGIN_LOCKOUT_DURATION = 300  # 5 minutes lockout


class LoginAttemptTracker:
    """Track failed login attempts per IP for brute force protection.

    State is persisted in the analytics SQLite database so lockouts survive
    container restarts and rolling deploys. Lookups happen per request but
    the tables are indexed and tiny (one row per offender), so overhead is
    negligible compared to the PBKDF2/timing-safe password check.
    """

    def __init__(self, db: "AnalyticsDatabase | None" = None) -> None:
        self._db = db  # Resolved lazily so tests can inject a throwaway DB.

    def _store(self) -> "AnalyticsDatabase":
        if self._db is None:
            # Local import to avoid circular dependency at module import time.
            from clorag.core.analytics_db import AnalyticsDatabase

            settings = get_settings()
            self._db = AnalyticsDatabase(settings.analytics_database_path)
        return self._db

    def is_locked_out(self, ip: str) -> bool:
        """Check if IP is currently locked out."""
        return self.get_lockout_remaining(ip) > 0

    def get_lockout_remaining(self, ip: str) -> int:
        """Get seconds remaining in lockout, or 0 if not locked out."""
        locked_until = self._store().get_login_lockout_until(ip)
        if not locked_until:
            return 0
        remaining = int(locked_until - time.time())
        return max(0, remaining)

    def record_failed_attempt(self, ip: str) -> bool:
        """Record a failed login attempt. Returns True if now locked out."""
        now = time.time()
        store = self._store()
        _, should_lock = store.record_login_attempt(
            ip,
            now=now,
            window_seconds=LOGIN_LOCKOUT_DURATION,
            threshold=LOGIN_LOCKOUT_THRESHOLD,
        )
        if should_lock:
            store.set_login_lockout(ip, now + LOGIN_LOCKOUT_DURATION)
            return True
        return False

    def clear_attempts(self, ip: str) -> None:
        """Clear attempts on successful login."""
        self._store().clear_login_attempts(ip)


# Global login attempt tracker singleton
_login_tracker: LoginAttemptTracker | None = None


def get_login_tracker() -> LoginAttemptTracker:
    """Get or create login attempt tracker singleton."""
    global _login_tracker
    if _login_tracker is None:
        _login_tracker = LoginAttemptTracker()
    return _login_tracker


def verify_admin(
    admin_session: Annotated[str | None, Cookie()] = None,
) -> bool:
    """Verify admin access via session cookie.

    Session cookies are signed with itsdangerous and validated against
    the session secret.
    """
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")

    if admin_session:
        try:
            serializer = get_session_serializer()
            data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
            if data.get("authenticated"):
                return True
        except SignatureExpired:
            logger.debug("Admin session expired")
        except BadSignature:
            logger.debug("Invalid admin session signature")

    raise HTTPException(status_code=401, detail="Unauthorized")
