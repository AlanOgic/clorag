"""Admin authentication and brute force protection.

This module provides admin authentication via session cookies or API header,
along with brute force protection through login attempt tracking.
"""

import secrets
import time
from typing import Annotated

import structlog
from fastapi import Cookie, Header, HTTPException
from itsdangerous import BadSignature, SignatureExpired

from clorag.config import get_settings
from clorag.web.auth.sessions import (
    ADMIN_SESSION_MAX_AGE,
    get_session_serializer,
)

logger = structlog.get_logger()

# Brute force protection settings
LOGIN_LOCKOUT_THRESHOLD = 5  # Failed attempts before lockout
LOGIN_LOCKOUT_DURATION = 300  # 5 minutes lockout


class LoginAttemptTracker:
    """Track failed login attempts per IP for brute force protection."""

    def __init__(self) -> None:
        self._attempts: dict[str, list[float]] = {}  # IP -> list of attempt timestamps
        self._lockouts: dict[str, float] = {}  # IP -> lockout expiry timestamp

    def is_locked_out(self, ip: str) -> bool:
        """Check if IP is currently locked out."""
        if ip in self._lockouts:
            if time.time() < self._lockouts[ip]:
                return True
            # Lockout expired, remove it
            del self._lockouts[ip]
            if ip in self._attempts:
                del self._attempts[ip]
        return False

    def get_lockout_remaining(self, ip: str) -> int:
        """Get seconds remaining in lockout, or 0 if not locked out."""
        if ip in self._lockouts:
            remaining = int(self._lockouts[ip] - time.time())
            return max(0, remaining)
        return 0

    def record_failed_attempt(self, ip: str) -> bool:
        """Record a failed login attempt. Returns True if now locked out."""
        now = time.time()

        # Clean old attempts (older than lockout duration)
        if ip in self._attempts:
            self._attempts[ip] = [
                t for t in self._attempts[ip] if now - t < LOGIN_LOCKOUT_DURATION
            ]
        else:
            self._attempts[ip] = []

        self._attempts[ip].append(now)

        # Check if should be locked out
        if len(self._attempts[ip]) >= LOGIN_LOCKOUT_THRESHOLD:
            self._lockouts[ip] = now + LOGIN_LOCKOUT_DURATION
            return True
        return False

    def clear_attempts(self, ip: str) -> None:
        """Clear attempts on successful login."""
        if ip in self._attempts:
            del self._attempts[ip]
        if ip in self._lockouts:
            del self._lockouts[ip]


# Global login attempt tracker singleton
_login_tracker: LoginAttemptTracker | None = None


def get_login_tracker() -> LoginAttemptTracker:
    """Get or create login attempt tracker singleton."""
    global _login_tracker
    if _login_tracker is None:
        _login_tracker = LoginAttemptTracker()
    return _login_tracker


def verify_admin(
    x_admin_password: Annotated[str | None, Header()] = None,
    admin_session: Annotated[str | None, Cookie()] = None,
) -> bool:
    """Verify admin access via header password OR session cookie.

    Two authentication methods are supported:
    1. X-Admin-Password header - for API calls (legacy support)
    2. admin_session cookie - for browser sessions (signed with itsdangerous)
    """
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")

    # Method 1: Check session cookie first (preferred for browser)
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

    # Method 2: Check header password (API calls / legacy)
    if x_admin_password is not None and secrets.compare_digest(
        x_admin_password.encode("utf-8"),
        settings.admin_password.get_secret_value().encode("utf-8"),
    ):
        return True

    raise HTTPException(status_code=401, detail="Unauthorized")
