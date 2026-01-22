"""CSRF (Cross-Site Request Forgery) protection.

This module provides CSRF token generation, validation, and request verification
for protecting state-changing operations in the web application.
"""

import secrets
import time
from typing import Annotated

import structlog
from fastapi import Cookie, Header, HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired

from clorag.config import get_settings
from clorag.web.auth.sessions import (
    ADMIN_SESSION_MAX_AGE,
    get_session_serializer,
)

logger = structlog.get_logger()

# CSRF constants
CSRF_TOKEN_MAX_AGE = 3600  # 1 hour validity
CSRF_HEADER_NAME = "X-CSRF-Token"


def generate_csrf_token(session_id: str | None = None) -> str:
    """Generate a signed CSRF token.

    Args:
        session_id: Optional session ID to tie token to session.

    Returns:
        Signed CSRF token string.
    """
    serializer = get_session_serializer()
    # Include timestamp and optional session binding
    payload = {
        "csrf": secrets.token_hex(16),
        "ts": int(time.time()),
    }
    if session_id:
        payload["sid"] = session_id
    return serializer.dumps(payload)


def validate_csrf_token(
    token: str,
    session_id: str | None = None,
) -> bool:
    """Validate a CSRF token.

    Args:
        token: The CSRF token to validate.
        session_id: Optional session ID to verify token binding.

    Returns:
        True if valid, False otherwise.
    """
    try:
        serializer = get_session_serializer()
        payload = serializer.loads(token, max_age=CSRF_TOKEN_MAX_AGE)

        # Verify session binding if provided
        if session_id and payload.get("sid") != session_id:
            logger.warning("CSRF token session mismatch")
            return False

        return True
    except SignatureExpired:
        logger.warning("CSRF token expired")
        return False
    except BadSignature:
        logger.warning("CSRF token invalid signature")
        return False
    except Exception as e:
        logger.warning("CSRF validation error", error=str(e))
        return False


def verify_csrf(
    request: Request,
    x_csrf_token: Annotated[str | None, Header(alias=CSRF_HEADER_NAME)] = None,
    admin_session: Annotated[str | None, Cookie()] = None,
) -> bool:
    """Dependency to verify CSRF token on state-changing requests.

    Checks for CSRF token in X-CSRF-Token header.
    Must be used on POST, PUT, DELETE endpoints that modify state.

    Args:
        request: The FastAPI request object.
        x_csrf_token: CSRF token from header.
        admin_session: Admin session cookie for session binding.

    Returns:
        True if CSRF is valid.

    Raises:
        HTTPException: If CSRF validation fails.
    """
    # Skip CSRF for safe methods (GET, HEAD, OPTIONS)
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return True

    # Skip CSRF for API endpoints that use valid X-Admin-Password header authentication
    # (these are typically programmatic API calls, not browser requests)
    # SECURITY: Must verify password is correct before skipping CSRF to prevent bypass
    x_admin_password = request.headers.get("X-Admin-Password")
    if x_admin_password is not None:
        settings = get_settings()
        if settings.admin_password and secrets.compare_digest(
            x_admin_password.encode("utf-8"),
            settings.admin_password.get_secret_value().encode("utf-8"),
        ):
            return True
        # Invalid password - don't skip CSRF, continue to normal validation

    if not x_csrf_token:
        raise HTTPException(
            status_code=403,
            detail="CSRF token missing. Include X-CSRF-Token header.",
        )

    # Extract session ID for binding verification (optional)
    session_id = None
    if admin_session:
        try:
            serializer = get_session_serializer()
            data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
            session_id = data.get("session_id")
        except Exception:
            pass  # Session validation failure is handled elsewhere

    if not validate_csrf_token(x_csrf_token, session_id):
        raise HTTPException(
            status_code=403,
            detail="CSRF token invalid or expired. Please refresh the page.",
        )

    return True
