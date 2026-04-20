"""Authentication module for the web application.

This module provides authentication, session management, and CSRF protection
for the web application.
"""

from clorag.web.auth.admin import (
    LOGIN_LOCKOUT_DURATION,
    LOGIN_LOCKOUT_THRESHOLD,
    LoginAttemptTracker,
    get_login_tracker,
    verify_admin,
)
from clorag.web.auth.csrf import (
    CSRF_HEADER_NAME,
    CSRF_TOKEN_MAX_AGE,
    generate_csrf_token,
    validate_csrf_token,
    verify_csrf,
)
from clorag.web.auth.sessions import (
    ADMIN_SESSION_COOKIE,
    ADMIN_SESSION_MAX_AGE,
    MAX_SESSIONS,
    SessionStore,
    get_session_serializer,
    get_session_store,
    reset_session_serializer_cache,
)

__all__ = [
    # Sessions
    "ADMIN_SESSION_COOKIE",
    "ADMIN_SESSION_MAX_AGE",
    "MAX_SESSIONS",
    "SessionStore",
    "get_session_serializer",
    "get_session_store",
    "reset_session_serializer_cache",
    # Admin auth
    "LOGIN_LOCKOUT_DURATION",
    "LOGIN_LOCKOUT_THRESHOLD",
    "LoginAttemptTracker",
    "get_login_tracker",
    "verify_admin",
    # CSRF
    "CSRF_HEADER_NAME",
    "CSRF_TOKEN_MAX_AGE",
    "generate_csrf_token",
    "validate_csrf_token",
    "verify_csrf",
]
