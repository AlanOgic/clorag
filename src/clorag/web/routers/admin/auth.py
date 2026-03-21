"""Admin authentication endpoints.

Provides login, logout, session management, CSRF tokens, and database backup.
"""

import io
import secrets
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from slowapi.util import get_remote_address

from clorag.config import get_settings
from clorag.web.auth import (
    ADMIN_SESSION_COOKIE,
    ADMIN_SESSION_MAX_AGE,
    LOGIN_LOCKOUT_DURATION,
    generate_csrf_token,
    get_login_tracker,
    get_session_serializer,
    verify_admin,
)
from clorag.web.dependencies import limiter
from clorag.web.schemas import LoginRequest, LoginResponse

router = APIRouter(tags=["Authentication"])
logger = structlog.get_logger()


@router.post("/login")
@limiter.limit("10/minute")
async def api_admin_login(
    request: Request,
    login_req: LoginRequest,
    response: Response,
) -> LoginResponse:
    """Login and set session cookie.

    Returns success status and sets httponly cookie on success.
    Implements brute force protection with lockout after 5 failed attempts.
    """
    # Import itsdangerous for BadSignature/SignatureExpired is not needed here
    # since we only create tokens, not validate them
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")

    # Get client IP for brute force tracking
    client_ip = get_remote_address(request)
    tracker = get_login_tracker()

    # Check if IP is locked out
    if tracker.is_locked_out(client_ip):
        remaining = tracker.get_lockout_remaining(client_ip)
        logger.warning(
            "Login attempt from locked out IP", ip=client_ip, remaining=remaining
        )
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {remaining} seconds.",
        )

    # Verify password with timing-safe comparison
    if not secrets.compare_digest(
        login_req.password.encode("utf-8"),
        settings.admin_password.get_secret_value().encode("utf-8"),
    ):
        # Record failed attempt
        locked_out = tracker.record_failed_attempt(client_ip)
        logger.warning("Failed login attempt", ip=client_ip, locked_out=locked_out)
        if locked_out:
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Try again in {LOGIN_LOCKOUT_DURATION} seconds.",
            )
        raise HTTPException(status_code=401, detail="Invalid password")

    # Successful login - clear any failed attempts
    tracker.clear_attempts(client_ip)

    # Create signed session token with unique session_id for CSRF binding
    serializer = get_session_serializer()
    session_id = secrets.token_hex(16)  # Generate unique session identifier
    token = serializer.dumps(
        {
            "authenticated": True,
            "ts": time.time(),
            "session_id": session_id,  # SECURITY: Used for CSRF token binding validation
        }
    )

    # Set httponly cookie (secure based on config)
    response.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=token,
        max_age=ADMIN_SESSION_MAX_AGE,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="strict",
    )

    logger.info("Successful admin login", ip=client_ip)
    return LoginResponse(success=True, message="Login successful")


@router.post("/logout")
async def api_admin_logout(response: Response) -> dict[str, Any]:
    """Logout and clear session cookie."""
    settings = get_settings()
    response.delete_cookie(
        key=ADMIN_SESSION_COOKIE,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="strict",
    )
    return {"success": True, "message": "Logged out"}


@router.get("/session")
async def api_admin_session(
    admin_session: Annotated[str | None, Cookie()] = None,
) -> dict[str, bool]:
    """Check if current session is valid.

    Returns authenticated status without requiring password.
    """
    from itsdangerous import BadSignature, SignatureExpired

    if not admin_session:
        return {"authenticated": False}

    try:
        serializer = get_session_serializer()
        data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
        if data.get("authenticated"):
            return {"authenticated": True}
    except (SignatureExpired, BadSignature):
        pass

    return {"authenticated": False}


@router.get("/csrf-token")
async def api_admin_csrf_token(
    admin_session: Annotated[str | None, Cookie()] = None,
) -> dict[str, str]:
    """Get a CSRF token bound to the current session.

    Requires a valid admin session cookie. The token is bound to the
    session ID and valid for 1 hour.
    """
    from itsdangerous import BadSignature, SignatureExpired

    if not admin_session:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        serializer = get_session_serializer()
        data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
        if not data.get("authenticated"):
            raise HTTPException(status_code=401, detail="Authentication required")
        session_id = data.get("session_id")
    except (SignatureExpired, BadSignature):
        raise HTTPException(
            status_code=401, detail="Session expired, please login again"
        )

    token = generate_csrf_token(session_id)
    return {"csrf_token": token}


@router.get("/backup")
@limiter.limit("3/hour")
async def api_admin_backup(
    request: Request,
    _auth: Annotated[bool, Depends(verify_admin)],
) -> StreamingResponse:
    """Download a backup ZIP of all SQLite databases."""
    settings = get_settings()
    camera_db_path = Path(settings.database_path)
    analytics_db_path = Path(settings.analytics_database_path)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if camera_db_path.exists():
            zf.write(camera_db_path, "clorag.db")
            logger.info("Added camera database to backup", path=str(camera_db_path))
        if analytics_db_path.exists():
            zf.write(analytics_db_path, "analytics.db")
            logger.info("Added analytics database to backup", path=str(analytics_db_path))

    zip_buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clorag_backup_{timestamp}.zip"

    logger.info("Database backup created", filename=filename)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
