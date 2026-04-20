"""FastAPI web application for AI Search with Claude synthesis."""

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import asyncio

import anyio
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import (
    FastAPI,
    Request,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from clorag.config import get_settings

# Import auth module
# Import routers
from clorag.web.routers import admin_router, public_router

# Import schemas
# Import search module
from clorag.web.search import (
    get_sparse_embeddings,
    get_vectorstore,
)

# Initialize logger
logger = structlog.get_logger()


def _inject_structlog_processor(processor: Any) -> None:
    """Inject a processor into structlog's chain (before ConsoleRenderer).

    This allows the ingestion log capture processor to intercept log events
    without disrupting normal stdout logging.

    Args:
        processor: Structlog processor callable.
    """
    config = structlog.get_config()
    processors = list(config.get("processors", []))

    # Insert before the last processor (typically ConsoleRenderer)
    if processors:
        processors.insert(-1, processor)
    else:
        processors.append(processor)

    structlog.configure(processors=processors)


# =============================================================================
# Middleware Configuration
# =============================================================================


_CORS_ALLOWED_METHODS = "GET, POST, PUT, DELETE, OPTIONS"
_CORS_ALLOWED_HEADERS = (
    "Content-Type, X-Admin-Password, X-CSRF-Token, X-Requested-With, Authorization"
)
_CORS_MAX_AGE = "600"


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware that reads the allowlist from settings per-request.

    Unlike fastapi.middleware.cors.CORSMiddleware, which freezes allow_origins
    at construction time, this middleware re-reads get_settings() on every
    request so admins can update CORS_ALLOWED_ORIGINS and trigger a settings
    reload without restarting the container.

    Behavior:
      * Origin in allowlist → echoes Origin, allows credentials, exposes
        configured headers/methods.
      * Origin missing/disallowed → pass-through (no CORS headers) so the
        browser enforces same-origin. Non-browser clients are unaffected.
      * Preflight (OPTIONS with Access-Control-Request-Method) → 200 with
        headers; no downstream call.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        origin = request.headers.get("origin")
        allowed = origin in get_settings().cors_allowed_origins if origin else False

        if (
            request.method == "OPTIONS"
            and "access-control-request-method" in request.headers
        ):
            if not allowed:
                return Response(status_code=400)
            preflight = Response(status_code=200)
            preflight.headers["Access-Control-Allow-Origin"] = origin or ""
            preflight.headers["Access-Control-Allow-Credentials"] = "true"
            preflight.headers["Access-Control-Allow-Methods"] = _CORS_ALLOWED_METHODS
            preflight.headers["Access-Control-Allow-Headers"] = _CORS_ALLOWED_HEADERS
            preflight.headers["Access-Control-Max-Age"] = _CORS_MAX_AGE
            preflight.headers["Vary"] = "Origin"
            return preflight

        response = await call_next(request)
        if allowed and origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            existing_vary = response.headers.get("Vary")
            response.headers["Vary"] = (
                f"{existing_vary}, Origin" if existing_vary else "Origin"
            )
        return response


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeout."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with 60 second timeout.

        SSE streaming endpoints are exempt from timeout.
        """
        # Skip timeout for SSE streaming and long-running ingestion endpoints
        if (
            "/logs/stream" in request.url.path
            or "/v1/chat/completions" in request.url.path
            or "/api/legacy/" in request.url.path
        ):
            return await call_next(request)

        try:
            with anyio.fail_after(60):  # 60 second timeout
                return await call_next(request)
        except TimeoutError:
            return JSONResponse({"detail": "Request timeout"}, status_code=504)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses.

    SECURITY: Implements nonce-based CSP for inline scripts.
    Nonce is generated per-request and stored in request.state.csp_nonce.
    Templates should use: <script nonce="{{ request.state.csp_nonce }}">
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Add security headers to response with nonce-based CSP."""
        # Generate CSP nonce BEFORE processing request (so templates can access it)
        csp_nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = csp_nonce

        response = await call_next(request)

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # HSTS - enforce HTTPS for 1 year, include subdomains
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        # Content Security Policy with per-request nonce for inline scripts.
        # SECURITY: Single nonce-only policy for both public and admin pages.
        # Admin templates have been migrated from inline event handlers
        # (onclick="...") to data-action attributes handled by AdminActions
        # event delegation in static/js/admin.js, so no 'unsafe-inline' is
        # needed for script-src anywhere.
        script_src = (
            f"script-src 'self' 'nonce-{csp_nonce}'"
            " https://cdn.jsdelivr.net https://esm.sh"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"{script_src}; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://esm.sh; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://cdn.jsdelivr.net https://esm.sh; "
            "connect-src 'self' https://esm.sh https://cdn.jsdelivr.net;"
            "frame-ancestors 'none'"
        )
        # Prevent caching of API responses (may contain sensitive data)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response


# =============================================================================
# Background Scheduler for Draft Creation
# =============================================================================

_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler | None:
    """Get the background scheduler instance."""
    return _scheduler


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan with background scheduler for draft creation."""
    global _scheduler
    settings = get_settings()

    # Ensure all Qdrant collections exist (including custom_docs)
    # This is a critical check - the app cannot function without Qdrant
    try:
        vs = get_vectorstore()
        await vs.ensure_collections(hybrid=True)
        logger.info("Qdrant collections ensured")
    except Exception as e:
        logger.critical(
            "Failed to connect to Qdrant - application cannot start",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise RuntimeError(
            f"Qdrant initialization failed: {e}. "
            "Check QDRANT_URL and ensure Qdrant server is running."
        ) from e

    # Pre-load BM25 sparse embedding model to eliminate cold start latency (~2-3s)
    try:
        sparse_emb = get_sparse_embeddings()
        logger.info(
            "Sparse embedding model pre-loaded",
            model=sparse_emb._model_name,
        )
    except Exception as e:
        logger.warning("Failed to pre-load sparse embedding model", error=str(e))

    if settings.draft_polling_enabled:
        from clorag.drafts import check_and_create_drafts

        _scheduler = AsyncIOScheduler()
        _scheduler.add_job(
            check_and_create_drafts,
            "interval",
            minutes=settings.draft_poll_interval_minutes,
            id="draft_creation",
            replace_existing=True,
        )
        _scheduler.start()
        logger.info(
            "Draft creation scheduler started",
            interval_minutes=settings.draft_poll_interval_minutes,
        )

    # Analytics retention (PII/GDPR) + expired-lockout cleanup.
    if (
        settings.analytics_retention_days > 0
        or settings.analytics_anonymize_after_days > 0
    ):
        from clorag.core.analytics_db import AnalyticsDatabase

        if _scheduler is None:
            _scheduler = AsyncIOScheduler()
        if not _scheduler.running:
            try:
                _scheduler.start()
            except Exception as _e:
                logger.warning(
                    "Failed to start retention scheduler, running sweep only",
                    error=str(_e),
                )
                _scheduler = None

        def _run_retention() -> None:
            import time as _time

            from clorag.web.auth import LOGIN_LOCKOUT_DURATION

            db = AnalyticsDatabase(settings.analytics_database_path)
            anonymized = db.anonymize_old_searches(
                settings.analytics_anonymize_after_days
            )
            purged = db.purge_old_searches(settings.analytics_retention_days)
            db.purge_login_state(_time.time(), LOGIN_LOCKOUT_DURATION)
            logger.info(
                "Analytics retention sweep",
                anonymized=anonymized,
                purged=purged,
            )

        if _scheduler is not None and _scheduler.running:
            _scheduler.add_job(
                _run_retention,
                "interval",
                hours=settings.analytics_cleanup_interval_hours,
                id="analytics_retention",
                replace_existing=True,
                next_run_time=None,
            )
        # Run once at startup so restarts don't leave stale rows indefinitely.
        # Offloaded to a worker thread so a large analytics DB (where the
        # UPDATE/DELETE can take seconds) can't block the lifespan past the
        # docker-compose healthcheck timeout and crashloop the container.
        await asyncio.to_thread(_run_retention)
        logger.info(
            "Analytics retention scheduler started",
            interval_hours=settings.analytics_cleanup_interval_hours,
            retention_days=settings.analytics_retention_days,
            anonymize_after_days=settings.analytics_anonymize_after_days,
        )

    # Initialize ingestion runner (marks stale jobs, cleans old entries)
    from clorag.services.ingestion_runner import get_ingestion_runner, ingestion_log_capture

    ingestion_runner = get_ingestion_runner()
    await ingestion_runner.startup()

    # Inject structlog processor for per-job log capture
    _inject_structlog_processor(ingestion_log_capture)
    logger.info("Ingestion runner initialized")

    yield

    # Shutdown ingestion runner (cancel running tasks)
    await ingestion_runner.shutdown()

    if _scheduler:
        try:
            if _scheduler.running:
                _scheduler.shutdown()
                logger.info("Scheduler stopped")
        except Exception as e:
            logger.warning("Scheduler shutdown failed", error=str(e))


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Cyanview AI Search",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


# Global exception handler to prevent stack trace exposure in production
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions without exposing stack traces.

    Logs the full error for debugging while returning a sanitized
    error message to the client.
    """
    # Log full error details for debugging
    logger.error(
        "unhandled_exception",
        path=str(request.url.path),
        method=request.method,
        error_type=type(exc).__name__,
        error_message=str(exc),
        exc_info=True,
    )

    # Return generic error to client - no stack trace exposure
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


app.add_exception_handler(Exception, generic_exception_handler)

# Add CORS middleware.
# SECURITY: DynamicCORSMiddleware reads the allowlist from settings on every
# request so rotating CORS_ALLOWED_ORIGINS doesn't require restarting the
# container (admins can trigger get_settings.cache_clear() via reload).
app.add_middleware(DynamicCORSMiddleware)

# Add timeout middleware
app.add_middleware(TimeoutMiddleware)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Trust X-Forwarded-* headers from the reverse proxy so that
# slowapi.get_remote_address (and any other request.client.host consumer)
# sees the real client IP, not the Docker bridge gateway. Without this,
# every rate limit — including the admin brute-force lockout — applies
# globally across all users sharing the same proxy hop, which is a
# production-breaking misconfiguration rather than a hardening detail.
#
# trusted_hosts="*" is safe here because the container port is bound to
# 127.0.0.1 in docker-compose (only the co-located Caddy reverse proxy
# can open a TCP connection to the app). If that assumption changes,
# narrow this to the reverse proxy's actual source IPs.
# Added LAST so it becomes the OUTERMOST middleware (Starlette wraps in
# reverse, so the last add_middleware call runs first on the request and
# gets to rewrite request.client before any other middleware sees it).
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

# Static files and templates
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include routers
app.include_router(admin_router)
app.include_router(public_router)


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app
