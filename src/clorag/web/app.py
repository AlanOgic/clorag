"""FastAPI web application for AI Search with Claude synthesis."""

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import anyio
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import (
    FastAPI,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from clorag.config import get_settings
from clorag.core.analytics_db import AnalyticsDatabase
from clorag.services.custom_docs import CustomDocumentService

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


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeout."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with 60 second timeout.

        SSE streaming endpoints are exempt from timeout.
        """
        # Skip timeout for SSE log streaming endpoints
        if "/logs/stream" in request.url.path:
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
            "max-age=31536000; includeSubDomains; preload"
        )
        # Content Security Policy with nonce for inline scripts
        # SECURITY: Uses per-request nonce for script execution (CSP Level 2+).
        # All templates use nonce="{{ request.state.csp_nonce }}" for inline scripts.
        # 'unsafe-inline' for style-src is needed for inline styles and libraries like Mermaid.
        #
        # NOTE: Admin pages use 'unsafe-hashes' to allow inline event handlers (onclick, etc.)
        # while templates are being migrated to use data-action attributes with event delegation.
        # Public pages use strict nonce-only policy.
        is_admin_page = request.url.path.startswith("/admin")
        if is_admin_page:
            # Admin pages: Allow inline event handlers via 'unsafe-inline'
            # NOTE: When a nonce is present, browsers ignore 'unsafe-inline' per CSP Level 2+.
            # So we omit the nonce for admin pages to allow onclick handlers to work.
            # This is a temporary measure while templates are migrated to data-action pattern.
            script_src = "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://esm.sh"
        else:
            # Public pages: Strict nonce-only policy (esm.sh for Excalidraw modules)
            script_src = f"script-src 'self' 'nonce-{csp_nonce}' https://cdn.jsdelivr.net https://esm.sh"

        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"{script_src}; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://esm.sh https://api.fontshare.com; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://cdn.jsdelivr.net https://esm.sh https://cdn.fontshare.com; "
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
async def lifespan(app: FastAPI):
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
        _scheduler.shutdown()
        logger.info("Draft creation scheduler stopped")


# Initialize FastAPI app with lifespan
app = FastAPI(title="Cyanview AI Search", version="1.0.0", lifespan=lifespan)

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

# Add CORS middleware
# SECURITY: Explicitly list allowed headers instead of wildcard to prevent header exposure
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cyanview.cloud"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Content-Type",
        "X-Admin-Password",
        "X-CSRF-Token",
        "X-Requested-With",
    ],
)

# Add timeout middleware
app.add_middleware(TimeoutMiddleware)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Static files and templates
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include routers
app.include_router(admin_router)
app.include_router(public_router)

# Initialize clients (lazy loading)
_analytics_db: AnalyticsDatabase | None = None
_custom_docs_service: CustomDocumentService | None = None


def get_analytics_db() -> AnalyticsDatabase:
    """Get or create AnalyticsDatabase instance (separate from camera DB)."""
    global _analytics_db
    if _analytics_db is None:
        settings = get_settings()
        _analytics_db = AnalyticsDatabase(settings.analytics_database_path)
    return _analytics_db


def get_custom_docs_service() -> CustomDocumentService:
    """Get or create CustomDocumentService instance."""
    global _custom_docs_service
    if _custom_docs_service is None:
        _custom_docs_service = CustomDocumentService()
    return _custom_docs_service


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app
