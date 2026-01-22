"""FastAPI web application for AI Search with Claude synthesis."""

import io
import json
import secrets
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    pass

import anyio
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import (
    Body,
    Cookie,
    Depends,
    FastAPI,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from clorag.config import get_settings
from clorag.core.analytics_db import AnalyticsDatabase
from clorag.core.database import get_camera_database
from clorag.core.metrics import get_metrics_collector
from clorag.models.camera import Camera, CameraCreate, CameraSource, CameraUpdate
from clorag.models.custom_document import (
    CustomDocument,
    CustomDocumentCreate,
    CustomDocumentUpdate,
    DocumentCategory,
)
from clorag.services.custom_docs import CustomDocumentService
from clorag.services.prompt_manager import get_prompt

# Import auth module
from clorag.web.auth import (
    ADMIN_SESSION_COOKIE,
    ADMIN_SESSION_MAX_AGE,
    LOGIN_LOCKOUT_DURATION,
    generate_csrf_token,
    get_login_tracker,
    get_session_serializer,
    get_session_store,
    verify_admin,
)

# Import schemas
from clorag.web.schemas import (
    ChunkCollection,
    ChunkDetail,
    ChunkListItem,
    ChunkListResponse,
    ChunkUpdate,
    DebugSearchResponse,
    KnowledgeListResponse,
    LoginRequest,
    LoginResponse,
    PromptCreateFromDefaultRequest,
    PromptRollbackRequest,
    PromptTestRequest,
    PromptUpdateRequest,
    RelationshipDeleteRequest,
    RelationshipUpdateRequest,
    SearchRequest,
    SearchResponse,
    TerminologyBatchStatusUpdate,
    TerminologyStatusUpdate,
)

# Import search module
from clorag.web.search import (
    extract_source_links,
    get_anthropic,
    get_embeddings,
    get_sparse_embeddings,
    get_vectorstore,
    perform_search,
    synthesize_answer,
    synthesize_answer_stream,
)

# Initialize logger
logger = structlog.get_logger()


# =============================================================================
# Middleware Configuration
# =============================================================================


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeout."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with 60 second timeout."""
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
        # All 28 templates have been updated to use nonce="{{ request.state.csp_nonce }}".
        # 'unsafe-inline' for style-src is needed for inline styles and libraries like Mermaid.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'nonce-{csp_nonce}' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://cdn.jsdelivr.net; "
            "connect-src 'self'; "
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
    try:
        vs = get_vectorstore()
        await vs.ensure_collections(hybrid=True)
        logger.info("Qdrant collections ensured")
    except Exception as e:
        logger.warning("Failed to ensure Qdrant collections", error=str(e))

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

    yield

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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the search home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/robots.txt")
async def robots_txt():
    """Block search engine crawlers from indexing."""
    return Response(
        content="User-agent: *\nDisallow: /\n",
        media_type="text/plain",
    )


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/search/stream")
@limiter.limit("30/minute")
async def search_stream(request: Request, req: SearchRequest):
    """Perform RAG search with streaming Claude-powered answer synthesis.

    Uses hybrid search (dense + sparse vectors) with RRF fusion for better
    results on queries with specific model numbers or technical terms.

    Supports follow-up conversations via session_id. If provided, the response
    will include conversation history context for more relevant answers.

    Rate limited to 30 requests per minute per IP to prevent abuse.
    """
    start_time = time.time()

    # Get or create conversation session
    session = get_session_store().get_or_create_session(req.session_id)
    conversation_history = session.get_context_messages()

    # Use helper to perform search
    _, chunks_for_synthesis, graph_context, was_reranked = await perform_search(req)

    # Extract top 3 unique source links
    source_links = extract_source_links(chunks_for_synthesis)

    # Capture start time for analytics
    search_start_time = start_time

    async def generate():
        collected_response = []
        try:
            # Send session_id first so frontend can track the conversation
            yield f"data: {json.dumps({'type': 'session', 'session_id': session.session_id})}\n\n"

            # Stream the answer with conversation history and graph context
            async for chunk in synthesize_answer_stream(
                req.query, chunks_for_synthesis, conversation_history, graph_context
            ):
                collected_response.append(chunk)
                yield f"data: {json.dumps({'type': 'text', 'text': chunk})}\n\n"

            # Then send sources at the end
            yield f"data: {json.dumps({'type': 'sources', 'sources': source_links})}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            # Store this exchange in the session for future follow-ups
            full_response = "".join(collected_response)
            session.add_exchange(req.query, full_response)

            # Log search to analytics with full data after streaming completes
            response_time_ms = int((time.time() - search_start_time) * 1000)
            try:
                get_analytics_db().log_search(
                    query=req.query,
                    source=req.source.value,
                    response_time_ms=response_time_ms,
                    results_count=len(chunks_for_synthesis),
                    response=full_response,
                    chunks=chunks_for_synthesis,
                    session_id=session.session_id,
                    reranked=was_reranked,
                )
            except Exception as e:
                logger.warning("Failed to log search analytics", error=str(e))
        except Exception as e:
            logger.error("Error during streaming response", error=str(e), query=req.query)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/search", response_model=SearchResponse)
@limiter.limit("30/minute")
async def search(request: Request, req: SearchRequest):
    """Perform RAG search with Claude-powered answer synthesis.

    Uses hybrid search (dense + sparse vectors) with RRF fusion for better
    results on queries with specific model numbers or technical terms.

    Supports follow-up conversations via session_id.

    Rate limited to 30 requests per minute per IP to prevent abuse.
    """
    start_time = time.time()

    # Get or create conversation session
    session = get_session_store().get_or_create_session(req.session_id)
    conversation_history = session.get_context_messages()

    # Use helper to perform search
    results, chunks_for_synthesis, graph_context, was_reranked = await perform_search(req)

    # Generate synthesized answer using Claude with conversation history and graph context
    answer = await synthesize_answer(
        req.query, chunks_for_synthesis, conversation_history, graph_context
    )

    # Store this exchange in the session for future follow-ups
    session.add_exchange(req.query, answer)

    # Extract top 3 unique source links
    source_links = extract_source_links(chunks_for_synthesis, as_model=True)

    # Log search to analytics with full data (non-blocking)
    response_time_ms = int((time.time() - start_time) * 1000)
    try:
        get_analytics_db().log_search(
            query=req.query,
            source=req.source.value,
            response_time_ms=response_time_ms,
            results_count=len(results),
            response=answer,
            chunks=chunks_for_synthesis,
            session_id=session.session_id,
            reranked=was_reranked,
        )
    except Exception as e:
        logger.warning("Failed to log search analytics", error=str(e))

    return SearchResponse(
        query=req.query,
        source=req.source.value,
        answer=answer,
        source_links=source_links,
        results=results,
        total=len(results),
        session_id=session.session_id,
    )

# =============================================================================
# Camera Compatibility Routes
# =============================================================================


@app.get("/cameras", response_class=HTMLResponse)
async def cameras_list(
    request: Request,
    manufacturer: str | None = None,
    device_type: str | None = None,
    port: str | None = None,
    protocol: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Render the public camera compatibility list with pagination."""
    db = get_camera_database()

    # Clamp page_size to reasonable limits
    page_size = max(10, min(100, page_size))
    page = max(1, page)

    # Get total count for pagination
    total_count = db.count_cameras(
        manufacturer=manufacturer,
        device_type=device_type,
        port=port,
        protocol=protocol,
    )
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    # Clamp page to valid range
    page = min(page, total_pages)
    offset = (page - 1) * page_size

    cameras = db.list_cameras(
        manufacturer=manufacturer,
        device_type=device_type,
        port=port,
        protocol=protocol,
        offset=offset,
        limit=page_size,
    )
    manufacturers = db.get_manufacturers()
    device_types = db.get_device_types()
    ports = db.get_all_ports()
    protocols = db.get_all_protocols()
    stats = db.get_stats()

    return templates.TemplateResponse(
        "cameras.html",
        {
            "request": request,
            "cameras": cameras,
            "manufacturers": manufacturers,
            "device_types": device_types,
            "ports": ports,
            "protocols": protocols,
            "selected_manufacturer": manufacturer,
            "selected_device_type": device_type,
            "selected_port": port,
            "selected_protocol": protocol,
            "stats": stats,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    )


@app.get("/api/cameras", response_model=list[Camera])
@limiter.limit("60/minute")
async def api_cameras_list(
    request: Request,
    manufacturer: str | None = None,
    device_type: str | None = None,
    port: str | None = None,
    protocol: str | None = None,
    page: int | None = None,
    page_size: int = 50,
):
    """Get cameras as JSON with optional pagination.

    If page is not specified, returns all cameras.
    If page is specified, returns paginated results.
    """
    db = get_camera_database()

    if page is not None:
        # Paginated request
        page_size = max(10, min(100, page_size))
        page = max(1, page)
        offset = (page - 1) * page_size
        return db.list_cameras(
            manufacturer=manufacturer,
            device_type=device_type,
            port=port,
            protocol=protocol,
            offset=offset,
            limit=page_size,
        )

    # Non-paginated (all cameras)
    return db.list_cameras(
        manufacturer=manufacturer,
        device_type=device_type,
        port=port,
        protocol=protocol,
    )


@app.get("/api/cameras/search", response_model=list[Camera])
@limiter.limit("60/minute")
async def api_cameras_search(request: Request, q: str):
    """Search cameras by name or manufacturer."""
    db = get_camera_database()
    return db.search_cameras(q)


@app.get("/api/cameras/stats")
@limiter.limit("60/minute")
async def api_cameras_stats(request: Request):
    """Get camera database statistics."""
    db = get_camera_database()
    return db.get_stats()


@app.get("/api/cameras/{camera_id}", response_model=Camera)
@limiter.limit("120/minute")
async def api_camera_get(request: Request, camera_id: int):
    """Get a single camera by ID."""
    db = get_camera_database()
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@app.get("/api/cameras/{camera_id}/related", response_model=list[Camera], tags=["Cameras"])
@limiter.limit("60/minute")
async def api_camera_related(request: Request, camera_id: int, limit: int = 5):
    """Get cameras related to the specified camera.

    Finds cameras with similar manufacturer, device type, ports, or protocols.
    """
    db = get_camera_database()
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return db.find_related_cameras(camera_id, limit=min(limit, 10))


@app.post("/api/cameras/compare", response_model=list[Camera], tags=["Cameras"])
@limiter.limit("60/minute")
async def api_cameras_compare(request: Request, camera_ids: list[int]):
    """Get multiple cameras for comparison.

    Accepts a list of camera IDs and returns the camera objects in order.
    Maximum 5 cameras can be compared at once.
    """
    if not camera_ids:
        raise HTTPException(status_code=400, detail="No camera IDs provided")
    if len(camera_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 cameras can be compared")

    db = get_camera_database()
    cameras = db.get_cameras_by_ids(camera_ids)
    if not cameras:
        raise HTTPException(status_code=404, detail="No cameras found")
    return cameras


@app.get("/api/cameras/export.csv", tags=["Cameras"])
@limiter.limit("10/minute")
async def api_cameras_export_csv(
    request: Request,
    manufacturer: str | None = None,
    device_type: str | None = None,
):
    """Export cameras to CSV format.

    Optionally filter by manufacturer or device type.
    """
    import csv
    import io

    db = get_camera_database()
    cameras = db.list_cameras(manufacturer=manufacturer, device_type=device_type)

    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "ID", "Name", "Manufacturer", "Code Model", "Device Type",
        "Ports", "Protocols", "Supported Controls", "Notes",
        "Doc URL", "Manufacturer URL", "Source", "Confidence"
    ])

    # Data rows
    for cam in cameras:
        writer.writerow([
            cam.id,
            cam.name,
            cam.manufacturer or "",
            cam.code_model or "",
            cam.device_type.value if cam.device_type else "",
            "|".join(cam.ports),
            "|".join(cam.protocols),
            "|".join(cam.supported_controls),
            "|".join(cam.notes),
            cam.doc_url or "",
            cam.manufacturer_url or "",
            cam.source.value if cam.source else "",
            cam.confidence,
        ])

    # Return as downloadable CSV
    from starlette.responses import Response
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cameras.csv"}
    )


# Admin routes (protected)


@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    """Public user guide page."""
    return templates.TemplateResponse("help.html", {"request": request})


@app.get("/video", response_class=HTMLResponse)
async def video_page(request: Request):
    """Public video showcase page."""
    return templates.TemplateResponse("video.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_index(request: Request):
    """Admin index page with links to all admin pages."""
    return templates.TemplateResponse("admin_index.html", {"request": request})


@app.get("/admin/docs", response_class=HTMLResponse)
async def admin_docs_index(request: Request):
    """Admin technical documentation - index page."""
    return templates.TemplateResponse("docs/index.html", {"request": request})


@app.get("/admin/docs/{page}", response_class=HTMLResponse)
async def admin_docs_page(request: Request, page: str):
    """Admin technical documentation - specific page."""
    # Validate page name to prevent path traversal
    if not page.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(status_code=404, detail="Page not found")
    return templates.TemplateResponse(f"docs/{page}.html", {"request": request})


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Admin login page."""
    return templates.TemplateResponse("admin_login.html", {"request": request})


@app.post("/api/admin/login", tags=["Authentication"])
@limiter.limit("10/minute")
async def api_admin_login(
    request: Request,
    login_req: LoginRequest,
    response: Response,
):
    """Login and set session cookie.

    Returns success status and sets httponly cookie on success.
    Implements brute force protection with lockout after 5 failed attempts.
    """
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")

    # Get client IP for brute force tracking
    client_ip = get_remote_address(request)
    tracker = get_login_tracker()

    # Check if IP is locked out
    if tracker.is_locked_out(client_ip):
        remaining = tracker.get_lockout_remaining(client_ip)
        logger.warning("Login attempt from locked out IP", ip=client_ip, remaining=remaining)
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {remaining} seconds."
        )

    # Verify password with timing-safe comparison
    if not secrets.compare_digest(
        login_req.password.encode('utf-8'),
        settings.admin_password.get_secret_value().encode('utf-8')
    ):
        # Record failed attempt
        locked_out = tracker.record_failed_attempt(client_ip)
        logger.warning("Failed login attempt", ip=client_ip, locked_out=locked_out)
        if locked_out:
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Try again in {LOGIN_LOCKOUT_DURATION} seconds."
            )
        raise HTTPException(status_code=401, detail="Invalid password")

    # Successful login - clear any failed attempts
    tracker.clear_attempts(client_ip)

    # Create signed session token with unique session_id for CSRF binding
    serializer = get_session_serializer()
    session_id = secrets.token_hex(16)  # Generate unique session identifier
    token = serializer.dumps({
        "authenticated": True,
        "ts": time.time(),
        "session_id": session_id,  # SECURITY: Used for CSRF token binding validation
    })

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


@app.post("/api/admin/logout", tags=["Authentication"])
async def api_admin_logout(response: Response):
    """Logout and clear session cookie."""
    settings = get_settings()
    response.delete_cookie(
        key=ADMIN_SESSION_COOKIE,
        httponly=True,
        secure=settings.secure_cookies,
        samesite="strict",
    )
    return {"success": True, "message": "Logged out"}


@app.get("/api/admin/session", tags=["Authentication"])
async def api_admin_session(
    admin_session: Annotated[str | None, Cookie()] = None,
):
    """Check if current session is valid.

    Returns authenticated status without requiring password.
    """
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


@app.get("/api/admin/csrf-token", tags=["Authentication"])
async def api_admin_csrf_token(
    admin_session: Annotated[str | None, Cookie()] = None,
):
    """Get a CSRF token for state-changing requests.

    Returns a signed CSRF token that must be included in the
    X-CSRF-Token header for POST, PUT, DELETE requests.

    The token is valid for 1 hour and should be refreshed periodically.
    """
    # Extract session ID for binding if authenticated
    session_id = None
    if admin_session:
        try:
            serializer = get_session_serializer()
            data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
            session_id = data.get("session_id")
        except (SignatureExpired, BadSignature):
            pass  # Generate unbound token for unauthenticated requests

    token = generate_csrf_token(session_id)
    return {"csrf_token": token}


@app.get("/api/admin/backup", tags=["Backup"])
async def api_admin_backup(
    request: Request,
    admin_session: Annotated[str | None, Cookie()] = None,
    x_admin_password: Annotated[str | None, Header()] = None,
):
    """Download a backup ZIP of all SQLite databases.

    Requires admin authentication via session cookie or X-Admin-Password header.
    Returns a ZIP file containing:
    - clorag.db (camera database)
    - analytics.db (search analytics)
    """
    # Verify authentication (session or header)
    authenticated = False

    # Check session cookie first
    if admin_session:
        try:
            serializer = get_session_serializer()
            data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
            if data.get("authenticated"):
                authenticated = True
        except (SignatureExpired, BadSignature):
            pass

    # Fall back to header auth
    if not authenticated and x_admin_password:
        settings = get_settings()
        if settings.admin_password and secrets.compare_digest(
            x_admin_password.encode('utf-8'),
            settings.admin_password.get_secret_value().encode('utf-8')
        ):
            authenticated = True

    if not authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    settings = get_settings()
    camera_db_path = Path(settings.database_path)
    analytics_db_path = Path(settings.analytics_database_path)

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add camera database
        if camera_db_path.exists():
            zf.write(camera_db_path, "clorag.db")
            logger.info("Added camera database to backup", path=str(camera_db_path))

        # Add analytics database
        if analytics_db_path.exists():
            zf.write(analytics_db_path, "analytics.db")
            logger.info("Added analytics database to backup", path=str(analytics_db_path))

    zip_buffer.seek(0)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clorag_backup_{timestamp}.zip"

    logger.info("Database backup created", filename=filename)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/admin/cameras", response_class=HTMLResponse)
async def admin_cameras_list(
    request: Request,
    manufacturer: str | None = None,
):
    """Admin camera list with edit capabilities.

    Note: HTML pages are accessible without auth - password is prompted
    via JavaScript and used for API calls (which are protected).
    """
    db = get_camera_database()
    cameras = db.list_cameras(manufacturer=manufacturer)
    manufacturers = db.get_manufacturers()
    stats = db.get_stats()

    return templates.TemplateResponse(
        "admin_cameras.html",
        {
            "request": request,
            "cameras": cameras,
            "manufacturers": manufacturers,
            "selected_manufacturer": manufacturer,
            "stats": stats,
        },
    )


@app.get("/admin/cameras/{camera_id}/edit", response_class=HTMLResponse)
async def admin_camera_edit(
    request: Request,
    camera_id: int,
):
    """Camera edit form."""
    db = get_camera_database()
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return templates.TemplateResponse(
        "camera_edit.html",
        {
            "request": request,
            "camera": camera,
        },
    )


@app.get("/admin/cameras/new", response_class=HTMLResponse)
async def admin_camera_new(
    request: Request,
):
    """New camera form."""
    return templates.TemplateResponse(
        "camera_edit.html",
        {
            "request": request,
            "camera": None,
        },
    )


@app.post("/api/admin/cameras", response_model=Camera, tags=["Cameras"])
@limiter.limit("5/minute")
async def api_camera_create(
    request: Request,
    camera: Annotated[CameraCreate, Body()],
    _: bool = Depends(verify_admin),
):
    """Create a new camera entry."""
    db = get_camera_database()
    return db.create_camera(camera, CameraSource.MANUAL)


@app.put("/api/admin/cameras/{camera_id}", response_model=Camera, tags=["Cameras"])
@limiter.limit("5/minute")
async def api_camera_update(
    camera_id: int,
    request: Request,
    updates: Annotated[CameraUpdate, Body()],
    _: bool = Depends(verify_admin),
):
    """Update an existing camera."""
    db = get_camera_database()
    camera = db.update_camera(camera_id, updates)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@app.delete("/api/admin/cameras/{camera_id}", tags=["Cameras"])
@limiter.limit("5/minute")
async def api_camera_delete(
    camera_id: int,
    request: Request,
    _: bool = Depends(verify_admin),
):
    """Delete a camera entry."""
    db = get_camera_database()
    if not db.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"status": "deleted", "id": camera_id}


# =============================================================================
# Camera Review Queue Routes (Admin)
# =============================================================================


@app.get("/admin/cameras/review", response_class=HTMLResponse)
async def admin_cameras_review(
    request: Request,
    page: int = 1,
    page_size: int = 25,
):
    """Admin page for reviewing low-confidence camera extractions."""
    db = get_camera_database()

    # Get total count for pagination
    total_count = db.count_cameras_needing_review()
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    # Clamp page to valid range
    page = max(1, min(page, total_pages))
    offset = (page - 1) * page_size

    cameras = db.list_cameras_needing_review(offset=offset, limit=page_size)

    return templates.TemplateResponse(
        "admin_cameras_review.html",
        {
            "request": request,
            "cameras": cameras,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    )


@app.get("/api/admin/cameras/review", response_model=list[Camera], tags=["Cameras"])
async def api_cameras_needing_review(
    request: Request,
    page: int = 1,
    page_size: int = 25,
    _: bool = Depends(verify_admin),
):
    """Get cameras needing review."""
    db = get_camera_database()
    page_size = max(10, min(100, page_size))
    page = max(1, page)
    offset = (page - 1) * page_size
    return db.list_cameras_needing_review(offset=offset, limit=page_size)


@app.get("/api/admin/cameras/review/count", tags=["Cameras"])
async def api_cameras_review_count(
    request: Request,
    _: bool = Depends(verify_admin),
):
    """Get count of cameras needing review."""
    db = get_camera_database()
    return {"count": db.count_cameras_needing_review()}


@app.post("/api/admin/cameras/{camera_id}/approve", response_model=Camera, tags=["Cameras"])
@limiter.limit("30/minute")
async def api_camera_approve(
    camera_id: int,
    request: Request,
    _: bool = Depends(verify_admin),
):
    """Approve a camera (clear needs_review flag)."""
    db = get_camera_database()
    camera = db.approve_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@app.post("/api/admin/cameras/import", tags=["Cameras"])
@limiter.limit("5/minute")
async def api_cameras_import_csv(
    request: Request,
    file: UploadFile,
    _: bool = Depends(verify_admin),
):
    """Import cameras from CSV file.

    CSV format should have headers:
    Name, Manufacturer, Code Model, Device Type, Ports, Protocols, Supported Controls, Notes

    Ports, Protocols, Controls, and Notes should be pipe-separated (|).
    Existing cameras with same name will be updated (merged).
    """
    import csv
    import io

    from clorag.models.camera import CameraCreate, CameraSource, DeviceType

    if not file.filename or not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    content = await file.read()
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        text = content.decode('latin-1')

    reader = csv.DictReader(io.StringIO(text))

    db = get_camera_database()
    imported = 0
    updated = 0
    errors = []

    for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
        try:
            name = row.get('Name', '').strip()
            if not name:
                errors.append(f"Row {row_num}: Missing name")
                continue

            # Parse fields
            manufacturer = row.get('Manufacturer', '').strip() or None
            code_model = row.get('Code Model', '').strip() or None

            # Parse device type
            device_type_str = row.get('Device Type', '').strip()
            device_type = None
            if device_type_str:
                try:
                    device_type = DeviceType(device_type_str)
                except ValueError:
                    pass  # Invalid device type, leave as None

            # Parse pipe-separated lists
            ports = [p.strip() for p in row.get('Ports', '').split('|') if p.strip()]
            protocols = [p.strip() for p in row.get('Protocols', '').split('|') if p.strip()]
            controls = [c.strip() for c in row.get('Supported Controls', '').split('|') if c.strip()]
            notes = [n.strip() for n in row.get('Notes', '').split('|') if n.strip()]

            # URLs
            doc_url = row.get('Doc URL', '').strip() or None
            manufacturer_url = row.get('Manufacturer URL', '').strip() or None

            camera_data = CameraCreate(
                name=name,
                manufacturer=manufacturer,
                code_model=code_model,
                device_type=device_type,
                ports=ports,
                protocols=protocols,
                supported_controls=controls,
                notes=notes,
                doc_url=doc_url,
                manufacturer_url=manufacturer_url,
            )

            # Check if camera exists
            existing = db.get_camera_by_name(name)
            if existing:
                db.upsert_camera(camera_data, CameraSource.MANUAL)
                updated += 1
            else:
                db.create_camera(camera_data, CameraSource.MANUAL)
                imported += 1

        except Exception as e:
            errors.append(f"Row {row_num}: {str(e)}")

    return {
        "imported": imported,
        "updated": updated,
        "errors": errors[:10],  # Return first 10 errors
        "total_errors": len(errors),
    }


# =============================================================================
# Search Analytics Routes (Admin)
# =============================================================================


@app.get("/admin/analytics", response_class=HTMLResponse)
async def admin_analytics(request: Request):
    """Admin analytics dashboard page."""
    return templates.TemplateResponse("admin_analytics.html", {"request": request})


@app.get("/api/admin/search-stats", tags=["Analytics"])
async def api_search_stats(
    days: int = 30,
    _: bool = Depends(verify_admin),
):
    """Get search analytics statistics."""
    analytics = get_analytics_db()
    return {
        "stats": analytics.get_search_stats(days=days),
        "popular_queries": analytics.get_popular_queries(limit=10, days=days),
        "recent_searches": analytics.get_recent_searches(limit=20),
    }


@app.get("/api/admin/search-stats/popular", tags=["Analytics"])
async def api_popular_queries(
    limit: int = 10,
    days: int = 30,
    _: bool = Depends(verify_admin),
):
    """Get popular search queries."""
    analytics = get_analytics_db()
    return analytics.get_popular_queries(limit=limit, days=days)


@app.get("/api/admin/cache-stats", tags=["Performance"])
async def api_cache_stats(_: bool = Depends(verify_admin)):
    """Get embedding cache statistics for performance monitoring.

    Returns hit/miss rates for both dense (Voyage AI) and sparse (BM25)
    query embedding caches. Higher hit rates indicate better cache efficiency.
    """
    from clorag.core.embeddings import get_query_cache

    sparse_emb = get_sparse_embeddings()
    dense_cache = get_query_cache()

    return {
        "dense_cache": dense_cache.stats(),
        "sparse_cache": sparse_emb.cache_stats(),
        "recommendations": _generate_cache_recommendations(
            dense_cache.stats(), sparse_emb.cache_stats()
        ),
    }


@app.get("/api/admin/metrics", tags=["Performance"])
async def api_performance_metrics(_: bool = Depends(verify_admin)):
    """Get comprehensive performance metrics for the RAG pipeline.

    Returns timing statistics for embedding generation, vector search,
    and total search latency with percentiles (p50, p90, p95, p99).

    Metrics are collected from a sliding window of the last 1000 operations.
    """
    metrics = get_metrics_collector()
    all_stats = metrics.get_all_stats()

    # Add target thresholds for comparison
    thresholds = {
        "embedding_generation": {"target_ms": 200, "warning_ms": 500},
        "vector_search": {"target_ms": 100, "warning_ms": 300},
        "total_search": {"target_ms": 500, "warning_ms": 1000},
        "llm_synthesis": {"target_ms": 2000, "warning_ms": 5000},
    }

    # Generate performance alerts
    alerts = []
    for metric_name, stats in all_stats.get("metrics", {}).items():
        if metric_name in thresholds:
            threshold = thresholds[metric_name]
            if stats.get("p95_ms", 0) > threshold["warning_ms"]:
                alerts.append({
                    "level": "warning",
                    "metric": metric_name,
                    "message": f"{metric_name} p95 ({stats['p95_ms']}ms) exceeds warning threshold ({threshold['warning_ms']}ms)",
                })
            elif stats.get("p95_ms", 0) > threshold["target_ms"]:
                alerts.append({
                    "level": "info",
                    "metric": metric_name,
                    "message": f"{metric_name} p95 ({stats['p95_ms']}ms) exceeds target ({threshold['target_ms']}ms)",
                })

    return {
        **all_stats,
        "thresholds": thresholds,
        "alerts": alerts,
    }


@app.get("/api/admin/metrics/recent/{metric_name}", tags=["Performance"])
async def api_recent_metrics(
    metric_name: str,
    count: int = Query(default=10, ge=1, le=100),
    _: bool = Depends(verify_admin),
):
    """Get recent measurements for a specific metric.

    Useful for debugging recent performance issues or viewing trends.
    """
    metrics = get_metrics_collector()
    recent = metrics.get_recent(metric_name, count)

    if not recent:
        raise HTTPException(status_code=404, detail=f"No data for metric: {metric_name}")

    return {
        "metric": metric_name,
        "count": len(recent),
        "measurements": recent,
    }


def _generate_cache_recommendations(
    dense_stats: dict[str, int],
    sparse_stats: dict[str, int],
) -> list[str]:
    """Generate actionable recommendations based on cache performance."""
    recommendations = []

    # Check dense cache hit rate
    if dense_stats.get("hit_rate_percent", 0) < 30:
        recommendations.append(
            "Dense cache hit rate is low (<30%). Consider increasing cache size "
            "or pre-warming with common queries."
        )

    # Check sparse cache hit rate
    if sparse_stats.get("hit_rate_percent", 0) < 30:
        recommendations.append(
            "Sparse cache hit rate is low (<30%). Users may be asking diverse queries."
        )

    # Check if caches are full
    if dense_stats.get("size", 0) >= 190:  # Near 200 limit
        recommendations.append(
            "Dense cache is near capacity. Consider increasing QUERY_CACHE_MAX_SIZE."
        )

    if sparse_stats.get("size", 0) >= 190:
        recommendations.append(
            "Sparse cache is near capacity. Consider increasing SPARSE_CACHE_MAX_SIZE."
        )

    if not recommendations:
        recommendations.append("Cache performance is healthy.")

    return recommendations


@app.get("/api/admin/search/{search_id}", tags=["Analytics"])
async def api_get_search(
    search_id: int,
    _: bool = Depends(verify_admin),
):
    """Get a stored search by ID with full response and chunks data."""
    analytics = get_analytics_db()
    search = analytics.get_search_by_id(search_id)
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")
    return search


@app.get("/api/admin/conversations", tags=["Analytics"])
async def api_get_conversations(
    limit: int = 20,
    _: bool = Depends(verify_admin),
):
    """Get recent conversations grouped by session_id."""
    analytics = get_analytics_db()
    return analytics.get_recent_conversations(limit=limit)


# =============================================================================
# Support Cases Routes (Admin)
# =============================================================================


@app.get("/api/admin/support-cases", tags=["Support Cases"])
async def api_list_support_cases(
    limit: int = 50,
    offset: int = 0,
    category: str | None = None,
    product: str | None = None,
    _: bool = Depends(verify_admin),
):
    """List support cases with optional filtering."""
    from dataclasses import asdict

    from clorag.core.support_case_db import get_support_case_database
    db = get_support_case_database()
    cases, total = db.list_cases(
        category=category,
        product=product,
        limit=limit,
        offset=offset,
    )
    return {
        "cases": [asdict(case) for case in cases],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/admin/support-cases/stats", tags=["Support Cases"])
async def api_support_cases_stats(_: bool = Depends(verify_admin)):
    """Get support case statistics."""
    from clorag.core.support_case_db import get_support_case_database
    db = get_support_case_database()
    return db.get_stats()


@app.get("/api/admin/support-cases/search", tags=["Support Cases"])
async def api_search_support_cases(
    q: str,
    limit: int = 20,
    _: bool = Depends(verify_admin),
):
    """Search support cases using full-text search."""
    from dataclasses import asdict

    from clorag.core.support_case_db import get_support_case_database
    db = get_support_case_database()
    cases = db.search_cases(q, limit=limit)
    return {"cases": [asdict(case) for case in cases], "total": len(cases)}


@app.get("/api/admin/support-cases/{case_id}", tags=["Support Cases"])
async def api_get_support_case(
    case_id: str,
    _: bool = Depends(verify_admin),
):
    """Get a support case by ID with full document."""
    from dataclasses import asdict

    from clorag.core.support_case_db import get_support_case_database
    db = get_support_case_database()
    case = db.get_case_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Support case not found")
    return asdict(case)


@app.get("/api/admin/support-cases/{case_id}/raw-thread", tags=["Support Cases"])
async def api_get_support_case_raw_thread(
    case_id: str,
    _: bool = Depends(verify_admin),
):
    """Get the raw anonymized thread content for a case."""
    from clorag.core.support_case_db import get_support_case_database
    db = get_support_case_database()
    raw_thread = db.get_raw_thread(case_id)
    if raw_thread is None:
        raise HTTPException(status_code=404, detail="Raw thread not found")
    return {"raw_thread": raw_thread}


@app.delete("/api/admin/support-cases/{case_id}", tags=["Support Cases"])
async def api_delete_support_case(
    case_id: str,
    _: bool = Depends(verify_admin),
):
    """Delete a support case."""
    from clorag.core.support_case_db import get_support_case_database
    db = get_support_case_database()
    deleted = db.delete_case(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Support case not found")
    return {"deleted": True, "case_id": case_id}


@app.get("/admin/support-cases", response_class=HTMLResponse)
async def admin_support_cases(request: Request):
    """Admin support cases management page."""
    return templates.TemplateResponse("admin_support_cases.html", {"request": request})


# =============================================================================
# Terminology Fixes Routes (Admin)
# =============================================================================


@app.get("/api/admin/terminology-fixes", tags=["Terminology Fixes"])
async def api_list_terminology_fixes(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    collection: str | None = None,
    _: bool = Depends(verify_admin),
):
    """List terminology fixes with optional filtering."""
    from clorag.core.terminology_db import get_terminology_fix_database

    db = get_terminology_fix_database()
    fixes, total = db.list_fixes(
        status=status,  # type: ignore[arg-type]
        collection=collection,
        limit=limit,
        offset=offset,
    )
    return {
        "fixes": [fix.to_dict() for fix in fixes],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/admin/terminology-fixes/stats", tags=["Terminology Fixes"])
async def api_terminology_fixes_stats(_: bool = Depends(verify_admin)):
    """Get terminology fix statistics."""
    from clorag.core.terminology_db import get_terminology_fix_database

    db = get_terminology_fix_database()
    return db.get_stats()


@app.get("/api/admin/terminology-fixes/{fix_id}", tags=["Terminology Fixes"])
async def api_get_terminology_fix(
    fix_id: str,
    _: bool = Depends(verify_admin),
):
    """Get a terminology fix by ID."""
    from clorag.core.terminology_db import get_terminology_fix_database

    db = get_terminology_fix_database()
    fix = db.get_fix(fix_id)
    if not fix:
        raise HTTPException(status_code=404, detail="Terminology fix not found")
    return fix.to_dict()


@app.put("/api/admin/terminology-fixes/{fix_id}/status", tags=["Terminology Fixes"])
async def api_update_terminology_fix_status(
    fix_id: str,
    body: TerminologyStatusUpdate,
    _: bool = Depends(verify_admin),
):
    """Update the status of a terminology fix."""
    from clorag.core.terminology_db import get_terminology_fix_database

    db = get_terminology_fix_database()
    updated = db.update_status(fix_id, body.status)  # type: ignore[arg-type]
    if not updated:
        raise HTTPException(status_code=404, detail="Terminology fix not found")
    return {"updated": True, "fix_id": fix_id, "status": body.status}


@app.put("/api/admin/terminology-fixes/batch-status", tags=["Terminology Fixes"])
async def api_batch_update_terminology_fix_status(
    body: TerminologyBatchStatusUpdate,
    _: bool = Depends(verify_admin),
):
    """Update status for multiple terminology fixes."""
    from clorag.core.terminology_db import get_terminology_fix_database

    db = get_terminology_fix_database()
    count = db.update_statuses_batch(body.ids, body.status)  # type: ignore[arg-type]
    return {"updated": count, "status": body.status}


@app.post("/api/admin/terminology-fixes/apply", tags=["Terminology Fixes"])
async def api_apply_terminology_fixes(_: bool = Depends(verify_admin)):
    """Apply all approved terminology fixes to the vector database.

    Uses document-context re-embedding: when a chunk is fixed, all sibling
    chunks from the same document are re-embedded together using contextualized
    embeddings to preserve semantic understanding of the full document.
    """
    from clorag.core.terminology_db import get_terminology_fix_database
    from clorag.scripts.fix_rio_terminology import apply_approved_fixes

    db = get_terminology_fix_database()
    approved_fixes = db.get_approved_fixes()

    if not approved_fixes:
        return {"applied": 0, "failed": 0, "message": "No approved fixes to apply"}

    vectorstore = get_vectorstore()
    embeddings = get_embeddings()
    sparse_embeddings = get_sparse_embeddings()

    try:
        applied_count = await apply_approved_fixes(
            vectorstore=vectorstore,
            embeddings=embeddings,
            sparse_embeddings=sparse_embeddings,
        )

        # Calculate failed from total - applied
        total_approved = len(approved_fixes)
        failed_count = total_approved - applied_count

        return {
            "applied": applied_count,
            "failed": failed_count,
            "message": f"Applied {applied_count} fixes with document-context re-embedding, {failed_count} failed",
        }
    except Exception as e:
        logger.error("Failed to apply terminology fixes", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to apply terminology fixes: {str(e)}",
        )


@app.post("/api/admin/terminology-fixes/scan", tags=["Terminology Fixes"])
async def api_scan_terminology_fixes(
    max_chunks: int | None = None,
    _: bool = Depends(verify_admin),
):
    """Scan vector database for RIO terminology issues.

    This runs the same scan as `uv run fix-rio-terminology --preview`.
    Suggestions are saved to the database for review.

    Args:
        max_chunks: Maximum chunks to scan (optional, for quick testing).
    """
    from clorag.analysis.rio_analyzer import RIOTerminologyAnalyzer
    from clorag.core.terminology_db import get_terminology_fix_database
    from clorag.scripts.fix_rio_terminology import scan_for_rio_mentions

    db = get_terminology_fix_database()
    vectorstore = get_vectorstore()
    analyzer = RIOTerminologyAnalyzer()

    try:
        # Clear previous pending fixes
        cleared = db.clear_pending()

        # Run scan
        fixes = await scan_for_rio_mentions(vectorstore, analyzer, max_chunks)

        if fixes:
            count = db.insert_fixes_batch(fixes)
        else:
            count = 0

        # Get updated stats
        stats = db.get_stats()

        return {
            "scanned": True,
            "max_chunks": max_chunks,
            "cleared_pending": cleared,
            "fixes_found": count,
            "stats": stats,
        }
    except Exception as e:
        logger.error("Failed to scan for terminology issues", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to scan: {str(e)}",
        )


@app.delete("/api/admin/terminology-fixes/{fix_id}", tags=["Terminology Fixes"])
async def api_delete_terminology_fix(
    fix_id: str,
    _: bool = Depends(verify_admin),
):
    """Delete a terminology fix."""
    from clorag.core.terminology_db import get_terminology_fix_database

    db = get_terminology_fix_database()
    deleted = db.delete_fix(fix_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Terminology fix not found")
    return {"deleted": True, "fix_id": fix_id}


@app.get("/admin/terminology-fixes", response_class=HTMLResponse)
async def admin_terminology_fixes(request: Request):
    """Admin terminology fixes management page."""
    return templates.TemplateResponse("admin_terminology_fixes.html", {"request": request})


# =============================================================================
# Graph Stats Routes (Admin)
# =============================================================================


@app.get("/api/admin/graph/stats", tags=["Graph"])
async def api_graph_stats(_: bool = Depends(verify_admin)):
    """Get knowledge graph statistics.

    Returns node and relationship counts from Neo4j.
    Returns empty stats if graph is not available.
    """
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {
            "available": False,
            "message": "Graph database not configured or unavailable",
            "stats": {},
        }

    try:
        from clorag.core.graph_store import get_graph_store
        store = await get_graph_store()
        stats = await store.get_stats()
        return {
            "available": True,
            "stats": stats,
        }
    except Exception as e:
        logger.warning("graph_stats_failed", error=str(e))
        return {
            "available": False,
            "message": str(e),
            "stats": {},
        }


@app.get("/api/admin/graph/entity-types", tags=["Graph"])
async def api_graph_entity_types(_: bool = Depends(verify_admin)):
    """Get available entity types with counts."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "types": []}

    try:
        from clorag.core.graph_store import get_graph_store
        store = await get_graph_store()
        types = await store.get_entity_type_counts()
        return {"available": True, "types": types}
    except Exception as e:
        logger.warning("graph_entity_types_failed", error=str(e))
        return {"available": False, "types": [], "error": str(e)}


@app.get("/api/admin/graph/relationship-types", tags=["Graph"])
async def api_graph_relationship_types(_: bool = Depends(verify_admin)):
    """Get available relationship types with counts."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "types": []}

    try:
        from clorag.core.graph_store import get_graph_store
        store = await get_graph_store()
        types = await store.get_relationship_type_counts()
        return {"available": True, "types": types}
    except Exception as e:
        logger.warning("graph_relationship_types_failed", error=str(e))
        return {"available": False, "types": [], "error": str(e)}


@app.get("/api/admin/graph/entities", tags=["Graph"])
async def api_graph_entities(
    entity_type: str = Query(..., description="Entity type: Camera, Product, etc."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None),
    _: bool = Depends(verify_admin),
):
    """List entities by type with pagination and optional search."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "entities": [], "total": 0}

    try:
        from clorag.core.graph_store import get_graph_store
        store = await get_graph_store()
        entities, total = await store.list_entities_by_type(
            entity_type=entity_type,
            page=page,
            page_size=page_size,
            search=search,
        )
        return {
            "available": True,
            "entities": entities,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": (page * page_size) < total,
        }
    except Exception as e:
        logger.warning("graph_entities_failed", error=str(e))
        return {"available": False, "entities": [], "total": 0, "error": str(e)}


@app.get("/api/admin/graph/entities/{entity_type}/{entity_id:path}", tags=["Graph"])
async def api_graph_entity_detail(
    entity_type: str,
    entity_id: str,
    _: bool = Depends(verify_admin),
):
    """Get entity details with relationships."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "entity": None}

    try:
        from clorag.core.graph_store import get_graph_store
        store = await get_graph_store()
        entity = await store.get_entity_with_relationships(entity_type, entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        return {"available": True, "entity": entity}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("graph_entity_detail_failed", error=str(e))
        return {"available": False, "entity": None, "error": str(e)}


@app.get("/api/admin/graph/relationships", tags=["Graph"])
async def api_graph_relationships(
    source_type: str | None = Query(None),
    source_name: str | None = Query(None),
    relationship_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    _: bool = Depends(verify_admin),
):
    """List relationships with optional filtering."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "relationships": []}

    try:
        from clorag.core.graph_store import get_graph_store
        store = await get_graph_store()
        relationships = await store.list_relationships(
            source_type=source_type,
            source_name=source_name,
            relationship_type=relationship_type,
            limit=limit,
        )
        return {"available": True, "relationships": relationships}
    except Exception as e:
        logger.warning("graph_relationships_failed", error=str(e))
        return {"available": False, "relationships": [], "error": str(e)}


@app.delete("/api/admin/graph/relationships", tags=["Graph"])
async def api_delete_relationship(
    request: RelationshipDeleteRequest,
    _: bool = Depends(verify_admin),
):
    """Delete a relationship between two nodes."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"success": False, "error": "Graph database not available"}

    try:
        from clorag.core.graph_store import get_graph_store
        store = await get_graph_store()
        success = await store.delete_relationship(
            source_type=request.source_type,
            source_name=request.source_name,
            rel_type=request.rel_type,
            target_type=request.target_type,
            target_name=request.target_name,
        )
        if success:
            return {"success": True}
        return {"success": False, "error": "Relationship not found"}
    except Exception as e:
        logger.warning("graph_relationship_delete_failed", error=str(e))
        return {"success": False, "error": str(e)}


@app.patch("/api/admin/graph/relationships", tags=["Graph"])
async def api_update_relationship(
    request: RelationshipUpdateRequest,
    _: bool = Depends(verify_admin),
):
    """Update the type of a relationship between two nodes."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"success": False, "error": "Graph database not available"}

    try:
        from clorag.core.graph_store import get_graph_store
        store = await get_graph_store()
        success = await store.update_relationship_type(
            source_type=request.source_type,
            source_name=request.source_name,
            old_rel_type=request.old_rel_type,
            new_rel_type=request.new_rel_type,
            target_type=request.target_type,
            target_name=request.target_name,
        )
        if success:
            return {"success": True}
        return {"success": False, "error": "Failed to update relationship"}
    except Exception as e:
        logger.warning("graph_relationship_update_failed", error=str(e))
        return {"success": False, "error": str(e)}


# =============================================================================
# Chunk Editor Routes (Admin)
# =============================================================================


@app.get("/admin/chunks", response_class=HTMLResponse)
async def admin_chunks_list(request: Request):
    """Admin chunk browser page."""
    return templates.TemplateResponse("admin_chunks.html", {"request": request})


@app.get("/admin/chunks/{collection}/{chunk_id}", response_class=HTMLResponse)
async def admin_chunk_detail_page(
    request: Request,
    collection: str,
    chunk_id: str,
):
    """Admin chunk detail/edit page."""
    return templates.TemplateResponse(
        "admin_chunk_edit.html",
        {"request": request, "collection": collection, "chunk_id": chunk_id},
    )


@app.get("/admin/graph", response_class=HTMLResponse)
async def admin_graph_page(request: Request):
    """Admin knowledge graph explorer page."""
    return templates.TemplateResponse("admin_graph.html", {"request": request})


@app.get("/api/admin/chunks", response_model=ChunkListResponse, tags=["Chunks"])
async def api_chunks_list(
    collection: ChunkCollection = ChunkCollection.DOCS,
    limit: int = 20,
    offset: str | None = None,
    search: str | None = None,
    _: bool = Depends(verify_admin),
):
    """List chunks with pagination and optional text search."""
    vs = get_vectorstore()

    # If search query provided, use hybrid search
    if search:
        # Generate embeddings in parallel for better latency
        dense_vector, sparse_vector = await _generate_embeddings_parallel(search)

        results = await vs.search_hybrid_rrf(
            collection=collection.value,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=limit,
        )

        chunks = [
            ChunkListItem(
                id=r.id,
                collection=collection.value,
                text_preview=r.text[:200] + "..." if len(r.text) > 200 else r.text,
                title=r.payload.get("title"),
                subject=r.payload.get("subject"),
                url=r.payload.get("url"),
                chunk_index=r.payload.get("chunk_index"),
                source=r.payload.get("source"),
            )
            for r in results
        ]
        return ChunkListResponse(chunks=chunks, next_offset=None, total=len(chunks))

    # Otherwise, scroll through collection
    raw_chunks, next_off = await vs.scroll_chunks(
        collection=collection.value,
        limit=limit,
        offset=offset,
    )

    chunks = [
        ChunkListItem(
            id=c["id"],
            collection=collection.value,
            text_preview=(
                c["payload"].get("text", "")[:200] + "..."
                if len(c["payload"].get("text", "")) > 200
                else c["payload"].get("text", "")
            ),
            title=c["payload"].get("title"),
            subject=c["payload"].get("subject"),
            url=c["payload"].get("url"),
            chunk_index=c["payload"].get("chunk_index"),
            source=c["payload"].get("source"),
        )
        for c in raw_chunks
    ]

    # Get total count
    info = await vs.get_collection_info(collection.value)
    total = info.get("points_count")

    return ChunkListResponse(chunks=chunks, next_offset=next_off, total=total)


@app.get("/api/admin/chunks/{collection}/{chunk_id}", response_model=ChunkDetail, tags=["Chunks"])
async def api_chunk_get(
    collection: str,
    chunk_id: str,
    _: bool = Depends(verify_admin),
):
    """Get a single chunk's full details."""
    # Validate collection
    if collection not in [c.value for c in ChunkCollection]:
        raise HTTPException(status_code=400, detail="Invalid collection")

    vs = get_vectorstore()
    chunk = await vs.get_chunk(collection, chunk_id)

    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    payload = chunk["payload"]
    return ChunkDetail(
        id=chunk["id"],
        collection=collection,
        text=payload.get("text", ""),
        source=payload.get("source"),
        chunk_index=payload.get("chunk_index"),
        url=payload.get("url"),
        title=payload.get("title"),
        lastmod=payload.get("lastmod"),
        parent_id=payload.get("parent_id"),
        subject=payload.get("subject"),
        thread_id=payload.get("thread_id"),
        parent_case_id=payload.get("parent_case_id"),
        problem_summary=payload.get("problem_summary"),
        solution_summary=payload.get("solution_summary"),
        category=payload.get("category"),
        product=payload.get("product"),
        keywords=payload.get("keywords"),
        metadata=payload,
    )


@app.put("/api/admin/chunks/{collection}/{chunk_id}", response_model=ChunkDetail, tags=["Chunks"])
@limiter.limit("10/minute")
async def api_chunk_update(
    request: Request,
    collection: str,
    chunk_id: str,
    updates: ChunkUpdate,
    _: bool = Depends(verify_admin),
):
    """Update a chunk. Re-embeds automatically if text changes."""
    # Validate collection
    if collection not in [c.value for c in ChunkCollection]:
        raise HTTPException(status_code=400, detail="Invalid collection")

    vs = get_vectorstore()

    # Get existing chunk
    existing = await vs.get_chunk(collection, chunk_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Chunk not found")

    old_text = existing["payload"].get("text", "")
    new_text = updates.text
    text_changed = new_text is not None and new_text != old_text

    # Build metadata updates
    metadata_updates: dict = {}
    if updates.title is not None:
        metadata_updates["title"] = updates.title
    if updates.subject is not None:
        metadata_updates["subject"] = updates.subject

    # If text changed, generate new embeddings
    dense_vector = None
    sparse_vector = None
    if text_changed and new_text:
        # Generate embeddings in parallel for better latency
        dense_vector, sparse_vector = await _generate_embeddings_parallel(new_text)
        logger.info("Re-embedding chunk", chunk_id=chunk_id, collection=collection)

    # Update chunk
    success = await vs.update_chunk(
        collection=collection,
        chunk_id=chunk_id,
        text=new_text,
        metadata_updates=metadata_updates if metadata_updates else None,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to update chunk")

    # Return updated chunk
    return await api_chunk_get(collection, chunk_id, _)


@app.delete("/api/admin/chunks/{collection}/{chunk_id}", tags=["Chunks"])
@limiter.limit("10/minute")
async def api_chunk_delete(
    request: Request,
    collection: str,
    chunk_id: str,
    _: bool = Depends(verify_admin),
):
    """Delete a single chunk."""
    # Validate collection
    if collection not in [c.value for c in ChunkCollection]:
        raise HTTPException(status_code=400, detail="Invalid collection")

    vs = get_vectorstore()

    # Check if exists
    existing = await vs.get_chunk(collection, chunk_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Chunk not found")

    await vs.delete_chunk(collection, chunk_id)
    logger.info("Deleted chunk", chunk_id=chunk_id, collection=collection)

    return {"status": "deleted", "id": chunk_id, "collection": collection}


# =============================================================================
# Custom Knowledge Document Routes (Admin)
# =============================================================================


@app.get("/admin/knowledge", response_class=HTMLResponse)
async def admin_knowledge(request: Request):
    """Admin custom knowledge management page."""
    return templates.TemplateResponse("admin_knowledge.html", {"request": request})


@app.get(
    "/api/admin/knowledge",
    response_model=KnowledgeListResponse,
    tags=["Knowledge"],
)
async def api_knowledge_list(
    category: str | None = None,
    include_expired: bool = False,
    limit: int = 50,
    offset: int = 0,
    _: bool = Depends(verify_admin),
):
    """List custom documents with pagination."""
    service = get_custom_docs_service()
    items, total = await service.list_documents(
        limit=limit,
        offset=offset,
        category=category,
        include_expired=include_expired,
    )
    return KnowledgeListResponse(items=items, total=total, limit=limit, offset=offset)


@app.get(
    "/api/admin/knowledge/categories",
    tags=["Knowledge"],
)
async def api_knowledge_categories(_: bool = Depends(verify_admin)):
    """Get available document categories."""
    service = get_custom_docs_service()
    return await service.get_categories()


@app.get(
    "/api/admin/knowledge/{doc_id}",
    response_model=CustomDocument,
    tags=["Knowledge"],
)
async def api_knowledge_get(
    doc_id: str,
    _: bool = Depends(verify_admin),
):
    """Get a custom document by ID."""
    service = get_custom_docs_service()
    doc = await service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.post(
    "/api/admin/knowledge",
    response_model=CustomDocument,
    tags=["Knowledge"],
)
@limiter.limit("10/minute")
async def api_knowledge_create(
    request: Request,
    doc: Annotated[CustomDocumentCreate, Body()],
    _: bool = Depends(verify_admin),
):
    """Create a new custom document."""
    service = get_custom_docs_service()
    return await service.create_document(doc, created_by="admin")


@app.put(
    "/api/admin/knowledge/{doc_id}",
    response_model=CustomDocument,
    tags=["Knowledge"],
)
@limiter.limit("10/minute")
async def api_knowledge_update(
    request: Request,
    doc_id: str,
    updates: Annotated[CustomDocumentUpdate, Body()],
    _: bool = Depends(verify_admin),
):
    """Update a custom document."""
    service = get_custom_docs_service()
    doc = await service.update_document(doc_id, updates)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/api/admin/knowledge/{doc_id}", tags=["Knowledge"])
@limiter.limit("10/minute")
async def api_knowledge_delete(
    request: Request,
    doc_id: str,
    _: bool = Depends(verify_admin),
):
    """Delete a custom document."""
    service = get_custom_docs_service()
    if not await service.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "id": doc_id}


# Maximum file upload size (10MB)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def read_upload_with_limit(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read uploaded file with streaming size limit to prevent memory exhaustion.

    Args:
        file: FastAPI UploadFile object.
        max_bytes: Maximum allowed file size in bytes.

    Returns:
        File contents as bytes.

    Raises:
        HTTPException: If file exceeds size limit.
    """
    chunks: list[bytes] = []
    total_size = 0
    chunk_size = 64 * 1024  # 64KB chunks

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {max_bytes // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


@app.post(
    "/api/admin/knowledge/upload",
    response_model=CustomDocument,
    tags=["Knowledge"],
)
@limiter.limit("10/minute")
async def api_knowledge_upload(
    request: Request,
    file: UploadFile,
    title: Annotated[str, Form()] = "",
    category: Annotated[str, Form()] = "other",
    tags: Annotated[str, Form()] = "",
    url_reference: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
    _: bool = Depends(verify_admin),
):
    """Upload a file (txt, md, pdf) as a custom document.

    Extracts text content from the uploaded file and creates a new document.
    Supported formats: .txt, .md, .pdf
    Max file size: 10MB
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file extension
    filename = file.filename.lower()
    if not filename.endswith((".txt", ".md", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: .txt, .md, .pdf",
        )

    # Read file content with streaming size limit
    content_bytes = await read_upload_with_limit(file)
    content = ""

    if filename.endswith((".txt", ".md")):
        # Text/Markdown files - decode as UTF-8
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Try latin-1 as fallback
            try:
                content = content_bytes.decode("latin-1")
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode file. Please ensure it's a valid text file.",
                )
    elif filename.endswith(".pdf"):
        # PDF files - extract text using pypdf
        try:
            from pypdf import PdfReader

            pdf_file = io.BytesIO(content_bytes)
            reader = PdfReader(pdf_file)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            content = "\n\n".join(text_parts)
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PDF support not available. Install pypdf package.",
            )
        except Exception as e:
            logger.warning("PDF extraction failed", error=str(e))
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from PDF. Please ensure the file is a valid PDF.",
            )

    # Validate content
    content = content.strip()
    if not content or len(content) < 10:
        raise HTTPException(
            status_code=400,
            detail="File content is empty or too short (min 10 characters).",
        )

    # Use filename as title if not provided
    doc_title = title.strip() if title.strip() else Path(file.filename).stem

    # Parse tags from comma-separated string
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Validate category
    try:
        doc_category = DocumentCategory(category)
    except ValueError:
        doc_category = DocumentCategory.OTHER

    # Create document
    doc_create = CustomDocumentCreate(
        title=doc_title,
        content=content,
        tags=tag_list,
        category=doc_category,
        url_reference=url_reference.strip() if url_reference.strip() else None,
        notes=notes.strip() if notes.strip() else None,
    )

    try:
        logger.info(
            "Creating custom document from upload",
            title=doc_title,
            category=doc_category.value,
            content_length=len(content),
            filename=file.filename,
        )
        service = get_custom_docs_service()
        doc = await service.create_document(doc_create, created_by="admin")
        logger.info("Custom document created successfully", doc_id=doc.id, title=doc.title)
        return doc
    except Exception as e:
        logger.error(
            "Failed to create custom document",
            error=str(e),
            error_type=type(e).__name__,
            title=doc_title,
            content_length=len(content),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to create document. Please try again or contact support.",
        )


# =============================================================================
# Draft Creation Routes (Admin)
# =============================================================================


@app.get("/admin/drafts", response_class=HTMLResponse)
async def admin_drafts(request: Request):
    """Admin draft management page."""
    return templates.TemplateResponse("admin_drafts.html", {"request": request})


@app.get("/api/admin/drafts/status", tags=["Drafts"])
async def api_drafts_status(_: bool = Depends(verify_admin)):
    """Get draft creation system status."""
    settings = get_settings()
    scheduler = get_scheduler()

    scheduler_running = scheduler is not None and scheduler.running if scheduler else False
    next_run = None

    if scheduler_running and scheduler:
        job = scheduler.get_job("draft_creation")
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

    return {
        "scheduler_running": scheduler_running,
        "next_run": next_run,
        "poll_interval_minutes": settings.draft_poll_interval_minutes,
        "polling_enabled": settings.draft_polling_enabled,
    }


@app.get("/api/admin/drafts/pending", tags=["Drafts"])
async def api_pending_threads(
    limit: int = 20,
    _: bool = Depends(verify_admin),
):
    """Get threads pending draft creation."""
    from clorag.drafts import DraftCreationPipeline

    pipeline = DraftCreationPipeline()
    pending = await pipeline.get_pending_threads(limit=limit)

    return [t.to_dict() for t in pending]


@app.get("/api/admin/drafts/thread/{thread_id}", tags=["Drafts"])
async def api_thread_detail(
    thread_id: str,
    _: bool = Depends(verify_admin),
):
    """Get full thread details with all messages."""
    from clorag.drafts import GmailDraftService

    gmail = GmailDraftService()
    thread = await gmail.get_thread(thread_id)

    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    detail = gmail.extract_thread_detail(thread)
    if not detail:
        raise HTTPException(status_code=404, detail="Could not extract thread details")

    return detail.to_dict()


@app.post("/api/admin/drafts/preview/{thread_id}", tags=["Drafts"])
async def api_preview_draft(
    thread_id: str,
    _: bool = Depends(verify_admin),
):
    """Preview draft for a specific thread without creating it."""
    from clorag.drafts import DraftCreationPipeline, DraftPreview

    pipeline = DraftCreationPipeline()
    result = await pipeline.process_single_thread(thread_id, preview_only=True)

    if not result or not isinstance(result, DraftPreview):
        raise HTTPException(status_code=404, detail="Thread not found or not eligible")

    return result.to_dict()


@app.post("/api/admin/drafts/create/{thread_id}", tags=["Drafts"])
async def api_create_draft(
    thread_id: str,
    _: bool = Depends(verify_admin),
):
    """Create draft for a specific thread."""
    from clorag.drafts import DraftCreationPipeline, DraftResult

    pipeline = DraftCreationPipeline()
    result = await pipeline.process_single_thread(thread_id, preview_only=False)

    if not result or not isinstance(result, DraftResult):
        raise HTTPException(status_code=404, detail="Thread not found or draft creation failed")

    return {
        "success": True,
        **result.to_dict(),
    }


@app.post("/api/admin/drafts/run", tags=["Drafts"])
async def api_run_draft_pipeline(
    max_drafts: int = 5,
    _: bool = Depends(verify_admin),
):
    """Manually trigger the draft creation pipeline."""
    from clorag.drafts import DraftCreationPipeline

    pipeline = DraftCreationPipeline()
    result = await pipeline.run(max_drafts=max_drafts)

    return result.to_dict()


# =============================================================================
# Search Debug Routes (Admin)
# =============================================================================


@app.get("/admin/search-debug", response_class=HTMLResponse)
async def admin_search_debug(request: Request):
    """Admin search debug page - shows chunks and LLM response."""
    return templates.TemplateResponse("admin_search_debug.html", {"request": request})


@app.post("/api/admin/search-debug", tags=["Debug"])
async def api_search_debug(
    req: SearchRequest,
    _: bool = Depends(verify_admin),
) -> DebugSearchResponse:
    """Debug search endpoint showing chunks and LLM details."""
    total_start = time.time()

    # Perform search and measure retrieval time
    retrieval_start = time.time()
    results, chunks_for_synthesis, graph_context, _ = await perform_search(req)
    retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

    # Build context (same as synthesis)
    context = _build_context(chunks_for_synthesis, graph_context=graph_context)
    user_prompt = f"Question: {req.query}\n\nContext:\n{context}"

    # Synthesize and measure time
    settings = get_settings()
    synthesis_start = time.time()
    if not chunks_for_synthesis:
        llm_response = "No relevant information found for your query."
    else:
        response = await get_anthropic().messages.create(
            model=settings.sonnet_model,
            max_tokens=1500,
            system=get_prompt("synthesis.web_answer"),
            messages=[{"role": "user", "content": user_prompt}],
        )
        llm_response = response.content[0].text
    synthesis_time_ms = int((time.time() - synthesis_start) * 1000)

    total_time_ms = int((time.time() - total_start) * 1000)

    # Build detailed chunk info
    detailed_chunks = []
    for i, (result, chunk) in enumerate(zip(results, chunks_for_synthesis)):
        detailed_chunks.append({
            "index": i + 1,
            "score": result.score,
            "source_type": chunk.get("source_type", "unknown"),
            "title": chunk.get("title") or chunk.get("subject", "Untitled"),
            "url": chunk.get("url"),
            "text": chunk.get("text", "")[:3000],  # Limit text size
            "text_length": len(chunk.get("text", "")),
        })

    return DebugSearchResponse(
        query=req.query,
        source=req.source.value,
        retrieval_time_ms=retrieval_time_ms,
        synthesis_time_ms=synthesis_time_ms,
        total_time_ms=total_time_ms,
        chunks=detailed_chunks,
        llm_prompt=user_prompt,
        system_prompt=get_prompt("synthesis.web_answer"),
        llm_response=llm_response,
        model=settings.haiku_model,
    )


# =============================================================================
# Admin OpenAPI Documentation
# =============================================================================

# Cache for admin OpenAPI schema
_admin_openapi_schema: dict | None = None


def get_admin_openapi_schema() -> dict:
    """Generate OpenAPI schema for admin endpoints only.

    Filters the main app schema to include only /api/admin/* routes.
    """
    global _admin_openapi_schema
    if _admin_openapi_schema:
        return _admin_openapi_schema

    # Generate full schema
    full_schema = get_openapi(
        title="Cyanview Admin API",
        version="1.0.0",
        description="Admin API for Cyanview AI Search.",
        routes=app.routes,
    )

    # Filter paths to include only /api/admin/* routes
    admin_paths = {
        path: ops
        for path, ops in full_schema.get("paths", {}).items()
        if path.startswith("/api/admin")
    }
    full_schema["paths"] = admin_paths

    # Filter tags to include only admin-related tags
    admin_tags = ["Authentication", "Backup", "Cameras", "Analytics", "Drafts", "Debug"]
    full_schema["tags"] = [
        {"name": tag, "description": f"{tag} operations"}
        for tag in admin_tags
    ]

    _admin_openapi_schema = full_schema
    return _admin_openapi_schema


@app.get("/admin/openapi.json", include_in_schema=False)
async def admin_openapi_json(
    admin_session: Annotated[str | None, Cookie()] = None,
):
    """Serve OpenAPI schema for admin endpoints only (requires authentication)."""
    # Verify session
    if not admin_session:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        serializer = get_session_serializer()
        data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
        if not data.get("authenticated"):
            raise HTTPException(status_code=401, detail="Authentication required")
    except (SignatureExpired, BadSignature):
        raise HTTPException(status_code=401, detail="Session expired or invalid")

    return JSONResponse(content=get_admin_openapi_schema())


@app.get("/admin/api-docs", include_in_schema=False)
async def admin_swagger_ui(
    admin_session: Annotated[str | None, Cookie()] = None,
):
    """Serve Swagger UI for admin API (requires authentication)."""
    # Verify session
    if not admin_session:
        return RedirectResponse(url="/admin/login?next=/admin/api-docs")

    try:
        serializer = get_session_serializer()
        data = serializer.loads(admin_session, max_age=ADMIN_SESSION_MAX_AGE)
        if not data.get("authenticated"):
            return RedirectResponse(url="/admin/login?next=/admin/api-docs")
    except (SignatureExpired, BadSignature):
        return RedirectResponse(url="/admin/login?next=/admin/api-docs")

    return get_swagger_ui_html(
        openapi_url="/admin/openapi.json",
        title="Cyanview Admin API",
        swagger_favicon_url="/static/favicon.ico",
    )


# =============================================================================
# Prompt Management Routes (Admin)
# =============================================================================


@app.get("/admin/prompts", response_class=HTMLResponse)
async def admin_prompts(request: Request):
    """Admin prompt management page."""
    return templates.TemplateResponse("admin_prompts.html", {"request": request})


@app.get("/api/admin/prompts", tags=["Prompts"])
async def api_prompts_list(
    category: str | None = None,
    _: bool = Depends(verify_admin),
):
    """List all prompts from database and defaults."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    return pm.list_all_prompts(category=category)


@app.get("/api/admin/prompts/by-key/{key:path}", tags=["Prompts"])
async def api_prompt_by_key(
    key: str,
    _: bool = Depends(verify_admin),
):
    """Get a prompt by its key (works for both database and default prompts)."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    try:
        return pm.get_prompt_with_metadata(key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {key}")


@app.get("/api/admin/prompts/{prompt_id}", tags=["Prompts"])
async def api_prompt_get(
    prompt_id: str,
    _: bool = Depends(verify_admin),
):
    """Get a database prompt by ID with metadata."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    prompt = pm.get_prompt_by_id(prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"source": "database", "prompt": prompt.to_dict()}


@app.get("/api/admin/prompts/{prompt_id}/versions", tags=["Prompts"])
async def api_prompt_versions(
    prompt_id: str,
    _: bool = Depends(verify_admin),
):
    """Get version history for a prompt."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    versions = pm.get_prompt_versions(prompt_id)
    return [v.to_dict() for v in versions]


@app.put("/api/admin/prompts/{prompt_id}", tags=["Prompts"])
@limiter.limit("30/minute")
async def api_prompt_update(
    request: Request,
    prompt_id: str,
    updates: Annotated[PromptUpdateRequest, Body()],
    _: bool = Depends(verify_admin),
):
    """Update a prompt (creates a new version if content changed)."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()

    # Detect variables in new content if provided
    variables = None
    if updates.content:
        variables = pm.detect_variables(updates.content)

    prompt = pm.update_prompt(
        prompt_id=prompt_id,
        content=updates.content,
        name=updates.name,
        description=updates.description,
        model=updates.model,
        variables=variables,
        change_note=updates.change_note,
        updated_by="admin",
    )
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return prompt.to_dict()


@app.post("/api/admin/prompts/from-default", tags=["Prompts"])
@limiter.limit("10/minute")
async def api_prompt_create_from_default(
    request: Request,
    data: Annotated[PromptCreateFromDefaultRequest, Body()],
    _: bool = Depends(verify_admin),
):
    """Create a database prompt from a default, optionally with modifications."""
    from clorag.core.prompt_db import get_prompt_database
    from clorag.services.default_prompts import get_default_prompt
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    default = get_default_prompt(data.key)
    if not default:
        raise HTTPException(status_code=404, detail=f"Default prompt not found: {data.key}")

    db = get_prompt_database()

    # Check if already exists in database
    existing = db.get_prompt_by_key(data.key)
    if existing:
        # Update existing
        variables = pm.detect_variables(data.content or default.content)
        prompt = pm.update_prompt(
            prompt_id=existing.id,
            content=data.content or existing.content,
            name=data.name or existing.name,
            description=data.description or existing.description,
            model=data.model or existing.model,
            variables=variables,
            change_note="Updated from admin UI",
            updated_by="admin",
        )
        return prompt.to_dict() if prompt else existing.to_dict()

    # Create new from default
    content = data.content or default.content
    variables = pm.detect_variables(content)

    prompt = db.create_prompt(
        key=data.key,
        name=data.name or default.name,
        content=content,
        category=default.category,
        description=data.description or default.description,
        model=data.model or default.model,
        variables=variables,
        created_by="admin",
    )
    return prompt.to_dict()


@app.post("/api/admin/prompts/{prompt_id}/rollback", tags=["Prompts"])
@limiter.limit("10/minute")
async def api_prompt_rollback(
    request: Request,
    prompt_id: str,
    data: Annotated[PromptRollbackRequest, Body()],
    _: bool = Depends(verify_admin),
):
    """Rollback a prompt to a previous version."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    prompt = pm.rollback_prompt(prompt_id, data.version, rolled_back_by="admin")
    if not prompt:
        raise HTTPException(status_code=404, detail="Version not found")
    return prompt.to_dict()


@app.post("/api/admin/prompts/initialize", tags=["Prompts"])
@limiter.limit("5/minute")
async def api_prompts_initialize(
    request: Request,
    force: bool = False,
    _: bool = Depends(verify_admin),
):
    """Initialize database with default prompts."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    result = pm.initialize_defaults(force=force)
    return result


@app.post("/api/admin/prompts/reload", tags=["Prompts"])
async def api_prompts_reload(
    _: bool = Depends(verify_admin),
):
    """Reload all prompt caches (hot reload)."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    pm.reload_all()
    return {"status": "ok", "message": "All prompt caches reloaded"}


@app.get("/api/admin/prompts/cache-stats", tags=["Prompts"])
async def api_prompts_cache_stats(
    _: bool = Depends(verify_admin),
):
    """Get prompt cache statistics."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    return pm.get_cache_stats()


@app.post("/api/admin/prompts/test", tags=["Prompts"])
@limiter.limit("20/minute")
async def api_prompt_test(
    request: Request,
    data: Annotated[PromptTestRequest, Body()],
    _: bool = Depends(verify_admin),
):
    """Test a prompt with variable substitution (no LLM call)."""
    from clorag.services.prompt_manager import get_prompt_manager

    pm = get_prompt_manager()
    try:
        result = pm._substitute_variables(data.content, data.variables)
        return {
            "success": True,
            "result": result,
            "detected_variables": pm.detect_variables(data.content),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app
