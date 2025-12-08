"""FastAPI web application for AI Search with Claude synthesis."""

import io
import json
import secrets
import time
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import anthropic
import anyio
import structlog
from fastapi import Body, Cookie, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from clorag.config import get_settings
from clorag.core.analytics_db import AnalyticsDatabase
from clorag.core.database import get_camera_database
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.models.camera import Camera, CameraCreate, CameraSource, CameraUpdate

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


# Initialize FastAPI app
app = FastAPI(title="Cyanview AI Search", version="1.0.0")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cyanview.cloud"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add timeout middleware
app.add_middleware(TimeoutMiddleware)

# Optimized system prompt - single source of truth
SYNTHESIS_SYSTEM_PROMPT = """You are a Cyanview support expert, representing Cyanview's excellence in broadcast camera control solutions.

TONE: Empathetic, warm, professional. Like a knowledgeable colleague who genuinely wants to help.

STRUCTURE:
1. Start conversationally - acknowledge the question warmly
2. Technical content - use bullet points, numbered steps, or tables
3. End conversationally - brief closing, then relevant doc links

FORMAT RULES:
- **Bold** product names (RCP, RIO, CI0, VP4, CVP)
- Bullet points for specs, features, options
- Numbered steps for procedures
- Code blocks for IP addresses, commands, config values
- Length adapts to complexity - brief for simple, detailed for complex

CONTENT RULES:
- Use ONLY the provided context - never invent
- Sound natural - avoid "based on the context" or "according to the documentation"
- For unknowns: suggest checking the specific product page
- Never say "contact Cyanview support" (you ARE the support)

ALWAYS END WITH:
After your answer, add a "📚 Related documentation:" section with 1-3 most relevant links from the context (use the URLs provided in [Doc: url] tags).

Match the user's language (EN/FR)."""

# Static files and templates
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize clients (lazy loading)
_vectorstore: VectorStore | None = None
_embeddings: EmbeddingsClient | None = None
_sparse_embeddings: SparseEmbeddingsClient | None = None
_anthropic: anthropic.AsyncAnthropic | None = None
_analytics_db: AnalyticsDatabase | None = None


def get_vectorstore() -> VectorStore:
    """Get or create VectorStore instance."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStore()
    return _vectorstore


def get_embeddings() -> EmbeddingsClient:
    """Get or create EmbeddingsClient instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingsClient()
    return _embeddings


def get_sparse_embeddings() -> SparseEmbeddingsClient:
    """Get or create SparseEmbeddingsClient instance for BM25."""
    global _sparse_embeddings
    if _sparse_embeddings is None:
        _sparse_embeddings = SparseEmbeddingsClient()
    return _sparse_embeddings


def get_anthropic() -> anthropic.AsyncAnthropic:
    """Get or create Anthropic client."""
    global _anthropic
    if _anthropic is None:
        settings = get_settings()
        _anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    return _anthropic


def get_analytics_db() -> AnalyticsDatabase:
    """Get or create AnalyticsDatabase instance (separate from camera DB)."""
    global _analytics_db
    if _analytics_db is None:
        settings = get_settings()
        _analytics_db = AnalyticsDatabase(settings.analytics_database_path)
    return _analytics_db


# =============================================================================
# Conversation Session Management
# =============================================================================

MAX_CONVERSATION_HISTORY = 3  # Keep last 3 Q&A exchanges
SESSION_TTL_SECONDS = 30 * 60  # 30 minutes session timeout
MAX_SESSIONS = 1000  # Maximum concurrent sessions


@dataclass
class ConversationExchange:
    """A single Q&A exchange in the conversation."""

    query: str
    answer: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    """Server-side conversation session with history."""

    session_id: str
    exchanges: list[ConversationExchange] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def add_exchange(self, query: str, answer: str) -> None:
        """Add a new exchange and trim to max history."""
        self.exchanges.append(ConversationExchange(query=query, answer=answer))
        # Keep only the last N exchanges
        if len(self.exchanges) > MAX_CONVERSATION_HISTORY:
            self.exchanges = self.exchanges[-MAX_CONVERSATION_HISTORY:]
        self.last_accessed = time.time()

    def get_context_messages(self) -> list[dict]:
        """Get conversation history as Claude message format."""
        messages = []
        for exchange in self.exchanges:
            messages.append({"role": "user", "content": exchange.query})
            messages.append({"role": "assistant", "content": exchange.answer})
        return messages

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_accessed) > SESSION_TTL_SECONDS


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


# Global session store
_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Get or create session store singleton."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store


def _build_context(chunks: list[dict], max_chunks: int = 8) -> str:
    """Build context string from chunks for Claude synthesis."""
    parts = []
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        text = chunk.get("text", "")[:2000]
        if chunk.get("source_type") == "documentation":
            parts.append(f"[{i} Doc: {chunk.get('url', '')}]\n{text}")
        else:
            parts.append(f"[{i} Case: {chunk.get('subject', 'Support')}]\n{text}")
    return "\n---\n".join(parts)


class SearchSource(str, Enum):
    """Search source options."""

    DOCS = "docs"
    GMAIL = "gmail"
    BOTH = "both"


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, max_length=2000)
    source: SearchSource = SearchSource.BOTH
    limit: int = Field(10, ge=1, le=50)
    session_id: str | None = Field(None, description="Session ID for follow-up conversations")


class SourceLink(BaseModel):
    """A source link for the answer."""

    title: str
    url: str | None = None
    source_type: str  # "documentation" or "gmail_case"


class SearchResult(BaseModel):
    """Individual search result."""

    score: float
    source: str
    title: str
    url: str | None = None
    subject: str | None = None
    snippet: str
    metadata: dict


class SearchResponse(BaseModel):
    """Search response model with AI-generated answer."""

    query: str
    source: str
    answer: str  # Claude-generated comprehensive answer
    source_links: list[SourceLink]  # Top 3 relevant sources
    results: list[SearchResult]
    total: int
    session_id: str | None = None  # Session ID for follow-up conversations


def _extract_source_links(
    chunks: list[dict],
    max_links: int = 3,
    as_model: bool = False,
) -> list[SourceLink] | list[dict]:
    """Extract unique source links from chunks."""
    seen: set[str] = set()
    links: list = []

    for chunk in chunks:
        if len(links) >= max_links:
            break

        if chunk.get("source_type") == "documentation":
            url = chunk.get("url")
            if url and url not in seen:
                seen.add(url)
                link = {"title": chunk.get("title", "Documentation"), "url": url, "source_type": "documentation"}
                links.append(SourceLink(**link) if as_model else link)
        else:
            subject = chunk.get("subject", "Support Case")
            key = f"case:{subject}"
            if key not in seen:
                seen.add(key)
                link = {"title": subject, "url": None, "source_type": "gmail_case"}
                links.append(SourceLink(**link) if as_model else link)

    return links


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


async def synthesize_answer(
    query: str,
    chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    """Use Claude Haiku to synthesize an answer from retrieved chunks.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        conversation_history: Optional list of previous messages for follow-up context.
    """
    if not chunks:
        return "No relevant information found for your query."

    settings = get_settings()
    context = _build_context(chunks)

    # Build messages with conversation history
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"})

    response = await get_anthropic().messages.create(
        model=settings.haiku_model,
        max_tokens=1500,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


async def synthesize_answer_stream(
    query: str,
    chunks: list[dict],
    conversation_history: list[dict] | None = None,
):
    """Stream answer synthesis using Claude Haiku.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        conversation_history: Optional list of previous messages for follow-up context.
    """
    if not chunks:
        yield "No relevant information found for your query."
        return

    settings = get_settings()
    context = _build_context(chunks)

    # Build messages with conversation history
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"})

    async with get_anthropic().messages.stream(
        model=settings.haiku_model,
        max_tokens=1500,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def _perform_search(req: SearchRequest) -> tuple[list[SearchResult], list[dict]]:
    """Perform hybrid search and return results with chunks for synthesis.

    Uses hybrid search (dense + sparse vectors) with RRF fusion for better
    results on queries with specific model numbers or technical terms.

    Returns:
        Tuple of (search_results, chunks_for_synthesis)
    """
    vs = get_vectorstore()
    emb = get_embeddings()
    sparse_emb = get_sparse_embeddings()

    # Generate both dense and sparse query embeddings
    dense_vector = await emb.embed_query(req.query)
    sparse_vector = sparse_emb.embed_query(req.query)

    results: list[SearchResult] = []
    chunks_for_synthesis: list[dict] = []

    if req.source == SearchSource.DOCS:
        # Search only documentation with hybrid RRF
        docs = await vs.search_docs_hybrid(dense_vector, sparse_vector, limit=req.limit)
        for doc in docs:
            results.append(
                SearchResult(
                    score=doc.score,
                    source="documentation",
                    title=doc.payload.get("title", "Untitled"),
                    url=doc.payload.get("url"),
                    snippet=_truncate(doc.payload.get("text", ""), 300),
                    metadata=doc.payload,
                )
            )
            chunks_for_synthesis.append({
                "text": doc.payload.get("text", ""),
                "source_type": "documentation",
                "url": doc.payload.get("url"),
                "title": doc.payload.get("title", "Untitled"),
            })

    elif req.source == SearchSource.GMAIL:
        # Search only Gmail cases with hybrid RRF
        cases = await vs.search_cases_hybrid(dense_vector, sparse_vector, limit=req.limit)
        for case in cases:
            results.append(
                SearchResult(
                    score=case.score,
                    source="gmail_case",
                    title=case.payload.get("subject", "No Subject"),
                    subject=case.payload.get("subject"),
                    snippet=_truncate(case.payload.get("text", ""), 300),
                    metadata=case.payload,
                )
            )
            chunks_for_synthesis.append({
                "text": case.payload.get("text", ""),
                "source_type": "gmail_case",
                "subject": case.payload.get("subject", "Support Case"),
            })

    else:
        # Hybrid RRF search across both collections
        hybrid = await vs.hybrid_search_rrf(dense_vector, sparse_vector, limit=req.limit)
        for item in hybrid:
            source_type = item.payload.get("_source", "unknown")
            if source_type == "documentation":
                results.append(
                    SearchResult(
                        score=item.score,
                        source="documentation",
                        title=item.payload.get("title", "Untitled"),
                        url=item.payload.get("url"),
                        snippet=_truncate(item.payload.get("text", ""), 300),
                        metadata=item.payload,
                    )
                )
                chunks_for_synthesis.append({
                    "text": item.payload.get("text", ""),
                    "source_type": "documentation",
                    "url": item.payload.get("url"),
                    "title": item.payload.get("title", "Untitled"),
                })
            else:
                results.append(
                    SearchResult(
                        score=item.score,
                        source="gmail_case",
                        title=item.payload.get("subject", "No Subject"),
                        subject=item.payload.get("subject"),
                        snippet=_truncate(item.payload.get("text", ""), 300),
                        metadata=item.payload,
                    )
                )
                chunks_for_synthesis.append({
                    "text": item.payload.get("text", ""),
                    "source_type": "gmail_case",
                    "subject": item.payload.get("subject", "Support Case"),
                })

    return results, chunks_for_synthesis


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

    # Use helper to perform search (we only need chunks_for_synthesis)
    _, chunks_for_synthesis = await _perform_search(req)

    # Extract top 3 unique source links
    source_links = _extract_source_links(chunks_for_synthesis)

    # Capture start time for analytics
    search_start_time = start_time

    async def generate():
        collected_response = []
        try:
            # Send session_id first so frontend can track the conversation
            yield f"data: {json.dumps({'type': 'session', 'session_id': session.session_id})}\n\n"

            # Stream the answer with conversation history context
            async for chunk in synthesize_answer_stream(
                req.query, chunks_for_synthesis, conversation_history
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
    results, chunks_for_synthesis = await _perform_search(req)

    # Generate synthesized answer using Claude Haiku with conversation history
    answer = await synthesize_answer(req.query, chunks_for_synthesis, conversation_history)

    # Store this exchange in the session for future follow-ups
    session.add_exchange(req.query, answer)

    # Extract top 3 unique source links
    source_links = _extract_source_links(chunks_for_synthesis, as_model=True)

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


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


# =============================================================================
# Camera Compatibility Routes
# =============================================================================


# Session cookie settings
ADMIN_SESSION_COOKIE = "admin_session"
ADMIN_SESSION_MAX_AGE = 24 * 60 * 60  # 24 hours


def get_session_serializer() -> URLSafeTimedSerializer:
    """Get session serializer using admin password as secret key."""
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")
    return URLSafeTimedSerializer(settings.admin_password.get_secret_value())


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
        x_admin_password.encode('utf-8'),
        settings.admin_password.get_secret_value().encode('utf-8')
    ):
        return True

    raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/cameras", response_class=HTMLResponse)
async def cameras_list(
    request: Request,
    manufacturer: str | None = None,
    device_type: str | None = None,
    port: str | None = None,
    protocol: str | None = None,
):
    """Render the public camera compatibility list."""
    db = get_camera_database()
    cameras = db.list_cameras(
        manufacturer=manufacturer,
        device_type=device_type,
        port=port,
        protocol=protocol,
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
        },
    )


@app.get("/api/cameras", response_model=list[Camera])
async def api_cameras_list(
    manufacturer: str | None = None,
    device_type: str | None = None,
    port: str | None = None,
    protocol: str | None = None,
):
    """Get all cameras as JSON."""
    db = get_camera_database()
    return db.list_cameras(
        manufacturer=manufacturer,
        device_type=device_type,
        port=port,
        protocol=protocol,
    )


@app.get("/api/cameras/search", response_model=list[Camera])
async def api_cameras_search(q: str):
    """Search cameras by name or manufacturer."""
    db = get_camera_database()
    return db.search_cameras(q)


@app.get("/api/cameras/stats")
async def api_cameras_stats():
    """Get camera database statistics."""
    db = get_camera_database()
    return db.get_stats()


@app.get("/api/cameras/{camera_id}", response_model=Camera)
async def api_camera_get(camera_id: int):
    """Get a single camera by ID."""
    db = get_camera_database()
    camera = db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


# Admin routes (protected)


@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    """Public user guide page."""
    return templates.TemplateResponse("help.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_index(request: Request):
    """Admin index page with links to all admin pages."""
    return templates.TemplateResponse("admin_index.html", {"request": request})


@app.get("/admin/docs", response_class=HTMLResponse)
async def admin_docs(request: Request):
    """Admin technical documentation page."""
    return templates.TemplateResponse("admin_docs.html", {"request": request})


@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Admin login page."""
    return templates.TemplateResponse("admin_login.html", {"request": request})


class LoginRequest(BaseModel):
    """Login request model."""

    password: str


class LoginResponse(BaseModel):
    """Login response model."""

    success: bool
    message: str


@app.post("/api/admin/login")
@limiter.limit("5/minute")
async def api_admin_login(
    request: Request,
    login_req: LoginRequest,
    response: Response,
):
    """Login and set session cookie.

    Returns success status and sets httponly cookie on success.
    """
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")

    # Verify password with timing-safe comparison
    if not secrets.compare_digest(
        login_req.password.encode('utf-8'),
        settings.admin_password.get_secret_value().encode('utf-8')
    ):
        raise HTTPException(status_code=401, detail="Invalid password")

    # Create signed session token
    serializer = get_session_serializer()
    token = serializer.dumps({"authenticated": True, "ts": time.time()})

    # Set httponly cookie (secure in production)
    response.set_cookie(
        key=ADMIN_SESSION_COOKIE,
        value=token,
        max_age=ADMIN_SESSION_MAX_AGE,
        httponly=True,
        secure=True,  # HTTPS only
        samesite="strict",
    )

    return LoginResponse(success=True, message="Login successful")


@app.post("/api/admin/logout")
async def api_admin_logout(response: Response):
    """Logout and clear session cookie."""
    response.delete_cookie(
        key=ADMIN_SESSION_COOKIE,
        httponly=True,
        secure=True,
        samesite="strict",
    )
    return {"success": True, "message": "Logged out"}


@app.get("/api/admin/session")
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


@app.get("/api/admin/backup")
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


@app.post("/api/admin/cameras", response_model=Camera)
@limiter.limit("5/minute")
async def api_camera_create(
    request: Request,
    camera: Annotated[CameraCreate, Body()],
    _: bool = Depends(verify_admin),
):
    """Create a new camera entry."""
    db = get_camera_database()
    return db.create_camera(camera, CameraSource.MANUAL)


@app.put("/api/admin/cameras/{camera_id}", response_model=Camera)
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


@app.delete("/api/admin/cameras/{camera_id}")
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
# Search Analytics Routes (Admin)
# =============================================================================


@app.get("/admin/analytics", response_class=HTMLResponse)
async def admin_analytics(request: Request):
    """Admin analytics dashboard page."""
    return templates.TemplateResponse("admin_analytics.html", {"request": request})


@app.get("/api/admin/search-stats")
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


@app.get("/api/admin/search-stats/popular")
async def api_popular_queries(
    limit: int = 10,
    days: int = 30,
    _: bool = Depends(verify_admin),
):
    """Get popular search queries."""
    analytics = get_analytics_db()
    return analytics.get_popular_queries(limit=limit, days=days)


@app.get("/api/admin/search/{search_id}")
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


@app.get("/api/admin/conversations")
async def api_get_conversations(
    limit: int = 20,
    _: bool = Depends(verify_admin),
):
    """Get recent conversations grouped by session_id."""
    analytics = get_analytics_db()
    return analytics.get_recent_conversations(limit=limit)


# =============================================================================
# Search Debug Routes (Admin)
# =============================================================================


class DebugSearchResponse(BaseModel):
    """Debug search response with full chunk details."""

    query: str
    source: str
    # Timing
    retrieval_time_ms: int
    synthesis_time_ms: int
    total_time_ms: int
    # Chunks retrieved
    chunks: list[dict]
    # Prompt sent to LLM
    llm_prompt: str
    system_prompt: str
    # LLM response
    llm_response: str
    model: str


@app.get("/admin/search-debug", response_class=HTMLResponse)
async def admin_search_debug(request: Request):
    """Admin search debug page - shows chunks and LLM response."""
    return templates.TemplateResponse("admin_search_debug.html", {"request": request})


@app.post("/api/admin/search-debug")
async def api_search_debug(
    req: SearchRequest,
    _: bool = Depends(verify_admin),
) -> DebugSearchResponse:
    """Debug search endpoint showing chunks and LLM details."""
    total_start = time.time()

    # Perform search and measure retrieval time
    retrieval_start = time.time()
    results, chunks_for_synthesis = await _perform_search(req)
    retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

    # Build context (same as synthesis)
    context = _build_context(chunks_for_synthesis)
    user_prompt = f"Question: {req.query}\n\nContext:\n{context}"

    # Synthesize and measure time
    settings = get_settings()
    synthesis_start = time.time()
    if not chunks_for_synthesis:
        llm_response = "No relevant information found for your query."
    else:
        response = await get_anthropic().messages.create(
            model=settings.haiku_model,
            max_tokens=1500,
            system=SYNTHESIS_SYSTEM_PROMPT,
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
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        llm_response=llm_response,
        model=settings.haiku_model,
    )


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app
