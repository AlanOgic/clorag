"""FastAPI web application for AI Search with Claude synthesis."""

import json
import secrets
import time
from enum import Enum
from pathlib import Path
from typing import Annotated

import anthropic
import anyio
import structlog
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

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


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "1.0.0"}


async def synthesize_answer(query: str, chunks: list[dict]) -> str:
    """Use Claude Haiku to synthesize an answer from retrieved chunks."""
    if not chunks:
        return "No relevant information found for your query."

    context = _build_context(chunks)
    response = await get_anthropic().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}],
    )
    return response.content[0].text


async def synthesize_answer_stream(query: str, chunks: list[dict]):
    """Stream answer synthesis using Claude Haiku 4.5."""
    if not chunks:
        yield "No relevant information found for your query."
        return

    context = _build_context(chunks)
    async with get_anthropic().messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}],
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
async def search_stream(req: SearchRequest):
    """Perform RAG search with streaming Claude-powered answer synthesis.

    Uses hybrid search (dense + sparse vectors) with RRF fusion for better
    results on queries with specific model numbers or technical terms.
    """
    start_time = time.time()

    # Use helper to perform search (we only need chunks_for_synthesis)
    _, chunks_for_synthesis = await _perform_search(req)

    # Extract top 3 unique source links
    source_links = _extract_source_links(chunks_for_synthesis)

    # Log search to analytics (non-blocking)
    response_time_ms = int((time.time() - start_time) * 1000)
    try:
        get_analytics_db().log_search(
            query=req.query,
            source=req.source.value,
            response_time_ms=response_time_ms,
            results_count=len(chunks_for_synthesis),
        )
    except Exception as e:
        logger.warning("Failed to log search analytics", error=str(e))

    async def generate():
        try:
            # Stream the answer first
            async for chunk in synthesize_answer_stream(req.query, chunks_for_synthesis):
                yield f"data: {json.dumps({'type': 'text', 'text': chunk})}\n\n"

            # Then send sources at the end
            yield f"data: {json.dumps({'type': 'sources', 'sources': source_links})}\n\n"

            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
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
async def search(req: SearchRequest):
    """Perform RAG search with Claude-powered answer synthesis.

    Uses hybrid search (dense + sparse vectors) with RRF fusion for better
    results on queries with specific model numbers or technical terms.
    """
    start_time = time.time()

    # Use helper to perform search
    results, chunks_for_synthesis = await _perform_search(req)

    # Generate synthesized answer using Claude Haiku
    answer = await synthesize_answer(req.query, chunks_for_synthesis)

    # Extract top 3 unique source links
    source_links = _extract_source_links(chunks_for_synthesis, as_model=True)

    # Log search to analytics (non-blocking)
    response_time_ms = int((time.time() - start_time) * 1000)
    try:
        get_analytics_db().log_search(
            query=req.query,
            source=req.source.value,
            response_time_ms=response_time_ms,
            results_count=len(results),
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
    )


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


# =============================================================================
# Camera Compatibility Routes
# =============================================================================


def verify_admin(x_admin_password: Annotated[str | None, Header()] = None) -> bool:
    """Verify admin password from header using timing-safe comparison."""
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")

    # Handle None password safely and use timing-safe comparison
    if x_admin_password is None or not secrets.compare_digest(
        x_admin_password.encode('utf-8'),
        settings.admin_password.get_secret_value().encode('utf-8')
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


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


@app.get("/admin", response_class=HTMLResponse)
async def admin_index(request: Request):
    """Admin index page with links to all admin pages."""
    return templates.TemplateResponse("admin_index.html", {"request": request})


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
    synthesis_start = time.time()
    if not chunks_for_synthesis:
        llm_response = "No relevant information found for your query."
    else:
        response = await get_anthropic().messages.create(
            model="claude-haiku-4-5-20251001",
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
        model="claude-haiku-4-5-20251001",
    )


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app
