"""FastAPI web application for AI Search with Claude synthesis."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from clorag.config import get_settings
from clorag.core.database import CameraDatabase, get_camera_database
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore
from clorag.models.camera import Camera, CameraCreate, CameraSource, CameraUpdate

app = FastAPI(title="Cyanview AI Search", version="1.0.0")

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
        _anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _anthropic


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

    query: str
    source: SearchSource = SearchSource.BOTH
    limit: int = 10


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


@app.post("/api/search/stream")
async def search_stream(req: SearchRequest):
    """Perform RAG search with streaming Claude-powered answer synthesis.

    Uses hybrid search (dense + sparse vectors) with RRF fusion for better
    results on queries with specific model numbers or technical terms.
    """
    vs = get_vectorstore()
    emb = get_embeddings()
    sparse_emb = get_sparse_embeddings()

    # Generate both dense and sparse query embeddings
    dense_vector = await emb.embed_query(req.query)
    sparse_vector = sparse_emb.embed_query(req.query)

    chunks_for_synthesis: list[dict] = []

    if req.source == SearchSource.DOCS:
        docs = await vs.search_docs_hybrid(dense_vector, sparse_vector, limit=req.limit)
        for doc in docs:
            chunks_for_synthesis.append({
                "text": doc.payload.get("text", ""),
                "source_type": "documentation",
                "url": doc.payload.get("url"),
                "title": doc.payload.get("title", "Untitled"),
            })
    elif req.source == SearchSource.GMAIL:
        cases = await vs.search_cases_hybrid(dense_vector, sparse_vector, limit=req.limit)
        for case in cases:
            chunks_for_synthesis.append({
                "text": case.payload.get("text", ""),
                "source_type": "gmail_case",
                "subject": case.payload.get("subject", "Support Case"),
            })
    else:
        hybrid = await vs.hybrid_search_rrf(dense_vector, sparse_vector, limit=req.limit)
        for item in hybrid:
            source_type = item.payload.get("_source", "unknown")
            if source_type == "documentation":
                chunks_for_synthesis.append({
                    "text": item.payload.get("text", ""),
                    "source_type": "documentation",
                    "url": item.payload.get("url"),
                    "title": item.payload.get("title", "Untitled"),
                })
            else:
                chunks_for_synthesis.append({
                    "text": item.payload.get("text", ""),
                    "source_type": "gmail_case",
                    "subject": item.payload.get("subject", "Support Case"),
                })

    # Extract top 3 unique source links
    source_links = _extract_source_links(chunks_for_synthesis)

    async def generate():
        # Stream the answer first
        async for chunk in synthesize_answer_stream(req.query, chunks_for_synthesis):
            yield f"data: {json.dumps({'type': 'text', 'text': chunk})}\n\n"

        # Then send sources at the end
        yield f"data: {json.dumps({'type': 'sources', 'sources': source_links})}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

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

    # Generate synthesized answer using Claude Haiku
    answer = await synthesize_answer(req.query, chunks_for_synthesis)

    # Extract top 3 unique source links
    source_links = _extract_source_links(chunks_for_synthesis, as_model=True)

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
    """Verify admin password from header."""
    settings = get_settings()
    if not settings.admin_password:
        raise HTTPException(status_code=503, detail="Admin access not configured")
    if x_admin_password != settings.admin_password:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


@app.get("/cameras", response_class=HTMLResponse)
async def cameras_list(request: Request, manufacturer: str | None = None):
    """Render the public camera compatibility list."""
    db = get_camera_database()
    cameras = db.list_cameras(manufacturer=manufacturer)
    manufacturers = db.get_manufacturers()
    stats = db.get_stats()

    return templates.TemplateResponse(
        "cameras.html",
        {
            "request": request,
            "cameras": cameras,
            "manufacturers": manufacturers,
            "selected_manufacturer": manufacturer,
            "stats": stats,
        },
    )


@app.get("/api/cameras", response_model=list[Camera])
async def api_cameras_list(manufacturer: str | None = None):
    """Get all cameras as JSON."""
    db = get_camera_database()
    return db.list_cameras(manufacturer=manufacturer)


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
async def api_camera_create(
    camera: CameraCreate,
    _: bool = Depends(verify_admin),
):
    """Create a new camera entry."""
    db = get_camera_database()
    return db.create_camera(camera, CameraSource.MANUAL)


@app.put("/api/admin/cameras/{camera_id}", response_model=Camera)
async def api_camera_update(
    camera_id: int,
    updates: CameraUpdate,
    _: bool = Depends(verify_admin),
):
    """Update an existing camera."""
    db = get_camera_database()
    camera = db.update_camera(camera_id, updates)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@app.delete("/api/admin/cameras/{camera_id}")
async def api_camera_delete(
    camera_id: int,
    _: bool = Depends(verify_admin),
):
    """Delete a camera entry."""
    db = get_camera_database()
    if not db.delete_camera(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"status": "deleted", "id": camera_id}


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app
