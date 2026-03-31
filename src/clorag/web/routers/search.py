"""Search API endpoints.

Provides public search endpoints with streaming and non-streaming Claude synthesis.
"""

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from clorag.web.auth import get_session_store
from clorag.web.dependencies import get_analytics_db, get_templates, limiter
from clorag.web.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    SearchRequest,
    SearchResponse,
)
from clorag.web.search import (
    extract_source_links,
    perform_search,
    synthesize_answer,
    synthesize_answer_stream,
)

router = APIRouter()
logger = structlog.get_logger()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> Response:
    """Render the search home page."""
    from clorag.core.messages_db import get_messages_database

    templates = get_templates()
    try:
        db = get_messages_database()
        messages = [m.to_dict() for m in db.get_active_messages()]
    except Exception:
        messages = []
    return templates.TemplateResponse("index.html", {"request": request, "messages": messages})


@router.get("/api/messages")
async def api_public_messages() -> list[dict[str, Any]]:
    """Get active messages for public display."""
    from clorag.core.messages_db import get_messages_database

    db = get_messages_database()
    return [m.to_dict() for m in db.get_active_messages()]


@router.get("/robots.txt")
async def robots_txt() -> Response:
    """Block search engine crawlers from indexing."""
    return Response(
        content="User-agent: *\nDisallow: /\n",
        media_type="text/plain",
    )


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "1.0.0"}


@router.get("/legacy", response_class=HTMLResponse)
async def legacy_home(request: Request) -> Response:
    """Render the legacy docs-only search page."""
    templates = get_templates()
    return templates.TemplateResponse("legacy.html", {"request": request})


@router.post("/api/legacy/stream")
@limiter.limit("30/minute")
async def legacy_search_stream(request: Request, req: SearchRequest) -> StreamingResponse:
    """Perform docs-only RAG search with streaming for the legacy page.

    Searches the separate docusaurus_docs_legacy collection (support.cyanview.com).
    Source links point to support.cyanview.com (no URL rewriting).
    """
    from clorag.config import get_settings
    from clorag.web.search.pipeline import (
        generate_embeddings_parallel,
        get_reranker,
        get_vectorstore,
    )

    start_time = time.time()
    settings = get_settings()
    legacy_collection = settings.qdrant_legacy_docs_collection

    # Get or create conversation session
    session = get_session_store().get_or_create_session(req.session_id)
    conversation_history = session.get_context_messages()

    # Search the legacy collection directly
    vs = get_vectorstore()
    dense_vector, sparse_vector = await generate_embeddings_parallel(req.query)

    fetch_limit = max(req.limit * 3, 15)
    docs = await vs.search_hybrid_rrf(
        collection=legacy_collection,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=fetch_limit,
    )

    chunks_for_synthesis: list[dict[str, Any]] = []
    for doc in docs:
        chunks_for_synthesis.append({
            "text": doc.payload.get("text", ""),
            "source_type": "documentation",
            "url": doc.payload.get("url"),
            "title": doc.payload.get("title", "Untitled"),
            "score": doc.score,
        })

    # Apply reranking if enabled
    was_reranked = False
    if settings.rerank_enabled and chunks_for_synthesis:
        try:
            reranker = get_reranker()
            texts_to_rerank = [c.get("text", "") for c in chunks_for_synthesis]
            rerank_response = reranker.rerank(
                query=req.query,
                documents=texts_to_rerank,
                top_k=settings.rerank_top_k,
            )
            reranked: list[dict[str, Any]] = []
            for rr in rerank_response.results:
                idx = rr.index
                if idx < len(chunks_for_synthesis):
                    chunk = {**chunks_for_synthesis[idx], "score": rr.relevance_score}
                    reranked.append(chunk)
            chunks_for_synthesis = reranked
            was_reranked = True
        except Exception:
            chunks_for_synthesis = chunks_for_synthesis[:req.limit]

    # No URL rewriting — links stay as support.cyanview.com
    source_links = extract_source_links(chunks_for_synthesis, rewrite_urls=False)

    search_start_time = start_time

    async def generate() -> AsyncGenerator[str, None]:
        collected_response: list[str] = []
        try:
            yield f"data: {json.dumps({'type': 'session', 'session_id': session.session_id})}\n\n"

            async for chunk in synthesize_answer_stream(
                req.query, chunks_for_synthesis, conversation_history, None
            ):
                collected_response.append(chunk)
                yield f"data: {json.dumps({'type': 'text', 'text': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'sources', 'sources': source_links})}\n\n"

            full_response = "".join(collected_response)
            search_id = 0
            response_time_ms = int((time.time() - search_start_time) * 1000)
            try:
                search_id = get_analytics_db().log_search(
                    query=req.query,
                    source="legacy_docs",
                    response_time_ms=response_time_ms,
                    results_count=len(chunks_for_synthesis),
                    response=full_response,
                    chunks=chunks_for_synthesis,
                    session_id=session.session_id,
                    reranked=was_reranked,
                    scores=[c.get("score", 0) for c in chunks_for_synthesis],
                    source_types=[c.get("source_type", "unknown") for c in chunks_for_synthesis],
                )
            except Exception as e:
                logger.warning("Failed to log legacy search analytics", error=str(e))

            yield f"data: {json.dumps({'type': 'done', 'search_id': search_id})}\n\n"

            session.add_exchange(req.query, full_response)
        except Exception as e:
            logger.error("Error during legacy streaming response", error=str(e), query=req.query)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/api/search/stream")
@limiter.limit("30/minute")
async def search_stream(request: Request, req: SearchRequest) -> StreamingResponse:
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

    async def generate() -> AsyncGenerator[str, None]:
        collected_response: list[str] = []
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

            # Assemble full response and log to analytics BEFORE done event
            # so we can include search_id for feedback linking
            full_response = "".join(collected_response)
            search_id = 0
            response_time_ms = int((time.time() - search_start_time) * 1000)
            try:
                search_id = get_analytics_db().log_search(
                    query=req.query,
                    source=req.source.value,
                    response_time_ms=response_time_ms,
                    results_count=len(chunks_for_synthesis),
                    response=full_response,
                    chunks=chunks_for_synthesis,
                    session_id=session.session_id,
                    reranked=was_reranked,
                    scores=[c.get("score", 0) for c in chunks_for_synthesis],
                    source_types=[c.get("source_type", "unknown") for c in chunks_for_synthesis],
                )
            except Exception as e:
                logger.warning("Failed to log search analytics", error=str(e))

            # Signal completion with search_id for feedback
            yield f"data: {json.dumps({'type': 'done', 'search_id': search_id})}\n\n"

            # Store this exchange in the session for future follow-ups
            session.add_exchange(req.query, full_response)
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


@router.post("/api/search", response_model=SearchResponse)
@limiter.limit("30/minute")
async def search(request: Request, req: SearchRequest) -> SearchResponse:
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

    # Log search to analytics to get search_id for feedback
    search_id = 0
    response_time_ms = int((time.time() - start_time) * 1000)
    try:
        search_id = get_analytics_db().log_search(
            query=req.query,
            source=req.source.value,
            response_time_ms=response_time_ms,
            results_count=len(results),
            response=answer,
            chunks=chunks_for_synthesis,
            session_id=session.session_id,
            reranked=was_reranked,
            scores=[r.score for r in results],
            source_types=[r.source for r in results],
        )
    except Exception as e:
        logger.warning("Failed to log search analytics", error=str(e))

    return SearchResponse(
        query=req.query,
        source=req.source.value,
        answer=answer,
        source_links=source_links,  # type: ignore[arg-type]
        results=results,
        total=len(results),
        session_id=session.session_id,
        search_id=search_id,
    )


@router.post("/api/feedback", response_model=FeedbackResponse)
@limiter.limit("10/minute")
async def submit_feedback(request: Request, req: FeedbackRequest) -> FeedbackResponse:
    """Submit thumbs up/down feedback for a search answer.

    One feedback per search_id — submitting again replaces the previous vote.
    Rate limited to 10 requests per minute per IP.
    """
    if req.search_id <= 0:
        return FeedbackResponse(success=False, search_id=req.search_id, rating=req.rating)

    try:
        get_analytics_db().save_feedback(
            search_id=req.search_id,
            rating=req.rating,
            comment=req.comment,
            session_id=req.session_id,
        )
        return FeedbackResponse(success=True, search_id=req.search_id, rating=req.rating)
    except Exception as e:
        logger.warning("Failed to save feedback", error=str(e), search_id=req.search_id)
        return FeedbackResponse(success=False, search_id=req.search_id, rating=req.rating)
