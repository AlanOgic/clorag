"""Search API endpoints.

Provides public search endpoints with streaming and non-streaming Claude synthesis.
"""

import json
import time
from collections.abc import AsyncGenerator

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from clorag.web.auth import get_session_store
from clorag.web.dependencies import get_analytics_db, get_templates, limiter
from clorag.web.schemas import SearchRequest, SearchResponse
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
    templates = get_templates()
    return templates.TemplateResponse("index.html", {"request": request})


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
        source_links=source_links,  # type: ignore[arg-type]
        results=results,
        total=len(results),
        session_id=session.session_id,
    )
