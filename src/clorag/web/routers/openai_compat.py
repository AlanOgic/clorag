"""OpenAI-compatible API endpoint.

Provides /v1/chat/completions and /v1/models endpoints that follow the
OpenAI API format, allowing external services to query CLORAG using
any OpenAI SDK client.

Auth: Bearer token via OPENAI_COMPAT_API_KEY environment variable.
"""

import json
import secrets
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

import structlog
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from clorag.config import get_settings
from clorag.web.dependencies import limiter
from clorag.web.schemas import SearchRequest, SearchSource
from clorag.web.search import (
    extract_source_links,
    perform_search,
    synthesize_answer,
    synthesize_answer_stream,
)

router = APIRouter()
logger = structlog.get_logger()

API_KEY_HEADER = "Authorization"


# =============================================================================
# Pydantic models (OpenAI format)
# =============================================================================


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = "clorag"
    messages: list[ChatMessage]
    stream: bool = False
    # Accepted but ignored — CLORAG controls its own synthesis parameters
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    user: str | None = None


# =============================================================================
# Auth helper
# =============================================================================


def _verify_api_key(authorization: str | None) -> JSONResponse | None:
    """Verify Bearer token against OPENAI_COMPAT_API_KEY.

    Returns a JSONResponse error if auth fails, or None if auth succeeds.
    """
    settings = get_settings()

    if settings.openai_compat_api_key is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": (
                        "OpenAI-compatible API is not configured."
                        " Set OPENAI_COMPAT_API_KEY."
                    ),
                    "type": "server_error",
                    "code": "api_not_configured",
                }
            },
        )

    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Missing or malformed Authorization header. Use: Bearer <api_key>",
                    "type": "authentication_error",
                    "code": "missing_api_key",
                }
            },
        )

    provided_key = authorization.removeprefix("Bearer ").strip()
    expected_key = settings.openai_compat_api_key.get_secret_value()

    if not secrets.compare_digest(provided_key.encode("utf-8"), expected_key.encode("utf-8")):
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Invalid API key.",
                    "type": "authentication_error",
                    "code": "invalid_api_key",
                }
            },
        )

    return None


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/v1/models")
@limiter.limit("20/minute")
async def list_models(
    request: Request,
    authorization: str | None = Header(None),
) -> JSONResponse:
    """List available models (OpenAI-compatible)."""
    auth_error = _verify_api_key(authorization)
    if auth_error:
        return auth_error

    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": "clorag",
                "object": "model",
                "created": 1700000000,
                "owned_by": "cyanview",
            }
        ],
    })


@router.post("/v1/chat/completions", response_model=None)
@limiter.limit("20/minute")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    authorization: str | None = Header(None),
) -> JSONResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Extracts the last user message as the search query, routes it through
    the full RAG pipeline, and returns the response in OpenAI format.
    """
    auth_error = _verify_api_key(authorization)
    if auth_error:
        return auth_error

    # Validate messages
    if not body.messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "messages must contain at least one message.",
                    "type": "invalid_request_error",
                    "code": "invalid_messages",
                }
            },
        )

    # Extract last user message as query
    user_messages = [m for m in body.messages if m.role == "user"]
    if not user_messages:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "messages must contain at least one user message.",
                    "type": "invalid_request_error",
                    "code": "no_user_message",
                }
            },
        )

    query = user_messages[-1].content

    # Build conversation history from prior user/assistant pairs (exclude last msg)
    conversation_history: list[dict[str, str]] = []
    for msg in body.messages[:-1]:
        if msg.role in ("user", "assistant"):
            conversation_history.append({"role": msg.role, "content": msg.content})

    # Perform RAG search
    search_req = SearchRequest(
        query=query, source=SearchSource.BOTH, limit=10, session_id=None,
    )
    _, chunks_for_synthesis, graph_context, _ = await perform_search(search_req)

    # Extract sources for appending to response
    source_links = cast(
        list[dict[str, Any]], extract_source_links(chunks_for_synthesis),
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.stream:
        return StreamingResponse(
            _stream_response(
                completion_id,
                query,
                chunks_for_synthesis,
                conversation_history if conversation_history else None,
                graph_context,
                source_links,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming: synthesize full answer
    answer = await synthesize_answer(
        query,
        chunks_for_synthesis,
        conversation_history if conversation_history else None,
        graph_context,
    )

    # Append sources
    answer_with_sources = _append_sources(answer, source_links)

    return JSONResponse(content={
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "clorag",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_with_sources,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    })


# =============================================================================
# Helpers
# =============================================================================


def _append_sources(answer: str, source_links: list[dict[str, Any]]) -> str:
    """Append source links as markdown to the answer text."""
    if not source_links:
        return answer

    sources_md = "\n\n---\n**Sources:**"
    for link in source_links:
        title = link.get("title", "Source")
        url = link.get("url")
        if url:
            sources_md += f"\n- [{title}]({url})"
        else:
            sources_md += f"\n- {title}"

    return answer + sources_md


async def _stream_response(
    completion_id: str,
    query: str,
    chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None,
    graph_context: str | None,
    source_links: list[dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Stream response in OpenAI SSE format."""
    created = int(time.time())

    # First chunk: role
    yield _sse_chunk(completion_id, created, {"role": "assistant", "content": ""})

    # Stream content
    async for text in synthesize_answer_stream(
        query, chunks, conversation_history, graph_context
    ):
        yield _sse_chunk(completion_id, created, {"content": text})

    # Append sources as final content chunks
    if source_links:
        sources_md = "\n\n---\n**Sources:**"
        yield _sse_chunk(completion_id, created, {"content": sources_md})
        for link in source_links:
            title = link.get("title", "Source")
            url = link.get("url")
            if url:
                line = f"\n- [{title}]({url})"
            else:
                line = f"\n- {title}"
            yield _sse_chunk(completion_id, created, {"content": line})

    # Final chunk: finish_reason
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "clorag",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _sse_chunk(completion_id: str, created: int, delta: dict[str, str]) -> str:
    """Format a single SSE chunk in OpenAI format."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "clorag",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"
