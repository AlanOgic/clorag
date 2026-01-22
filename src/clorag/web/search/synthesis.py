"""Answer synthesis functions using Claude.

This module provides functions for generating answers from retrieved chunks
using Claude, with support for both streaming and non-streaming responses.
"""

from collections.abc import AsyncGenerator
from typing import Any

import anthropic

from clorag.config import get_settings
from clorag.services.prompt_manager import get_prompt
from clorag.web.search.utils import build_context

# Lazy-loaded Anthropic client
_anthropic_client: anthropic.AsyncAnthropic | None = None


def get_anthropic() -> anthropic.AsyncAnthropic:
    """Get or create Anthropic client singleton."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


async def synthesize_answer(
    query: str,
    chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]] | None = None,
    graph_context: str | None = None,
) -> str:
    """Use Claude to synthesize an answer from retrieved chunks.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        conversation_history: Optional list of previous messages for follow-up context.
        graph_context: Optional graph enrichment context string.

    Returns:
        Synthesized answer string.
    """
    if not chunks:
        return "No relevant information found for your query."

    settings = get_settings()
    context = build_context(chunks, graph_context=graph_context)

    # Build messages with conversation history
    messages: list[dict[str, Any]] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"})

    response = await get_anthropic().messages.create(
        model=settings.sonnet_model,
        max_tokens=1500,
        system=get_prompt("synthesis.web_answer"),
        messages=messages,
    )
    # Extract text from response
    content_block = response.content[0]
    if hasattr(content_block, "text"):
        return str(content_block.text)
    return str(content_block)


async def synthesize_answer_stream(
    query: str,
    chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]] | None = None,
    graph_context: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream answer synthesis using Claude.

    Args:
        query: The user's question.
        chunks: Retrieved context chunks.
        conversation_history: Optional list of previous messages for follow-up context.
        graph_context: Optional graph enrichment context string.

    Yields:
        Text chunks from the streaming response.
    """
    if not chunks:
        yield "No relevant information found for your query."
        return

    settings = get_settings()
    context = build_context(chunks, graph_context=graph_context)

    # Build messages with conversation history
    messages: list[dict[str, Any]] = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"})

    async with get_anthropic().messages.stream(
        model=settings.sonnet_model,
        max_tokens=1500,
        system=get_prompt("synthesis.web_answer"),
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text
