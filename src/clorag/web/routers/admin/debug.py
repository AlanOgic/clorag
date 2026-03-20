"""Admin debug endpoints.

Provides search debugging with detailed chunk and LLM information.
"""

import time
from typing import Any

from fastapi import APIRouter, Depends

from clorag.config import get_settings
from clorag.services.prompt_manager import get_composed_prompt
from clorag.web.auth import verify_admin
from clorag.web.schemas import DebugSearchResponse, SearchRequest
from clorag.web.search import (
    build_context,
    get_anthropic,
    perform_search,
)

router = APIRouter(tags=["Debug"])


@router.post("/search-debug", response_model=DebugSearchResponse)
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
    context = build_context(chunks_for_synthesis, graph_context=graph_context)
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
            system=get_composed_prompt("base.system_prompt", "synthesis.web_layer"),
            messages=[{"role": "user", "content": user_prompt}],
        )
        content_block = response.content[0]
        llm_response = (
            str(content_block.text)
            if hasattr(content_block, "text")
            else str(content_block)
        )
    synthesis_time_ms = int((time.time() - synthesis_start) * 1000)

    total_time_ms = int((time.time() - total_start) * 1000)

    # Build detailed chunk info
    detailed_chunks: list[dict[str, Any]] = []
    for i, (result, chunk) in enumerate(zip(results, chunks_for_synthesis)):
        detailed_chunks.append(
            {
                "index": i + 1,
                "score": result.score,
                "source_type": chunk.get("source_type", "unknown"),
                "title": chunk.get("title") or chunk.get("subject", "Untitled"),
                "url": chunk.get("url"),
                "text": chunk.get("text", "")[:3000],  # Limit text size
                "text_length": len(chunk.get("text", "")),
            }
        )

    return DebugSearchResponse(
        query=req.query,
        source=req.source.value,
        retrieval_time_ms=retrieval_time_ms,
        synthesis_time_ms=synthesis_time_ms,
        total_time_ms=total_time_ms,
        chunks=detailed_chunks,
        llm_prompt=user_prompt,
        system_prompt=get_composed_prompt("base.system_prompt", "synthesis.web_layer"),
        llm_response=llm_response,
        model=settings.sonnet_model,
    )
