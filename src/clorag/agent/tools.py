"""Custom MCP tools for the CLORAG agent with hybrid RRF search."""

import time
import uuid
from typing import Any

import structlog
from claude_agent_sdk import create_sdk_mcp_server, tool

from clorag.core.analytics_db import AnalyticsDatabase
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.retriever import MultiSourceRetriever, SearchSource
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.vectorstore import VectorStore

logger = structlog.get_logger()

# Initialize clients (will be replaced with proper DI in production)
_embeddings_client: EmbeddingsClient | None = None
_sparse_embeddings_client: SparseEmbeddingsClient | None = None
_vector_store: VectorStore | None = None
_retriever: MultiSourceRetriever | None = None

# One session id per CLI process, so all tool calls in a single `uv run clorag ...`
# invocation group together in the analytics Recent Conversations view.
_CLI_SESSION_ID = f"cli-{uuid.uuid4()}"
_analytics_db: AnalyticsDatabase | None = None


def _get_retriever() -> MultiSourceRetriever:
    """Get or create the retriever instance with hybrid RRF search support."""
    global _embeddings_client, _sparse_embeddings_client, _vector_store, _retriever
    if _retriever is None:
        _embeddings_client = EmbeddingsClient()
        _sparse_embeddings_client = SparseEmbeddingsClient()
        _vector_store = VectorStore()
        _retriever = MultiSourceRetriever(
            embeddings_client=_embeddings_client,
            sparse_embeddings_client=_sparse_embeddings_client,
            vector_store=_vector_store,
        )
    return _retriever


def _get_analytics_db() -> AnalyticsDatabase | None:
    """Lazy-load analytics DB; returns None if initialization fails."""
    global _analytics_db
    if _analytics_db is not None:
        return _analytics_db
    try:
        from clorag.config import get_settings
        settings = get_settings()
        _analytics_db = AnalyticsDatabase(db_path=settings.analytics_database_path)
    except Exception as e:
        logger.warning("cli_analytics_init_failed", error=str(e))
        _analytics_db = None
    return _analytics_db


def _log_tool_call(
    tool_name: str,
    query: str,
    source: str,
    result: Any,
    elapsed_ms: int,
) -> None:
    """Log a CLI-agent tool invocation to analytics_db.

    Each tool call becomes its own ``search_queries`` row tagged
    ``pipeline='cli_agent'`` so the admin UI can show the actual LLM-rewritten
    query that the agent chose. Failures are swallowed — analytics must never
    break the tool call.
    """
    db = _get_analytics_db()
    if db is None:
        return
    try:
        chunks: list[dict[str, Any]] = []
        scores: list[float] = []
        source_types: list[str] = []
        for r in getattr(result, "results", [])[:10]:
            text = (getattr(r, "text", "") or "")[:800]
            score = float(getattr(r, "score", 0.0) or 0.0)
            payload = getattr(r, "payload", None) or {}
            stype = payload.get("_source") or payload.get("source_type") or "unknown"
            chunks.append({"text": text, "source_type": stype, "score": score})
            scores.append(score)
            source_types.append(str(stype))
        db.log_search(
            query=query,
            source=source,
            response_time_ms=elapsed_ms,
            results_count=getattr(result, "total_found", len(chunks)),
            chunks=chunks,
            session_id=_CLI_SESSION_ID,
            reranked=bool(getattr(result, "reranked", False)),
            scores=scores,
            source_types=source_types,
            pipeline="cli_agent",
            tool_calls=[
                {
                    "tool": tool_name,
                    "query": query,
                    "result_count": getattr(result, "total_found", len(chunks)),
                }
            ],
        )
    except Exception as e:
        logger.debug("cli_tool_call_log_failed", tool=tool_name, error=str(e))


@tool(
    name="search_docs",
    description=(
        "Search the documentation knowledge base for relevant information using hybrid search. "
        "Combines semantic (AI) search with keyword matching for best results. "
        "Use this when the user asks questions about product features, how-to guides, "
        "technical specifications, or official documentation."
    ),
    input_schema={
        "query": str,
        "limit": int,
    },
)
async def search_docs(args: dict[str, Any]) -> dict[str, Any]:
    """Search documentation for relevant content using hybrid RRF search.

    Args:
        args: Dictionary with 'query' (required) and 'limit' (optional, default 5).

    Returns:
        Tool result with retrieved documentation.
    """
    query = args.get("query", "")
    limit = args.get("limit", 5)

    if not query:
        return {
            "content": [{"type": "text", "text": "Error: Query is required"}],
            "isError": True,
        }

    retriever = _get_retriever()
    t0 = time.time()
    result = await retriever.retrieve_docs(query=query, limit=limit)
    _log_tool_call("search_docs", query, "docs", result, int((time.time() - t0) * 1000))
    context = retriever.format_context(result)

    return {
        "content": [
            {
                "type": "text",
                "text": f"Found {result.total_found} relevant documents:\n\n{context}",
            }
        ]
    }


@tool(
    name="search_cases",
    description=(
        "Search past support cases and examples from Gmail threads using hybrid search. "
        "Combines semantic (AI) search with keyword matching for best results. "
        "Use this when looking for similar issues, past resolutions, "
        "or real-world examples of how problems were solved."
    ),
    input_schema={
        "query": str,
        "limit": int,
    },
)
async def search_cases(args: dict[str, Any]) -> dict[str, Any]:
    """Search Gmail support cases for similar examples using hybrid RRF search.

    Args:
        args: Dictionary with 'query' (required) and 'limit' (optional, default 5).

    Returns:
        Tool result with retrieved support cases.
    """
    query = args.get("query", "")
    limit = args.get("limit", 5)

    if not query:
        return {
            "content": [{"type": "text", "text": "Error: Query is required"}],
            "isError": True,
        }

    retriever = _get_retriever()
    t0 = time.time()
    result = await retriever.retrieve_cases(query=query, limit=limit)
    _log_tool_call("search_cases", query, "gmail", result, int((time.time() - t0) * 1000))
    context = retriever.format_context(result)

    return {
        "content": [
            {
                "type": "text",
                "text": f"Found {result.total_found} similar support cases:\n\n{context}",
            }
        ]
    }


@tool(
    name="search_custom",
    description=(
        "Search custom knowledge documents added by administrators. "
        "Use this when looking for internal documentation, product updates, "
        "pre-sales information, or admin-curated knowledge."
    ),
    input_schema={
        "query": str,
        "limit": int,
    },
)
async def search_custom(args: dict[str, Any]) -> dict[str, Any]:
    """Search custom documents for relevant content using hybrid RRF search.

    Args:
        args: Dictionary with 'query' (required) and 'limit' (optional, default 5).

    Returns:
        Tool result with retrieved custom documents.
    """
    query = args.get("query", "")
    limit = args.get("limit", 5)

    if not query:
        return {
            "content": [{"type": "text", "text": "Error: Query is required"}],
            "isError": True,
        }

    retriever = _get_retriever()
    t0 = time.time()
    result = await retriever.retrieve_custom(query=query, limit=limit)
    _log_tool_call("search_custom", query, "custom", result, int((time.time() - t0) * 1000))
    context = retriever.format_context(result)

    return {
        "content": [
            {
                "type": "text",
                "text": f"Found {result.total_found} relevant custom documents:\n\n{context}",
            }
        ]
    }


@tool(
    name="hybrid_search",
    description=(
        "Search across ALL knowledge sources simultaneously: documentation, support cases, "
        "and custom documents. Uses hybrid RRF (Reciprocal Rank Fusion) combining semantic "
        "AI search with BM25 keyword matching for optimal results. "
        "Use this for comprehensive answers that combine official documentation "
        "with real-world examples and curated knowledge."
    ),
    input_schema={
        "query": str,
        "limit": int,
    },
)
async def hybrid_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search all sources using hybrid RRF, merging results by relevance.

    Args:
        args: Dictionary with 'query' (required) and 'limit' (optional, default 5).

    Returns:
        Tool result with merged results from all sources.
    """
    query = args.get("query", "")
    limit = args.get("limit", 5)

    if not query:
        return {
            "content": [{"type": "text", "text": "Error: Query is required"}],
            "isError": True,
        }

    retriever = _get_retriever()
    t0 = time.time()
    result = await retriever.retrieve(
        query=query,
        source=SearchSource.HYBRID,
        limit=limit,
    )
    _log_tool_call("hybrid_search", query, "both", result, int((time.time() - t0) * 1000))
    context = retriever.format_context(result)

    # Include cache stats for debugging (can be removed in production)
    cache_stats = retriever.get_cache_stats()
    rerank_status = "on" if retriever.rerank_enabled else "off"
    dense_hit = cache_stats["dense"]["hit_rate_percent"]
    sparse_hit = cache_stats["sparse"]["hit_rate_percent"]
    rerank_hit = cache_stats["rerank"]["hit_rate_percent"]

    return {
        "content": [
            {
                "type": "text",
                "text": (
                    f"Found {result.total_found} relevant results "
                    f"(docs + cases + custom):\n\n{context}\n\n"
                    f"[Rerank: {rerank_status} | Cache: dense={dense_hit}%, "
                    f"sparse={sparse_hit}%, rerank={rerank_hit}%]"
                ),
            }
        ]
    }


def create_rag_tools_server() -> Any:
    """Create an MCP server with all RAG tools.

    Returns:
        SDK MCP server instance configured with RAG tools.
    """
    return create_sdk_mcp_server(
        name="clorag-tools",
        version="1.1.0",  # Bumped for hybrid RRF support
        tools=[search_docs, search_cases, search_custom, hybrid_search],
    )
