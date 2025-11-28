"""Custom MCP tools for the CLORAG agent."""

from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from clorag.core.embeddings import EmbeddingsClient
from clorag.core.retriever import MultiSourceRetriever, SearchSource
from clorag.core.vectorstore import VectorStore

# Initialize clients (will be replaced with proper DI in production)
_embeddings_client: EmbeddingsClient | None = None
_vector_store: VectorStore | None = None
_retriever: MultiSourceRetriever | None = None


def _get_retriever() -> MultiSourceRetriever:
    """Get or create the retriever instance."""
    global _embeddings_client, _vector_store, _retriever
    if _retriever is None:
        _embeddings_client = EmbeddingsClient()
        _vector_store = VectorStore()
        _retriever = MultiSourceRetriever(_embeddings_client, _vector_store)
    return _retriever


@tool(
    name="search_docs",
    description=(
        "Search the documentation knowledge base for relevant information. "
        "Use this when the user asks questions about product features, how-to guides, "
        "technical specifications, or official documentation."
    ),
    input_schema={
        "query": str,
        "limit": int,
    },
)
async def search_docs(args: dict[str, Any]) -> dict[str, Any]:
    """Search documentation for relevant content.

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
    result = await retriever.retrieve_docs(query=query, limit=limit)
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
        "Search past support cases and examples from Gmail threads. "
        "Use this when looking for similar issues, past resolutions, "
        "or real-world examples of how problems were solved."
    ),
    input_schema={
        "query": str,
        "limit": int,
    },
)
async def search_cases(args: dict[str, Any]) -> dict[str, Any]:
    """Search Gmail support cases for similar examples.

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
    result = await retriever.retrieve_cases(query=query, limit=limit)
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
    name="hybrid_search",
    description=(
        "Search across both documentation AND support cases simultaneously. "
        "Use this for comprehensive answers that combine official documentation "
        "with real-world examples and past resolutions."
    ),
    input_schema={
        "query": str,
        "limit": int,
    },
)
async def hybrid_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search both documentation and cases, merging results by relevance.

    Args:
        args: Dictionary with 'query' (required) and 'limit' (optional, default 5).

    Returns:
        Tool result with merged results from both sources.
    """
    query = args.get("query", "")
    limit = args.get("limit", 5)

    if not query:
        return {
            "content": [{"type": "text", "text": "Error: Query is required"}],
            "isError": True,
        }

    retriever = _get_retriever()
    result = await retriever.retrieve(
        query=query,
        source=SearchSource.HYBRID,
        limit=limit,
    )
    context = retriever.format_context(result)

    return {
        "content": [
            {
                "type": "text",
                "text": f"Found {result.total_found} relevant results (docs + cases):\n\n{context}",
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
        version="1.0.0",
        tools=[search_docs, search_cases, hybrid_search],
    )
