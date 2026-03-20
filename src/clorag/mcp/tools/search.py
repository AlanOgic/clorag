"""RAG search tools for CLORAG MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from clorag.core.retriever import SearchSource

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def register_search_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register search-related MCP tools.

    Args:
        mcp: FastMCP server instance to register tools on.
    """

    @mcp.tool()
    async def search(
        query: str,
        source: str = "both",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Search CLORAG knowledge base using hybrid RAG with reranking.

        Combines semantic (AI) search with BM25 keyword matching for optimal results.
        Results are reranked using Voyage's cross-encoder for improved relevance.

        Args:
            query: Search query - be specific for best results.
            source: Source to search:
                - "docs" - Official Docusaurus documentation only
                - "cases" - Gmail support cases and examples
                - "custom" - Admin-managed custom documents
                - "both" - Search all sources (recommended)
            limit: Maximum number of results to return (1-20, default 5).

        Returns:
            Search results with document text, source, score, and metadata.
        """
        from clorag.mcp.server import get_services

        services = get_services()

        # Map source string to enum
        source_map = {
            "docs": SearchSource.DOCS,
            "cases": SearchSource.CASES,
            "custom": SearchSource.CUSTOM,
            "both": SearchSource.HYBRID,
        }
        search_source = source_map.get(source.lower(), SearchSource.HYBRID)

        # Clamp limit
        limit = max(1, min(20, limit))

        # Execute search
        result = await services.retriever.retrieve(
            query=query,
            source=search_source,
            limit=limit,
        )

        # Format results
        results = []
        for r in result.results:
            results.append({
                "text": r.text,
                "score": round(r.score, 4),
                "source": r.payload.get("_source", search_source.value),
                "title": r.payload.get("title", ""),
                "url": r.payload.get("url", ""),
                "category": r.payload.get("category", ""),
            })

        return {
            "query": query,
            "source": search_source.value,
            "total_found": result.total_found,
            "reranked": result.reranked,
            "results": results,
        }
