"""Search analytics tools for CLORAG MCP server.

Provides tools to monitor search quality and usage patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def register_analytics_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register search analytics MCP tools."""

    @mcp.tool()
    def get_search_quality(
        days: int = 30,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get recent low-quality searches for review and improvement.

        Identifies queries with average relevance score below 0.3,
        indicating potential gaps in the knowledge base.

        Args:
            days: Look back period in days (default 30).
            limit: Maximum results (1-100, default 50).

        Returns:
            Low-scoring queries with scores and timestamps.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        limit = max(1, min(100, limit))
        days = max(1, min(365, days))

        searches = services.analytics_db.get_low_quality_searches(
            limit=limit, days=days,
        )

        return {
            "threshold": 0.3,
            "days": days,
            "count": len(searches),
            "searches": searches,
        }

    @mcp.tool()
    def get_search_stats(
        days: int = 30,
    ) -> dict[str, Any]:
        """Get search analytics: total queries, response times, popular queries.

        Args:
            days: Look back period in days (default 30).

        Returns:
            Search statistics and top queries.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        days = max(1, min(365, days))

        stats = services.analytics_db.get_search_stats(days=days)
        popular = services.analytics_db.get_popular_queries(limit=10, days=days)

        return {
            "days": days,
            "stats": stats,
            "popular_queries": popular,
        }
