"""Support case tools for CLORAG MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def _case_to_dict(case: Any, include_document: bool = False) -> dict[str, Any]:
    """Convert SupportCase model to serializable dict."""
    result = {
        "id": case.id,
        "thread_id": case.thread_id,
        "subject": case.subject,
        "status": case.status.value,
        "resolution_quality": case.resolution_quality.value if case.resolution_quality else None,
        "problem_summary": case.problem_summary,
        "solution_summary": case.solution_summary,
        "keywords": case.keywords,
        "category": case.category,
        "product": case.product,
        "messages_count": case.messages_count,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "resolved_at": case.resolved_at.isoformat() if case.resolved_at else None,
    }

    if include_document:
        result["document"] = case.document

    return result


def register_support_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register support case MCP tools.

    Args:
        mcp: FastMCP server instance to register tools on.
    """

    @mcp.tool()
    def search_support_cases(query: str, limit: int = 10) -> dict[str, Any]:
        """Search support cases using full-text search.

        Searches across subject, problem summary, solution summary,
        full document content, and keywords using BM25 ranking.

        Args:
            query: Search query describing the issue or topic.
            limit: Maximum results (1-20, default 10).

        Returns:
            Matching support cases with summaries and metadata.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        limit = max(1, min(20, limit))

        cases = services.support_case_db.search_cases(query, limit=limit)

        return {
            "query": query,
            "total_found": len(cases),
            "cases": [_case_to_dict(c) for c in cases],
        }

    @mcp.tool()
    def get_support_case(case_id: str) -> dict[str, Any]:
        """Get full details of a support case.

        Includes the complete anonymized document with problem description
        and resolution details.

        Args:
            case_id: Support case ID.

        Returns:
            Full support case with document content.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        case = services.support_case_db.get_case_by_id(case_id)

        if not case:
            return {"error": f"Support case with ID {case_id} not found"}

        return {"case": _case_to_dict(case, include_document=True)}

    @mcp.tool()
    def list_support_cases(
        category: str | None = None,
        product: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List support cases with optional filtering.

        Args:
            category: Filter by category (e.g., "troubleshooting", "configuration").
            product: Filter by product (e.g., "RIO", "CI0").
            limit: Maximum cases to return (1-50, default 20).
            offset: Number of cases to skip for pagination.

        Returns:
            List of support cases with summaries.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        limit = max(1, min(50, limit))
        offset = max(0, offset)

        cases, total = services.support_case_db.list_cases(
            category=category,
            product=product,
            limit=limit,
            offset=offset,
        )

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "category": category,
                "product": product,
            },
            "cases": [_case_to_dict(c) for c in cases],
        }

    @mcp.tool()
    def get_support_case_raw(case_id: str) -> dict[str, Any]:
        """Get the raw anonymized thread content of a support case.

        Returns the full email thread as-is (not just the
        problem/solution summaries). Useful for seeing the
        complete conversation flow, quoted replies, and
        original context.

        Args:
            case_id: Support case ID.

        Returns:
            Raw thread content with case metadata.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        case = services.support_case_db.get_case_by_id(case_id)

        if not case:
            return {
                "error": f"Support case '{case_id}' not found",
            }

        raw_thread = services.support_case_db.get_raw_thread(
            case_id,
        )

        return {
            "case_id": case.id,
            "thread_id": case.thread_id,
            "subject": case.subject,
            "raw_thread": raw_thread or case.document,
            "has_raw_thread": raw_thread is not None,
            "messages_count": case.messages_count,
        }

    @mcp.tool()
    def get_support_stats() -> dict[str, Any]:
        """Get statistics about the support case database.

        Returns:
            Statistics including total count, categories, products, and quality.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        stats = services.support_case_db.get_stats()

        return {
            "total_cases": stats.get("total", 0),
            "by_category": stats.get("by_category", {}),
            "by_product": stats.get("by_product", {}),
            "by_quality": stats.get("by_quality", {}),
        }
