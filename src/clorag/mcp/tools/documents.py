"""Custom document management tools for CLORAG MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def register_document_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register document-related MCP tools.

    Args:
        mcp: FastMCP server instance to register tools on.
    """

    @mcp.tool()
    async def list_documents(
        category: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List custom documents in the knowledge base.

        Custom documents are admin-managed content including product info,
        troubleshooting guides, FAQs, release notes, and internal documentation.

        Args:
            category: Filter by category. Valid categories:
                - product_info, troubleshooting, configuration
                - firmware, release_notes, faq, best_practices
                - pre_sales, internal, other
            limit: Maximum documents to return (1-50, default 20).
            offset: Number of documents to skip for pagination.

        Returns:
            List of documents with metadata and preview.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        limit = max(1, min(50, limit))
        offset = max(0, offset)

        docs, total = await services.document_service.list_documents(
            limit=limit,
            offset=offset,
            category=category,
            include_expired=False,
        )

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "category_filter": category,
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "category": doc.category.value,
                    "tags": doc.tags,
                    "content_preview": doc.content_preview,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "is_expired": doc.is_expired,
                }
                for doc in docs
            ],
        }

    @mcp.tool()
    async def get_document(doc_id: str) -> dict[str, Any]:
        """Get full content of a custom document.

        Args:
            doc_id: Document ID (UUID format).

        Returns:
            Full document content with metadata.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        doc = await services.document_service.get_document(doc_id)

        if not doc:
            return {"error": f"Document with ID {doc_id} not found"}

        return {
            "document": {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "category": doc.category.value,
                "tags": doc.tags,
                "url_reference": doc.url_reference,
                "notes": doc.notes,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "created_by": doc.created_by,
            }
        }

    @mcp.tool()
    async def get_document_categories() -> dict[str, Any]:
        """Get available document categories.

        Returns:
            List of valid categories for filtering and document creation.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        categories = await services.document_service.get_categories()

        return {"categories": categories}
