"""CLORAG MCP Server - Exposes RAG capabilities to Claude Desktop and other MCP clients."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from clorag.core.database import get_camera_database
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.reranker import RerankerClient
from clorag.core.retriever import MultiSourceRetriever
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.support_case_db import get_support_case_database
from clorag.core.vectorstore import VectorStore
from clorag.mcp.tools import (
    register_camera_tools,
    register_document_tools,
    register_odoo_tools,
    register_search_tools,
    register_support_tools,
)
from clorag.services.custom_docs import CustomDocumentService


class MCPServices:
    """Container for CLORAG services used by MCP tools."""

    def __init__(self) -> None:
        """Initialize all services."""
        # Initialize vector search components
        self.embeddings = EmbeddingsClient()
        self.sparse_embeddings = SparseEmbeddingsClient()
        self.vectorstore = VectorStore()
        self.reranker = RerankerClient()

        # Initialize retriever with reranking
        self.retriever = MultiSourceRetriever(
            embeddings_client=self.embeddings,
            sparse_embeddings_client=self.sparse_embeddings,
            vector_store=self.vectorstore,
            reranker_client=self.reranker,
        )

        # Initialize database services
        self.camera_db = get_camera_database()
        self.support_case_db = get_support_case_database()

        # Initialize document service
        self.document_service = CustomDocumentService(
            vectorstore=self.vectorstore,
            embeddings=self.embeddings,
            sparse_embeddings=self.sparse_embeddings,
        )


# Global services instance (initialized in lifespan)
_services: MCPServices | None = None


def get_services() -> MCPServices:
    """Get the initialized services instance."""
    if _services is None:
        raise RuntimeError("MCP services not initialized. Run server with lifespan.")
    return _services


@asynccontextmanager
async def lifespan(app: FastMCP[MCPServices]) -> AsyncIterator[MCPServices]:
    """Initialize and cleanup services."""
    global _services
    _services = MCPServices()
    try:
        yield _services
    finally:
        # Cleanup
        _services.camera_db.close()
        _services.support_case_db.close()
        _services = None


def create_mcp_server() -> FastMCP[MCPServices]:
    """Create and configure the CLORAG MCP server.

    Returns:
        Configured FastMCP server instance.
    """
    mcp = FastMCP[MCPServices](
        name="clorag",
        instructions=(
            "CLORAG is a Multi-RAG knowledge base for Cyanview support. "
            "It combines Docusaurus documentation, curated Gmail support cases, "
            "and custom admin-managed documents. Use the search tools to find "
            "relevant information about Cyanview products, camera compatibility, "
            "troubleshooting, and integration guides."
        ),
        lifespan=lifespan,
    )

    # Register all tools
    register_search_tools(mcp)
    register_camera_tools(mcp)
    register_document_tools(mcp)
    register_support_tools(mcp)
    register_odoo_tools(mcp)  # Only registered if odoo_mcp_enabled=True

    return mcp


def main() -> None:
    """Entry point for the MCP server."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
