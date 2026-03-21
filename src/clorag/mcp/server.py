"""CLORAG MCP Server - Exposes RAG capabilities to Claude Desktop and other MCP clients."""

from __future__ import annotations

# Configure logging to stderr BEFORE any application imports.
# Module-level imports trigger database init and model loading that log messages.
# For stdio transport, stdout must stay clean for JSON-RPC protocol.
import logging
import sys

logging.basicConfig(format="%(message)s", stream=sys.stderr, level=logging.INFO, force=True)

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=False,
)

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from clorag.core.analytics_db import AnalyticsDatabase
from clorag.core.database import get_camera_database
from clorag.core.embeddings import EmbeddingsClient
from clorag.core.reranker import RerankerClient
from clorag.core.retriever import MultiSourceRetriever
from clorag.core.sparse_embeddings import SparseEmbeddingsClient
from clorag.core.support_case_db import get_support_case_database
from clorag.core.vectorstore import VectorStore
from clorag.mcp.prompts import register_prompts
from clorag.mcp.resources import register_resources
from clorag.mcp.tools import (
    register_analytics_tools,
    register_camera_tools,
    register_chunk_tools,
    register_document_tools,
    register_ingestion_tools,
    register_prompt_tools,
    register_search_tools,
    register_settings_tools,
    register_support_tools,
)
from clorag.services.custom_docs import CustomDocumentService
from clorag.services.prompt_manager import get_prompt_manager


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

        # Initialize analytics and prompt services
        from clorag.config import get_settings

        settings = get_settings()
        self.analytics_db = AnalyticsDatabase(settings.analytics_database_path)
        self.prompt_manager = get_prompt_manager()


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


def create_mcp_server(
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP[MCPServices]:
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
        host=host,
        port=port,
    )

    # Register all tools
    register_search_tools(mcp)
    register_camera_tools(mcp)
    register_document_tools(mcp)
    register_support_tools(mcp)
    register_chunk_tools(mcp)
    register_prompt_tools(mcp)
    register_settings_tools(mcp)
    register_analytics_tools(mcp)
    register_ingestion_tools(mcp)

    # Register resources and resource templates
    register_resources(mcp)

    # Register prompt templates
    register_prompts(mcp)

    return mcp


def main() -> None:
    """Entry point for the MCP server (stdio transport for local use)."""
    mcp = create_mcp_server()
    mcp.run(transport="stdio")


def main_http() -> None:
    """Entry point for the MCP server (streamable-http transport for Docker/remote)."""
    import os

    import anyio
    import structlog
    import uvicorn

    from clorag.config import get_settings

    settings = get_settings()
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8080"))
    # Initialize services before creating the app (lifespan doesn't run with streamable_http_app)
    # Must set on the actual module (not __main__) to avoid python -m dual-module issue
    import clorag.mcp.server as _mod

    _mod._services = MCPServices()

    mcp = create_mcp_server(host=host, port=port)

    # Get the Starlette app, wrap with auth if configured
    app = mcp.streamable_http_app()

    if settings.mcp_api_key:
        from clorag.mcp.auth import apply_bearer_auth

        app = apply_bearer_auth(app, settings.mcp_api_key.get_secret_value())
        structlog.get_logger().info("mcp_http_auth_enabled")
    else:
        structlog.get_logger().warning(
            "mcp_http_no_auth",
            msg="MCP HTTP running without authentication. Set MCP_API_KEY.",
        )

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    anyio.run(server.serve)


if __name__ == "__main__":
    import sys

    if "--http" in sys.argv:
        main_http()
    else:
        main()
