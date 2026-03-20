"""MCP tool implementations for CLORAG."""

from clorag.mcp.tools.analytics import register_analytics_tools
from clorag.mcp.tools.cameras import register_camera_tools
from clorag.mcp.tools.chunks import register_chunk_tools
from clorag.mcp.tools.documents import register_document_tools
from clorag.mcp.tools.ingestion import register_ingestion_tools
from clorag.mcp.tools.prompts import register_prompt_tools
from clorag.mcp.tools.search import register_search_tools
from clorag.mcp.tools.settings import register_settings_tools
from clorag.mcp.tools.support import register_support_tools

__all__ = [
    "register_analytics_tools",
    "register_camera_tools",
    "register_chunk_tools",
    "register_document_tools",
    "register_ingestion_tools",
    "register_prompt_tools",
    "register_search_tools",
    "register_settings_tools",
    "register_support_tools",
]
