"""MCP tool implementations for CLORAG."""

from clorag.mcp.tools.cameras import register_camera_tools
from clorag.mcp.tools.documents import register_document_tools
from clorag.mcp.tools.odoo import register_odoo_tools
from clorag.mcp.tools.search import register_search_tools
from clorag.mcp.tools.support import register_support_tools

__all__ = [
    "register_camera_tools",
    "register_document_tools",
    "register_odoo_tools",
    "register_search_tools",
    "register_support_tools",
]
