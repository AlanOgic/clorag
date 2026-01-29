"""MCP server for CLORAG exposing RAG capabilities to Claude Desktop."""

from clorag.mcp.server import create_mcp_server, main

__all__ = ["create_mcp_server", "main"]
