"""MCP resources and resource templates for CLORAG.

Only exposes resources that have NO equivalent tool, to avoid
bloating the client's context window with duplicate definitions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


CYANVIEW_PRODUCTS = {
    "RCP": {
        "name": "RCP (Remote Control Panel)",
        "description": (
            "Software-based remote control panel for cameras."
            " Runs on tablets, computers, or touchscreens."
        ),
        "variants": ["RCP", "RCP-J (joystick version)"],
        "key_features": [
            "Camera paint control",
            "Iris/gain/shutter",
            "Color matrix",
            "Multi-camera management",
        ],
    },
    "RIO": {
        "name": "RIO (Remote I/O)",
        "description": (
            "Hardware gateway connecting cameras to the RCP"
            " ecosystem. Handles protocol translation."
        ),
        "variants": [
            "RIO (generic hardware)",
            "RIO +WAN (full license: LAN+WAN, cloud, REMI, 1-128 cameras)",
            "RIO +LAN (formerly RIO-Live: LAN only, 1-2 cameras, no cloud/REMI)",
        ],
        "key_features": ["Protocol translation", "Camera connectivity", "Network bridging"],
    },
    "CI0": {
        "name": "CI0 (Camera Interface Zero)",
        "description": "Compact camera interface module for direct camera connection.",
        "variants": ["CI0", "CI0BM (broadcast monitor version)"],
        "key_features": ["Compact form factor", "Direct camera link", "Low latency"],
    },
    "VP4": {
        "name": "VP4 (Video Processor 4)",
        "description": "4-channel video processor for multi-camera setups.",
        "key_features": ["4 video channels", "Real-time processing", "Multi-format support"],
    },
    "NIO": {
        "name": "NIO (Network I/O)",
        "description": "Network-based I/O device for IP camera integration.",
        "key_features": ["IP camera support", "Network integration", "NDI/SRT compatible"],
    },
    "RSBM": {
        "name": "RSBM (Reference Signal / Broadcast Monitor)",
        "description": "Reference signal and broadcast monitoring device.",
        "key_features": ["Reference signal generation", "Broadcast monitoring"],
    },
}


def _serialize(data: Any) -> str:
    """Serialize data to formatted JSON string."""
    return json.dumps(data, indent=2, default=str, ensure_ascii=False)


def register_resources(mcp: FastMCP[MCPServices]) -> None:
    """Register MCP resources that have no equivalent tool."""

    # --- Static product catalog (no tool equivalent) ---

    @mcp.resource("clorag://products")
    def products_catalog() -> str:
        """Cyanview product catalog with all products, variants, and features."""
        return _serialize(CYANVIEW_PRODUCTS)

    # --- Prompt version history (no tool equivalent before v0.10.1) ---
    # Now covered by get_llm_prompt_versions tool — kept for clients that prefer resources.

    @mcp.resource("clorag://prompt/{prompt_id}/versions")
    def prompt_versions(prompt_id: str) -> str:
        """Version history of a prompt (newest first)."""
        from clorag.mcp.server import get_services

        services = get_services()
        versions = services.prompt_manager.get_prompt_versions(prompt_id)
        return _serialize({
            "prompt_id": prompt_id,
            "versions": [v.to_dict() for v in versions],
        })
