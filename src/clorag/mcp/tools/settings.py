"""RAG settings management tools for CLORAG MCP server.

Provides tools to list, view, update, and rollback RAG tuning parameters
(retrieval thresholds, reranking, synthesis, caches, prefetch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def register_settings_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register RAG settings management MCP tools."""

    @mcp.tool()
    def list_settings(
        category: str | None = None,
    ) -> dict[str, Any]:
        """List all RAG tuning settings with current values.

        Args:
            category: Optional filter by category:
                retrieval, reranking, synthesis, caches, prefetch.

        Returns:
            List of settings with key, name, value, default, type, and bounds.
        """
        from clorag.services.settings_manager import get_settings_manager

        sm = get_settings_manager()
        settings = sm.get_all(category=category)
        return {"count": len(settings), "settings": settings}

    @mcp.tool()
    def get_setting(
        key: str,
    ) -> dict[str, Any]:
        """Get a RAG setting's current value and metadata.

        Args:
            key: Setting key (e.g. 'retrieval.short_query_threshold',
                'reranking.top_k'). Use list_settings to discover keys.

        Returns:
            Setting value, type, bounds, description, and source.
        """
        from clorag.services.settings_manager import get_settings_manager

        sm = get_settings_manager()
        all_settings = sm.get_all()

        for s in all_settings:
            if s["key"] == key:
                return s

        return {"error": f"Setting with key '{key}' not found"}

    @mcp.tool()
    def update_setting(
        setting_id: str,
        value: str,
        change_note: str | None = None,
    ) -> dict[str, Any]:
        """Update a RAG setting value.

        Creates a new version in history for audit and rollback.
        Some settings (cache sizes) require a server restart to take effect.

        Args:
            setting_id: Setting UUID (get from list_settings).
            value: New value as string (validated against type and bounds).
            change_note: Note explaining the change.

        Returns:
            Updated setting details.
        """
        from clorag.services.settings_manager import get_settings_manager

        sm = get_settings_manager()

        try:
            setting = sm.update(
                setting_id=setting_id,
                value=value,
                change_note=change_note,
                updated_by="mcp",
            )
        except ValueError as e:
            return {"error": str(e)}

        if not setting:
            return {"error": f"Setting with ID '{setting_id}' not found"}

        return {
            "status": "updated",
            "setting": setting.to_dict(),
        }

    @mcp.tool()
    def rollback_setting(
        setting_id: str,
        version: int,
    ) -> dict[str, Any]:
        """Rollback a RAG setting to a previous version.

        Args:
            setting_id: Setting UUID.
            version: Version number to rollback to.

        Returns:
            Restored setting details.
        """
        from clorag.services.settings_manager import get_settings_manager

        sm = get_settings_manager()

        setting = sm.rollback_setting(
            setting_id=setting_id,
            version=version,
            rolled_back_by="mcp",
        )

        if not setting:
            return {
                "error": f"Setting '{setting_id}' or version {version} not found",
            }

        return {
            "status": "rolled_back",
            "to_version": version,
            "setting": setting.to_dict(),
        }
