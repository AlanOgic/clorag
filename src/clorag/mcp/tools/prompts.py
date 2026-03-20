"""Prompt management tools for CLORAG MCP server.

Provides tools to list, view, update, and rollback LLM prompts used across
the RAG pipeline (synthesis, analysis, drafts, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def register_prompt_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register prompt management MCP tools."""

    @mcp.tool(name="list_llm_prompts")
    def list_llm_prompts(
        category: str | None = None,
    ) -> dict[str, Any]:
        """List all LLM prompts with metadata and source (database or default).

        Args:
            category: Optional filter by category
                (agent, analysis, synthesis, drafts, graph, scripts).

        Returns:
            List of all prompts with key, name, description, category, and source.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        prompts = services.prompt_manager.list_all_prompts(category=category)
        return {"count": len(prompts), "prompts": prompts}

    @mcp.tool(name="get_llm_prompt")
    def get_llm_prompt(
        key: str,
    ) -> dict[str, Any]:
        """Get full LLM prompt content and metadata by key.

        Args:
            key: Prompt key (e.g. 'base.system_prompt', 'analysis.thread_analyzer').
                Use list_llm_prompts to discover available keys.

        Returns:
            Prompt content, variables, version info, and source (database or default).
        """
        from clorag.mcp.server import get_services

        services = get_services()
        try:
            return services.prompt_manager.get_prompt_with_metadata(key)
        except KeyError:
            return {"error": f"Prompt with key '{key}' not found"}

    @mcp.tool(name="get_llm_prompt_versions")
    def get_llm_prompt_versions(
        prompt_id: str,
    ) -> dict[str, Any]:
        """Get version history of an LLM prompt (newest first).

        Use before rollback_prompt to see available versions.

        Args:
            prompt_id: Prompt UUID (get from list_llm_prompts).

        Returns:
            List of versions with content, change notes, and timestamps.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        versions = services.prompt_manager.get_prompt_versions(prompt_id)

        if not versions:
            return {"error": f"No versions found for prompt '{prompt_id}'"}

        return {
            "prompt_id": prompt_id,
            "total_versions": len(versions),
            "versions": [v.to_dict() for v in versions],
        }

    @mcp.tool()
    def update_prompt(
        prompt_id: str,
        content: str | None = None,
        name: str | None = None,
        description: str | None = None,
        change_note: str | None = None,
    ) -> dict[str, Any]:
        """Update an LLM prompt's content or metadata.

        If content is changed, a new version is automatically created
        in the version history for audit and rollback.

        Args:
            prompt_id: Prompt UUID (get from list_llm_prompts).
            content: New prompt content (creates a new version).
            name: New display name.
            description: New description.
            change_note: Note explaining the change (for version history).

        Returns:
            Updated prompt details.
        """
        from clorag.mcp.server import get_services

        services = get_services()

        if content is None and name is None and description is None:
            return {"error": "At least one field must be provided"}

        # Auto-detect variables if content changed
        variables = None
        if content is not None:
            variables = services.prompt_manager.detect_variables(content)

        prompt = services.prompt_manager.update_prompt(
            prompt_id=prompt_id,
            content=content,
            name=name,
            description=description,
            variables=variables,
            change_note=change_note,
            updated_by="mcp",
        )

        if not prompt:
            return {"error": f"Prompt with ID '{prompt_id}' not found"}

        return {
            "status": "updated",
            "version_created": content is not None,
            "prompt": prompt.to_dict(),
        }

    @mcp.tool()
    def rollback_prompt(
        prompt_id: str,
        version: int,
    ) -> dict[str, Any]:
        """Rollback a prompt to a previous version.

        Creates a new version with the content from the target version,
        preserving the full history. Use get_llm_prompt_versions to see
        available versions before rolling back.

        Args:
            prompt_id: Prompt UUID.
            version: Version number to rollback to.

        Returns:
            Restored prompt details.
        """
        from clorag.mcp.server import get_services

        services = get_services()

        prompt = services.prompt_manager.rollback_prompt(
            prompt_id=prompt_id,
            version=version,
            rolled_back_by="mcp",
        )

        if not prompt:
            return {
                "error": f"Prompt '{prompt_id}' or version {version} not found",
            }

        return {
            "status": "rolled_back",
            "to_version": version,
            "prompt": prompt.to_dict(),
        }
