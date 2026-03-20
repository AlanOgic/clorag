"""MCP prompt templates for CLORAG."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def register_prompts(mcp: FastMCP[MCPServices]) -> None:
    """Register MCP prompt templates for guided Cyanview support."""

    @mcp.prompt()
    def troubleshoot(
        issue: str, product: str = "",
    ) -> list[dict[str, str]]:
        """Guided troubleshooting for a Cyanview issue."""
        product_context = f" with {product}" if product else ""
        return [
            {
                "role": "user",
                "content": (
                    "I need help troubleshooting a Cyanview"
                    f" issue{product_context}.\n\n"
                    f"**Issue:** {issue}\n\n"
                    "Please:\n"
                    "1. Search the CLORAG knowledge base for relevant"
                    " documentation and past support cases\n"
                    "2. Identify the most likely cause based on the"
                    " search results\n"
                    "3. Provide step-by-step troubleshooting"
                    " instructions\n"
                    "4. Reference specific documentation pages or"
                    " support cases that are relevant\n"
                    "5. If the issue involves camera connectivity,"
                    " check the camera database for compatibility"
                    " info\n\n"
                    "Use all available search tools (docs, cases,"
                    " cameras) to provide comprehensive help."
                ),
            }
        ]

    @mcp.prompt()
    def compare_cameras(
        camera_names: str,
    ) -> list[dict[str, str]]:
        """Compare camera models for Cyanview compatibility."""
        return [
            {
                "role": "user",
                "content": (
                    "Compare the following cameras for Cyanview"
                    f" compatibility: {camera_names}\n\n"
                    "Please:\n"
                    "1. Search the camera database for each model\n"
                    "2. Use the compare_cameras tool if IDs are"
                    " found\n"
                    "3. Highlight compatible ports and protocols\n"
                    "4. Recommend which Cyanview products (RIO, CI0,"
                    " VP4) work best with each camera\n"
                    "5. Note any known issues or limitations from"
                    " support cases"
                ),
            }
        ]

    @mcp.prompt()
    def find_camera(
        use_case: str, requirements: str = "",
    ) -> list[dict[str, str]]:
        """Find compatible cameras for a specific use case."""
        req_context = (
            f"\n**Additional requirements:** {requirements}"
            if requirements else ""
        )
        return [
            {
                "role": "user",
                "content": (
                    "Help me find cameras compatible with Cyanview"
                    " for this use case:\n\n"
                    f"**Use case:** {use_case}{req_context}\n\n"
                    "Please:\n"
                    "1. Search the camera database by relevant"
                    " protocols and ports\n"
                    "2. Filter for cameras that match the use case"
                    " (PTZ, cinema, studio, etc.)\n"
                    "3. For each recommended camera, specify:\n"
                    "   - Which Cyanview product to use"
                    " (RIO +WAN, RIO +LAN, CI0, etc.)\n"
                    "   - Which connection port/protocol to use\n"
                    "   - Any known limitations\n"
                    "4. Check documentation for integration guides"
                    " related to this setup"
                ),
            }
        ]

    @mcp.prompt()
    def explain_product(product: str) -> list[dict[str, str]]:
        """Explain a Cyanview product with features and usage."""
        return [
            {
                "role": "user",
                "content": (
                    "Explain the Cyanview"
                    f" **{product}** product in detail.\n\n"
                    "Please:\n"
                    "1. Search documentation for official product"
                    " information\n"
                    "2. Read the products catalog resource for"
                    " variant details\n"
                    "3. Cover:\n"
                    "   - What it is and what it does\n"
                    "   - Available variants and differences\n"
                    "   - Key features and capabilities\n"
                    "   - Common use cases and scenarios\n"
                    "   - How it connects to other Cyanview"
                    " products\n"
                    "   - Licensing considerations (if applicable)\n"
                    "4. Reference relevant documentation pages for"
                    " further reading"
                ),
            }
        ]

    @mcp.prompt()
    def integration_guide(
        scenario: str,
    ) -> list[dict[str, str]]:
        """Generate an integration guide for a production setup."""
        return [
            {
                "role": "user",
                "content": (
                    "Create an integration guide for the following"
                    " scenario:\n\n"
                    f"**Scenario:** {scenario}\n\n"
                    "Please:\n"
                    "1. Search documentation for relevant setup"
                    " guides\n"
                    "2. Search support cases for similar integration"
                    " experiences\n"
                    "3. Check camera compatibility for any mentioned"
                    " camera models\n"
                    "4. Provide:\n"
                    "   - Required Cyanview hardware and licenses\n"
                    "   - Network architecture (LAN/WAN"
                    " requirements)\n"
                    "   - Step-by-step connection guide\n"
                    "   - Configuration parameters\n"
                    "   - Common pitfalls and troubleshooting tips\n"
                    "5. Include a connection diagram description if"
                    " the setup involves multiple devices"
                ),
            }
        ]

    @mcp.prompt()
    def ingest_status() -> list[dict[str, str]]:
        """Check data sources status and suggest re-ingestion."""
        return [
            {
                "role": "user",
                "content": (
                    "Analyze the current state of the CLORAG"
                    " knowledge base and suggest maintenance"
                    " actions.\n\n"
                    "Please:\n"
                    "1. Read the system stats, camera stats, and"
                    " support stats resources\n"
                    "2. Check document categories for coverage\n"
                    "3. Report:\n"
                    "   - Total documents, cameras, and support"
                    " cases\n"
                    "   - Data distribution across sources and"
                    " categories\n"
                    "   - Any signs of stale or missing data\n"
                    "4. Suggest which ingestion operations would"
                    " improve the knowledge base:\n"
                    "   - ingest_docs (Docusaurus documentation)\n"
                    "   - ingest_curated (Gmail support cases)\n"
                    "   - enrich_cameras (camera model codes/URLs)\n"
                    "   - populate_graph (Neo4j knowledge graph)\n"
                    "   - rebuild_fts_index (camera search index)\n"
                    "   - fix_rio_preview (RIO terminology scan)"
                ),
            }
        ]
