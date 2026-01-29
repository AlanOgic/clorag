"""Camera database tools for CLORAG MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def _camera_to_dict(camera: Any) -> dict[str, Any]:
    """Convert Camera model to serializable dict."""
    return {
        "id": camera.id,
        "name": camera.name,
        "manufacturer": camera.manufacturer,
        "code_model": camera.code_model,
        "device_type": camera.device_type.value if camera.device_type else None,
        "ports": camera.ports,
        "protocols": camera.protocols,
        "supported_controls": camera.supported_controls,
        "notes": camera.notes,
        "doc_url": camera.doc_url,
        "manufacturer_url": camera.manufacturer_url,
        "confidence": camera.confidence,
    }


def register_camera_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register camera-related MCP tools.

    Args:
        mcp: FastMCP server instance to register tools on.
    """

    @mcp.tool()
    def search_cameras(query: str) -> dict[str, Any]:
        """Search camera compatibility database using full-text search.

        Searches across camera names, manufacturers, model codes, ports,
        protocols, and notes using BM25 ranking.

        Args:
            query: Search query (e.g., "Sony PTZ", "VISCA protocol", "SDI").

        Returns:
            List of matching cameras with compatibility information.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        cameras = services.camera_db.search_cameras(query)

        return {
            "query": query,
            "total_found": len(cameras),
            "cameras": [_camera_to_dict(c) for c in cameras[:20]],  # Limit to 20
        }

    @mcp.tool()
    def get_camera(camera_id: int) -> dict[str, Any]:
        """Get detailed information about a specific camera.

        Args:
            camera_id: Camera database ID.

        Returns:
            Full camera details including ports, protocols, and compatibility notes.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        camera = services.camera_db.get_camera(camera_id)

        if not camera:
            return {"error": f"Camera with ID {camera_id} not found"}

        return {"camera": _camera_to_dict(camera)}

    @mcp.tool()
    def find_related_cameras(camera_id: int, limit: int = 5) -> dict[str, Any]:
        """Find cameras similar to a given camera.

        Finds cameras that share the same manufacturer, device type,
        ports, or protocols - useful for finding alternatives or
        compatible equipment.

        Args:
            camera_id: Reference camera ID.
            limit: Maximum number of related cameras (1-10, default 5).

        Returns:
            List of related cameras with similarity information.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        limit = max(1, min(10, limit))

        # Get reference camera for context
        reference = services.camera_db.get_camera(camera_id)
        if not reference:
            return {"error": f"Camera with ID {camera_id} not found"}

        related = services.camera_db.find_related_cameras(camera_id, limit=limit)

        return {
            "reference_camera": _camera_to_dict(reference),
            "related_cameras": [_camera_to_dict(c) for c in related],
        }

    @mcp.tool()
    def compare_cameras(camera_ids: list[int]) -> dict[str, Any]:
        """Compare multiple cameras side-by-side.

        Useful for understanding differences between camera models
        and their compatibility with Cyanview products.

        Args:
            camera_ids: List of camera IDs to compare (2-5 cameras).

        Returns:
            Side-by-side comparison of camera specifications.
        """
        from clorag.mcp.server import get_services

        services = get_services()

        if len(camera_ids) < 2:
            return {"error": "At least 2 camera IDs required for comparison"}
        if len(camera_ids) > 5:
            return {"error": "Maximum 5 cameras can be compared at once"}

        cameras = services.camera_db.get_cameras_by_ids(camera_ids)
        if not cameras:
            return {"error": "No cameras found with provided IDs"}

        # Build comparison data
        comparison = {
            "cameras": [_camera_to_dict(c) for c in cameras],
            "common_ports": list(
                set.intersection(*[set(c.ports) for c in cameras]) if cameras else set()
            ),
            "common_protocols": list(
                set.intersection(*[set(c.protocols) for c in cameras]) if cameras else set()
            ),
            "manufacturers": list({c.manufacturer for c in cameras if c.manufacturer}),
        }

        return comparison

    @mcp.tool()
    def list_cameras(
        manufacturer: str | None = None,
        device_type: str | None = None,
        protocol: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """List cameras with optional filtering.

        Args:
            manufacturer: Filter by manufacturer (e.g., "Sony", "Panasonic").
            device_type: Filter by device type (e.g., "camera_ptz", "camera_cinema").
            protocol: Filter by protocol (e.g., "VISCA", "Sony RCP").
            limit: Maximum results (1-50, default 20).

        Returns:
            Filtered list of cameras.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        limit = max(1, min(50, limit))

        cameras = services.camera_db.list_cameras(
            manufacturer=manufacturer,
            device_type=device_type,
            protocol=protocol,
            limit=limit,
        )

        return {
            "filters": {
                "manufacturer": manufacturer,
                "device_type": device_type,
                "protocol": protocol,
            },
            "total_found": len(cameras),
            "cameras": [_camera_to_dict(c) for c in cameras],
        }

    @mcp.tool()
    def get_camera_stats() -> dict[str, Any]:
        """Get statistics about the camera database.

        Returns:
            Database statistics including total count, manufacturers, and sources.
        """
        from clorag.mcp.server import get_services

        services = get_services()
        stats = services.camera_db.get_stats()
        manufacturers = services.camera_db.get_manufacturers()
        device_types = services.camera_db.get_device_types()
        ports = services.camera_db.get_all_ports()
        protocols = services.camera_db.get_all_protocols()

        return {
            "total_cameras": stats.get("total_cameras", 0),
            "by_source": stats.get("by_source", {}),
            "manufacturers_count": stats.get("manufacturers", 0),
            "manufacturers": manufacturers[:20],  # Sample of manufacturers
            "device_types": device_types,
            "ports": ports,
            "protocols": protocols,
        }
