"""Admin knowledge graph endpoints.

Provides statistics, entity browsing, and relationship management for Neo4j.
"""

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from clorag.web.auth import verify_admin
from clorag.web.schemas import RelationshipDeleteRequest, RelationshipUpdateRequest
from clorag.web.search import get_graph_enrichment

router = APIRouter(tags=["Graph"])
logger = structlog.get_logger()


@router.get("/graph/stats")
async def api_graph_stats(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get knowledge graph statistics.

    Returns node and relationship counts from Neo4j.
    Returns empty stats if graph is not available.
    """
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {
            "available": False,
            "message": "Graph database not configured or unavailable",
            "stats": {},
        }

    try:
        from clorag.core.graph_store import get_graph_store

        store = await get_graph_store()
        stats = await store.get_stats()
        return {
            "available": True,
            "stats": stats,
        }
    except Exception as e:
        logger.warning("graph_stats_failed", error=str(e))
        return {
            "available": False,
            "message": str(e),
            "stats": {},
        }


@router.get("/graph/entity-types")
async def api_graph_entity_types(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get available entity types with counts."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "types": []}

    try:
        from clorag.core.graph_store import get_graph_store

        store = await get_graph_store()
        types = await store.get_entity_type_counts()
        return {"available": True, "types": types}
    except Exception as e:
        logger.warning("graph_entity_types_failed", error=str(e))
        return {"available": False, "types": [], "error": str(e)}


@router.get("/graph/relationship-types")
async def api_graph_relationship_types(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get available relationship types with counts."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "types": []}

    try:
        from clorag.core.graph_store import get_graph_store

        store = await get_graph_store()
        types = await store.get_relationship_type_counts()
        return {"available": True, "types": types}
    except Exception as e:
        logger.warning("graph_relationship_types_failed", error=str(e))
        return {"available": False, "types": [], "error": str(e)}


@router.get("/graph/entities")
async def api_graph_entities(
    entity_type: str = Query(..., description="Entity type: Camera, Product, etc."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None),
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """List entities by type with pagination and optional search."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "entities": [], "total": 0}

    try:
        from clorag.core.graph_store import get_graph_store

        store = await get_graph_store()
        entities, total = await store.list_entities_by_type(
            entity_type=entity_type,
            page=page,
            page_size=page_size,
            search=search,
        )
        return {
            "available": True,
            "entities": entities,
            "total": total,
            "page": page,
            "page_size": page_size,
            "has_more": (page * page_size) < total,
        }
    except Exception as e:
        logger.warning("graph_entities_failed", error=str(e))
        return {"available": False, "entities": [], "total": 0, "error": str(e)}


@router.get("/graph/entities/{entity_type}/{entity_id:path}")
async def api_graph_entity_detail(
    entity_type: str,
    entity_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get entity details with relationships."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "entity": None}

    try:
        from clorag.core.graph_store import get_graph_store

        store = await get_graph_store()
        entity = await store.get_entity_with_relationships(entity_type, entity_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        return {"available": True, "entity": entity}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("graph_entity_detail_failed", error=str(e))
        return {"available": False, "entity": None, "error": str(e)}


@router.get("/graph/relationships")
async def api_graph_relationships(
    source_type: str | None = Query(None),
    source_name: str | None = Query(None),
    relationship_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """List relationships with optional filtering."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"available": False, "relationships": []}

    try:
        from clorag.core.graph_store import get_graph_store

        store = await get_graph_store()
        relationships = await store.list_relationships(
            source_type=source_type,
            source_name=source_name,
            relationship_type=relationship_type,
            limit=limit,
        )
        return {"available": True, "relationships": relationships}
    except Exception as e:
        logger.warning("graph_relationships_failed", error=str(e))
        return {"available": False, "relationships": [], "error": str(e)}


@router.delete("/graph/relationships")
async def api_delete_relationship(
    request: RelationshipDeleteRequest,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Delete a relationship between two nodes."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"success": False, "error": "Graph database not available"}

    try:
        from clorag.core.graph_store import get_graph_store

        store = await get_graph_store()
        success = await store.delete_relationship(
            source_type=request.source_type,
            source_name=request.source_name,
            rel_type=request.rel_type,
            target_type=request.target_type,
            target_name=request.target_name,
        )
        if success:
            return {"success": True}
        return {"success": False, "error": "Relationship not found"}
    except Exception as e:
        logger.warning("graph_relationship_delete_failed", error=str(e))
        return {"success": False, "error": str(e)}


@router.patch("/graph/relationships")
async def api_update_relationship(
    request: RelationshipUpdateRequest,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Update the type of a relationship between two nodes."""
    enrichment = await get_graph_enrichment()
    if not enrichment:
        return {"success": False, "error": "Graph database not available"}

    try:
        from clorag.core.graph_store import get_graph_store

        store = await get_graph_store()
        success = await store.update_relationship_type(
            source_type=request.source_type,
            source_name=request.source_name,
            old_rel_type=request.old_rel_type,
            new_rel_type=request.new_rel_type,
            target_type=request.target_type,
            target_name=request.target_name,
        )
        if success:
            return {"success": True}
        return {"success": False, "error": "Failed to update relationship"}
    except Exception as e:
        logger.warning("graph_relationship_update_failed", error=str(e))
        return {"success": False, "error": str(e)}
