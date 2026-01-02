"""Graph context enrichment for RAG search results."""

from __future__ import annotations

import asyncio

import structlog

from clorag.core.graph_store import GraphStore, get_graph_store
from clorag.graph.schema import GraphEnrichmentContext

logger = structlog.get_logger()


class GraphEnrichmentService:
    """Service for enriching vector search results with graph context."""

    def __init__(self, graph_store: GraphStore) -> None:
        self._store = graph_store

    async def enrich_from_chunks(
        self,
        chunk_ids: list[str],
        max_hops: int = 2,
    ) -> GraphEnrichmentContext:
        """Enrich search results by traversing graph from chunk nodes.

        Args:
            chunk_ids: List of Qdrant chunk UUIDs from vector search results.
            max_hops: Maximum relationship hops for graph traversal.

        Returns:
            GraphEnrichmentContext with related entities and paths.
        """
        if not chunk_ids:
            return GraphEnrichmentContext()

        # Query graph for each chunk in parallel
        tasks = [
            self._store.find_related_to_chunk(chunk_id, max_hops)
            for chunk_id in chunk_ids[:10]  # Limit to first 10 chunks
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge contexts from all chunks
        merged = GraphEnrichmentContext()
        seen_cameras: set[str] = set()
        seen_products: set[str] = set()
        seen_protocols: set[str] = set()
        seen_issues: set[str] = set()
        seen_solutions: set[str] = set()

        for result in results:
            if isinstance(result, Exception):
                logger.warning("graph_enrichment_chunk_failed", error=str(result))
                continue

            # Deduplicate cameras
            for camera in result.related_cameras:
                if camera.name not in seen_cameras:
                    seen_cameras.add(camera.name)
                    merged.related_cameras.append(camera)

            # Deduplicate products
            for product in result.related_products:
                if product.name not in seen_products:
                    seen_products.add(product.name)
                    merged.related_products.append(product)

            # Deduplicate protocols
            for protocol in result.related_protocols:
                if protocol.name not in seen_protocols:
                    seen_protocols.add(protocol.name)
                    merged.related_protocols.append(protocol)

            # Deduplicate issues (by description prefix)
            for issue in result.related_issues:
                key = issue.description[:50]
                if key not in seen_issues:
                    seen_issues.add(key)
                    merged.related_issues.append(issue)

            # Deduplicate solutions
            for solution in result.related_solutions:
                key = solution.description[:50]
                if key not in seen_solutions:
                    seen_solutions.add(key)
                    merged.related_solutions.append(solution)

            # Add firmware versions
            merged.related_firmware.extend(result.related_firmware)

            # Add relationship paths
            merged.paths.extend(result.paths)

        logger.debug(
            "graph_enrichment_complete",
            chunks=len(chunk_ids),
            cameras=len(merged.related_cameras),
            products=len(merged.related_products),
            protocols=len(merged.related_protocols),
        )

        return merged

    async def enrich_from_query(
        self,
        query: str,
        limit: int = 10,
    ) -> GraphEnrichmentContext:
        """Enrich context by searching the graph directly.

        Uses full-text search on issues and solutions to find relevant
        graph context that may not be linked to vector search results.

        Args:
            query: User search query.
            limit: Maximum entities to return per type.

        Returns:
            GraphEnrichmentContext with search results from graph.
        """
        context = GraphEnrichmentContext()

        # Search issues and solutions using full-text index
        search_results = await self._store.search_entities(query, limit=limit)

        for result in search_results:
            if result["type"] == "issue":
                from clorag.graph.schema import GraphIssue
                context.related_issues.append(GraphIssue(
                    description=result["text"],
                ))
            elif result["type"] == "solution":
                from clorag.graph.schema import GraphSolution
                context.related_solutions.append(GraphSolution(
                    description=result["text"],
                ))

        # Extract potential camera/product names from query and look them up
        await self._enrich_from_entity_mentions(query, context)

        return context

    async def _enrich_from_entity_mentions(
        self,
        query: str,
        context: GraphEnrichmentContext,
    ) -> None:
        """Look up entities mentioned in the query."""
        query_lower = query.lower()

        # Check for Cyanview product mentions
        product_names = ["rio", "rcp", "ci0", "vp4", "live composer"]
        for product in product_names:
            if product in query_lower:
                issues = await self._store.find_issues_for_product(product.upper())
                for issue_data in issues[:3]:
                    from clorag.graph.schema import GraphIssue, GraphSolution
                    if issue_data["issue"]:
                        context.related_issues.append(GraphIssue(
                            description=issue_data["issue"],
                        ))
                    for sol in issue_data.get("solutions", [])[:2]:
                        if sol:
                            context.related_solutions.append(GraphSolution(
                                description=sol,
                            ))

    async def get_camera_context(self, camera_name: str) -> dict:
        """Get full context for a specific camera.

        Args:
            camera_name: Camera model name.

        Returns:
            Dict with products, protocols, ports, controls, and issues.
        """
        compatibility = await self._store.find_camera_compatibility(camera_name)

        # Also get any issues affecting this camera
        # (would need additional query - simplified for now)

        return {
            "camera": camera_name,
            "compatible_products": compatibility["products"],
            "protocols": compatibility["protocols"],
            "ports": compatibility["ports"],
            "controls": compatibility["controls"],
        }


# Factory function
_enrichment_service: GraphEnrichmentService | None = None
_service_lock = asyncio.Lock()


async def get_enrichment_service() -> GraphEnrichmentService:
    """Get or create the GraphEnrichmentService singleton."""
    global _enrichment_service

    async with _service_lock:
        if _enrichment_service is None:
            store = await get_graph_store()
            _enrichment_service = GraphEnrichmentService(store)

    return _enrichment_service
