"""Neo4j graph store for GraphRAG knowledge graph operations."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from clorag.config import get_settings
from clorag.graph.schema import (
    EntityExtractionResult,
    GraphCamera,
    GraphChunk,
    GraphControl,
    GraphEnrichmentContext,
    GraphFirmware,
    GraphIssue,
    GraphPort,
    GraphProduct,
    GraphProtocol,
    GraphRelationship,
    GraphSolution,
    RelationType,
)

if TYPE_CHECKING:
    from neo4j import AsyncSession

logger = structlog.get_logger()

# Singleton driver instance
_driver: AsyncDriver | None = None
_driver_lock = asyncio.Lock()


async def get_graph_driver() -> AsyncDriver:
    """Get or create the Neo4j async driver singleton.

    Returns:
        AsyncDriver: Neo4j async driver instance

    Raises:
        AuthError: If Neo4j authentication fails
        ServiceUnavailable: If Neo4j is not reachable
    """
    global _driver

    async with _driver_lock:
        if _driver is None:
            settings = get_settings()
            password = settings.neo4j_password.get_secret_value() if settings.neo4j_password else ""

            _driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
            )

            # Verify connectivity
            try:
                await _driver.verify_connectivity()
                logger.info("neo4j_connected", uri=settings.neo4j_uri)
            except (ServiceUnavailable, AuthError) as e:
                logger.error("neo4j_connection_failed", error=str(e))
                await _driver.close()
                _driver = None
                raise

    return _driver


async def close_graph_driver() -> None:
    """Close the Neo4j driver connection."""
    global _driver

    async with _driver_lock:
        if _driver is not None:
            await _driver.close()
            _driver = None
            logger.info("neo4j_disconnected")


class GraphStore:
    """Neo4j graph store for knowledge graph operations."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver
        self._database = get_settings().neo4j_database

    async def _session(self) -> AsyncSession:
        """Create a new database session."""
        return self._driver.session(database=self._database)

    # =========================================================================
    # Schema Initialization
    # =========================================================================

    async def init_schema(self) -> None:
        """Initialize Neo4j constraints and indexes for optimal performance."""
        constraints = [
            # Unique constraints (also create indexes)
            "CREATE CONSTRAINT camera_name IF NOT EXISTS "
            "FOR (c:Camera) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT product_name IF NOT EXISTS "
            "FOR (p:Product) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT protocol_name IF NOT EXISTS "
            "FOR (p:Protocol) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT port_name IF NOT EXISTS "
            "FOR (p:Port) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT control_name IF NOT EXISTS "
            "FOR (c:Control) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT firmware_version IF NOT EXISTS "
            "FOR (f:Firmware) REQUIRE f.version IS UNIQUE",
        ]

        indexes = [
            # Full-text search indexes
            "CREATE FULLTEXT INDEX issue_description IF NOT EXISTS "
            "FOR (i:Issue) ON EACH [i.description]",
            "CREATE FULLTEXT INDEX solution_description IF NOT EXISTS "
            "FOR (s:Solution) ON EACH [s.description]",
            # Regular indexes for lookups
            "CREATE INDEX camera_manufacturer IF NOT EXISTS "
            "FOR (c:Camera) ON (c.manufacturer)",
            "CREATE INDEX chunk_collection IF NOT EXISTS "
            "FOR (c:Chunk) ON (c.collection)",
        ]

        async with await self._session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    logger.warning(
                        "constraint_creation_skipped", query=constraint[:50], error=str(e)
                    )

            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.warning("index_creation_skipped", query=index[:50], error=str(e))

        logger.info("graph_schema_initialized")

    # =========================================================================
    # Node CRUD Operations
    # =========================================================================

    async def upsert_camera(self, camera: GraphCamera) -> str:
        """Upsert a camera node, respecting human_edited flag."""
        query = """
        MERGE (c:Camera {name: $name})
        ON CREATE SET
            c.manufacturer = $manufacturer,
            c.doc_url = $doc_url,
            c.manufacturer_url = $manufacturer_url,
            c.camera_db_id = $camera_db_id,
            c.human_edited = $human_edited,
            c.created_at = datetime(),
            c.updated_at = datetime()
        ON MATCH SET
            c.manufacturer = CASE WHEN c.human_edited
                THEN c.manufacturer ELSE $manufacturer END,
            c.doc_url = CASE WHEN c.human_edited THEN c.doc_url ELSE $doc_url END,
            c.manufacturer_url = CASE WHEN c.human_edited
                THEN c.manufacturer_url ELSE $manufacturer_url END,
            c.camera_db_id = CASE WHEN c.human_edited
                THEN c.camera_db_id ELSE $camera_db_id END,
            c.updated_at = datetime()
        RETURN elementId(c) as id
        """
        async with await self._session() as session:
            result = await session.run(
                query,
                name=camera.name,
                manufacturer=camera.manufacturer,
                doc_url=camera.doc_url,
                manufacturer_url=camera.manufacturer_url,
                camera_db_id=camera.camera_db_id,
                human_edited=camera.human_edited,
            )
            record = await result.single()
            return record["id"] if record else ""

    async def upsert_product(self, product: GraphProduct) -> str:
        """Upsert a Cyanview product node."""
        query = """
        MERGE (p:Product {name: $name})
        ON CREATE SET
            p.product_type = $product_type,
            p.doc_url = $doc_url,
            p.human_edited = $human_edited,
            p.created_at = datetime(),
            p.updated_at = datetime()
        ON MATCH SET
            p.product_type = CASE WHEN p.human_edited THEN p.product_type ELSE $product_type END,
            p.doc_url = CASE WHEN p.human_edited THEN p.doc_url ELSE $doc_url END,
            p.updated_at = datetime()
        RETURN elementId(p) as id
        """
        async with await self._session() as session:
            result = await session.run(
                query,
                name=product.name,
                product_type=product.product_type,
                doc_url=product.doc_url,
                human_edited=product.human_edited,
            )
            record = await result.single()
            return record["id"] if record else ""

    async def upsert_protocol(self, protocol: GraphProtocol) -> str:
        """Upsert a protocol node."""
        query = """
        MERGE (p:Protocol {name: $name})
        ON CREATE SET
            p.protocol_type = $protocol_type,
            p.description = $description,
            p.created_at = datetime(),
            p.updated_at = datetime()
        ON MATCH SET
            p.protocol_type = COALESCE($protocol_type, p.protocol_type),
            p.description = COALESCE($description, p.description),
            p.updated_at = datetime()
        RETURN elementId(p) as id
        """
        async with await self._session() as session:
            result = await session.run(
                query,
                name=protocol.name,
                protocol_type=protocol.protocol_type,
                description=protocol.description,
            )
            record = await result.single()
            return record["id"] if record else ""

    async def upsert_port(self, port: GraphPort) -> str:
        """Upsert a port/connector node."""
        query = """
        MERGE (p:Port {name: $name})
        ON CREATE SET
            p.port_type = $port_type,
            p.created_at = datetime(),
            p.updated_at = datetime()
        ON MATCH SET
            p.port_type = COALESCE($port_type, p.port_type),
            p.updated_at = datetime()
        RETURN elementId(p) as id
        """
        async with await self._session() as session:
            result = await session.run(query, name=port.name, port_type=port.port_type)
            record = await result.single()
            return record["id"] if record else ""

    async def upsert_control(self, control: GraphControl) -> str:
        """Upsert a camera control node."""
        query = """
        MERGE (c:Control {name: $name})
        ON CREATE SET
            c.description = $description,
            c.control_type = $control_type,
            c.created_at = datetime(),
            c.updated_at = datetime()
        ON MATCH SET
            c.description = COALESCE($description, c.description),
            c.control_type = COALESCE($control_type, c.control_type),
            c.updated_at = datetime()
        RETURN elementId(c) as id
        """
        async with await self._session() as session:
            result = await session.run(
                query,
                name=control.name,
                description=control.description,
                control_type=control.control_type,
            )
            record = await result.single()
            return record["id"] if record else ""

    async def upsert_issue(self, issue: GraphIssue) -> str:
        """Upsert an issue node."""
        query = """
        MERGE (i:Issue {description: $description})
        ON CREATE SET
            i.symptoms = $symptoms,
            i.error_codes = $error_codes,
            i.created_at = datetime(),
            i.updated_at = datetime()
        ON MATCH SET
            i.symptoms = $symptoms,
            i.error_codes = $error_codes,
            i.updated_at = datetime()
        RETURN elementId(i) as id
        """
        async with await self._session() as session:
            result = await session.run(
                query,
                description=issue.description,
                symptoms=issue.symptoms,
                error_codes=issue.error_codes,
            )
            record = await result.single()
            return record["id"] if record else ""

    async def upsert_solution(self, solution: GraphSolution) -> str:
        """Upsert a solution node."""
        query = """
        MERGE (s:Solution {description: $description})
        ON CREATE SET
            s.steps = $steps,
            s.verified = $verified,
            s.created_at = datetime(),
            s.updated_at = datetime()
        ON MATCH SET
            s.steps = $steps,
            s.verified = $verified,
            s.updated_at = datetime()
        RETURN elementId(s) as id
        """
        async with await self._session() as session:
            result = await session.run(
                query,
                description=solution.description,
                steps=solution.steps,
                verified=solution.verified,
            )
            record = await result.single()
            return record["id"] if record else ""

    async def upsert_firmware(self, firmware: GraphFirmware) -> str:
        """Upsert a firmware version node."""
        query = """
        MERGE (f:Firmware {version: $version})
        ON CREATE SET
            f.release_date = $release_date,
            f.changelog = $changelog,
            f.created_at = datetime(),
            f.updated_at = datetime()
        ON MATCH SET
            f.release_date = COALESCE($release_date, f.release_date),
            f.changelog = COALESCE($changelog, f.changelog),
            f.updated_at = datetime()
        RETURN elementId(f) as id
        """
        async with await self._session() as session:
            result = await session.run(
                query,
                version=firmware.version,
                release_date=firmware.release_date.isoformat() if firmware.release_date else None,
                changelog=firmware.changelog,
            )
            record = await result.single()
            return record["id"] if record else ""

    async def upsert_chunk(self, chunk: GraphChunk) -> str:
        """Upsert a chunk reference node."""
        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        ON CREATE SET
            c.collection = $collection,
            c.title = $title,
            c.source_url = $source_url,
            c.created_at = datetime(),
            c.updated_at = datetime()
        ON MATCH SET
            c.collection = $collection,
            c.title = $title,
            c.source_url = $source_url,
            c.updated_at = datetime()
        RETURN elementId(c) as id
        """
        async with await self._session() as session:
            result = await session.run(
                query,
                chunk_id=chunk.chunk_id,
                collection=chunk.collection,
                title=chunk.title,
                source_url=chunk.source_url,
            )
            record = await result.single()
            return record["id"] if record else ""

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    async def create_relationship(self, rel: GraphRelationship) -> bool:
        """Create a relationship between two nodes."""
        # Map relationship types to Cypher patterns
        rel_queries = {
            RelationType.COMPATIBLE_WITH: """
                MATCH (a:Camera {name: $source_id}), (b:Product {name: $target_id})
                MERGE (a)-[r:COMPATIBLE_WITH]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.USES_PROTOCOL: """
                MATCH (a:Camera {name: $source_id}), (b:Protocol {name: $target_id})
                MERGE (a)-[r:USES_PROTOCOL]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.HAS_PORT: """
                MATCH (a {name: $source_id}), (b:Port {name: $target_id})
                WHERE a:Camera OR a:Product
                MERGE (a)-[r:HAS_PORT]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.SUPPORTS_CONTROL: """
                MATCH (a:Camera {name: $source_id}), (b:Control {name: $target_id})
                MERGE (a)-[r:SUPPORTS_CONTROL]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.SUPPORTS_PROTOCOL: """
                MATCH (a:Product {name: $source_id}), (b:Protocol {name: $target_id})
                MERGE (a)-[r:SUPPORTS_PROTOCOL]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.AFFECTS: """
                MATCH (a:Issue {description: $source_id}), (b {name: $target_id})
                WHERE b:Product OR b:Camera
                MERGE (a)-[r:AFFECTS]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.RESOLVED_BY: """
                MATCH (a:Issue {description: $source_id}), (b:Solution {description: $target_id})
                MERGE (a)-[r:RESOLVED_BY]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.MENTIONED_IN: """
                MATCH (a {description: $source_id}), (b:Chunk {chunk_id: $target_id})
                MERGE (a)-[r:MENTIONED_IN]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.FOR_PRODUCT: """
                MATCH (a:Firmware {version: $source_id}), (b:Product {name: $target_id})
                MERGE (a)-[r:FOR_PRODUCT]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.FIXES: """
                MATCH (a:Firmware {version: $source_id}), (b:Issue {description: $target_id})
                MERGE (a)-[r:FIXES]->(b)
                SET r += $properties
                RETURN r
            """,
            RelationType.MENTIONS: """
                MATCH (a:Chunk {chunk_id: $source_id}), (b {name: $target_id})
                MERGE (a)-[r:MENTIONS]->(b)
                SET r += $properties
                RETURN r
            """,
        }

        query = rel_queries.get(rel.relationship)
        if not query:
            logger.warning("unknown_relationship_type", rel_type=rel.relationship)
            return False

        async with await self._session() as session:
            try:
                result = await session.run(
                    query,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    properties=rel.properties,
                )
                record = await result.single()
                return record is not None
            except Exception as e:
                logger.warning(
                    "relationship_creation_failed",
                    rel_type=rel.relationship.value,
                    error=str(e),
                )
                return False

    async def delete_relationship(
        self,
        source_type: str,
        source_name: str,
        rel_type: str,
        target_type: str,
        target_name: str,
    ) -> bool:
        """Delete a relationship between two nodes.

        Args:
            source_type: Source node type (Camera, Product, etc.)
            source_name: Source node identifier (name/description/version/chunk_id)
            rel_type: Relationship type (COMPATIBLE_WITH, USES_PROTOCOL, etc.)
            target_type: Target node type
            target_name: Target node identifier

        Returns:
            True if relationship was deleted, False otherwise
        """
        valid_types = {
            "Camera", "Product", "Protocol", "Port", "Control",
            "Issue", "Solution", "Firmware", "Chunk"
        }
        if source_type not in valid_types or target_type not in valid_types:
            return False

        # Determine the property name based on node type
        def get_prop_name(node_type: str) -> str:
            if node_type in ("Issue", "Solution"):
                return "description"
            elif node_type == "Firmware":
                return "version"
            elif node_type == "Chunk":
                return "chunk_id"
            return "name"

        source_prop = get_prop_name(source_type)
        target_prop = get_prop_name(target_type)

        query = f"""
        MATCH (a:{source_type} {{{source_prop}: $source_name}})
              -[r:{rel_type}]->
              (b:{target_type} {{{target_prop}: $target_name}})
        DELETE r
        RETURN count(r) as deleted
        """

        async with await self._session() as session:
            try:
                result = await session.run(
                    query,
                    source_name=source_name,
                    target_name=target_name,
                )
                record = await result.single()
                deleted = record["deleted"] if record else 0
                if deleted > 0:
                    logger.info(
                        "relationship_deleted",
                        source_type=source_type,
                        rel_type=rel_type,
                        target_type=target_type,
                    )
                return deleted > 0
            except Exception as e:
                logger.warning("relationship_delete_failed", error=str(e))
                return False

    async def update_relationship_type(
        self,
        source_type: str,
        source_name: str,
        old_rel_type: str,
        new_rel_type: str,
        target_type: str,
        target_name: str,
    ) -> bool:
        """Update a relationship type between two nodes.

        Args:
            source_type: Source node type
            source_name: Source node identifier
            old_rel_type: Current relationship type
            new_rel_type: New relationship type
            target_type: Target node type
            target_name: Target node identifier

        Returns:
            True if relationship was updated, False otherwise
        """
        valid_types = {
            "Camera", "Product", "Protocol", "Port", "Control",
            "Issue", "Solution", "Firmware", "Chunk"
        }
        if source_type not in valid_types or target_type not in valid_types:
            return False

        # Valid relationship types
        valid_rel_types = {
            "COMPATIBLE_WITH", "USES_PROTOCOL", "HAS_PORT", "SUPPORTS_CONTROL",
            "SUPPORTS_PROTOCOL", "AFFECTS", "RESOLVED_BY", "MENTIONED_IN",
            "FOR_PRODUCT", "FIXES", "MENTIONS"
        }
        if new_rel_type not in valid_rel_types:
            return False

        def get_prop_name(node_type: str) -> str:
            if node_type in ("Issue", "Solution"):
                return "description"
            elif node_type == "Firmware":
                return "version"
            elif node_type == "Chunk":
                return "chunk_id"
            return "name"

        source_prop = get_prop_name(source_type)
        target_prop = get_prop_name(target_type)

        # Delete old and create new relationship (Neo4j doesn't support changing rel type)
        delete_query = f"""
        MATCH (a:{source_type} {{{source_prop}: $source_name}})
              -[r:{old_rel_type}]->
              (b:{target_type} {{{target_prop}: $target_name}})
        WITH a, b, properties(r) as props
        DELETE r
        RETURN a, b, props
        """

        async with await self._session() as session:
            try:
                # Delete old relationship and get nodes
                result = await session.run(
                    delete_query,
                    source_name=source_name,
                    target_name=target_name,
                )
                record = await result.single()
                if not record:
                    return False

                # Create new relationship with same properties
                props = record["props"] or {}
                create_query = f"""
                MATCH (a:{source_type} {{{source_prop}: $source_name}}),
                      (b:{target_type} {{{target_prop}: $target_name}})
                CREATE (a)-[r:{new_rel_type}]->(b)
                SET r += $props
                RETURN r
                """

                create_result = await session.run(
                    create_query,
                    source_name=source_name,
                    target_name=target_name,
                    props=props,
                )
                create_record = await create_result.single()
                if create_record:
                    logger.info(
                        "relationship_type_updated",
                        source_type=source_type,
                        old_type=old_rel_type,
                        new_type=new_rel_type,
                        target_type=target_type,
                    )
                    return True
                return False
            except Exception as e:
                logger.warning("relationship_update_failed", error=str(e))
                return False

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def ingest_extraction_result(self, result: EntityExtractionResult) -> dict[str, int]:
        """Ingest all entities and relationships from an extraction result.

        Returns:
            Dict with counts of created entities by type
        """
        counts: dict[str, int] = {
            "cameras": 0,
            "products": 0,
            "protocols": 0,
            "ports": 0,
            "controls": 0,
            "issues": 0,
            "solutions": 0,
            "firmware": 0,
            "relationships": 0,
        }

        # Upsert all entities
        for camera in result.cameras:
            await self.upsert_camera(camera)
            counts["cameras"] += 1

        for product in result.products:
            await self.upsert_product(product)
            counts["products"] += 1

        for protocol in result.protocols:
            await self.upsert_protocol(protocol)
            counts["protocols"] += 1

        for port in result.ports:
            await self.upsert_port(port)
            counts["ports"] += 1

        for control in result.controls:
            await self.upsert_control(control)
            counts["controls"] += 1

        for issue in result.issues:
            await self.upsert_issue(issue)
            counts["issues"] += 1

        for solution in result.solutions:
            await self.upsert_solution(solution)
            counts["solutions"] += 1

        for firmware in result.firmware_versions:
            await self.upsert_firmware(firmware)
            counts["firmware"] += 1

        # Create relationships
        for rel in result.relationships:
            if await self.create_relationship(rel):
                counts["relationships"] += 1

        return counts

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def get_stats(self) -> dict[str, int]:
        """Get graph statistics (node and relationship counts)."""
        query = """
        CALL {
            MATCH (c:Camera) RETURN 'cameras' as label, count(c) as count
            UNION ALL
            MATCH (p:Product) RETURN 'products' as label, count(p) as count
            UNION ALL
            MATCH (p:Protocol) RETURN 'protocols' as label, count(p) as count
            UNION ALL
            MATCH (p:Port) RETURN 'ports' as label, count(p) as count
            UNION ALL
            MATCH (c:Control) RETURN 'controls' as label, count(c) as count
            UNION ALL
            MATCH (i:Issue) RETURN 'issues' as label, count(i) as count
            UNION ALL
            MATCH (s:Solution) RETURN 'solutions' as label, count(s) as count
            UNION ALL
            MATCH (f:Firmware) RETURN 'firmware' as label, count(f) as count
            UNION ALL
            MATCH (c:Chunk) RETURN 'chunks' as label, count(c) as count
            UNION ALL
            MATCH ()-[r]->() RETURN 'relationships' as label, count(r) as count
        }
        RETURN label, count
        """
        stats: dict[str, int] = {}
        async with await self._session() as session:
            result = await session.run(query)
            async for record in result:
                stats[record["label"]] = record["count"]
        return stats

    async def find_related_to_chunk(
        self, chunk_id: str, max_hops: int = 2
    ) -> GraphEnrichmentContext:
        """Find all entities related to a chunk within N hops.

        Args:
            chunk_id: Qdrant chunk UUID
            max_hops: Maximum relationship hops (default 2)

        Returns:
            GraphEnrichmentContext with related entities
        """
        rel_filter = (
            "MENTIONS|MENTIONED_IN|COMPATIBLE_WITH|USES_PROTOCOL|AFFECTS|RESOLVED_BY"
        )
        query = f"""
        MATCH (c:Chunk {{chunk_id: $chunk_id}})
        CALL apoc.path.subgraphNodes(c, {{
            maxLevel: $max_hops,
            relationshipFilter: "{rel_filter}"
        }}) YIELD node
        WITH node, labels(node) as nodeLabels
        RETURN node, nodeLabels
        """

        context = GraphEnrichmentContext()

        async with await self._session() as session:
            try:
                result = await session.run(query, chunk_id=chunk_id, max_hops=max_hops)
                async for record in result:
                    node = record["node"]
                    labels = record["nodeLabels"]

                    if "Camera" in labels:
                        context.related_cameras.append(GraphCamera(
                            name=node.get("name", ""),
                            manufacturer=node.get("manufacturer"),
                        ))
                    elif "Product" in labels:
                        context.related_products.append(GraphProduct(
                            name=node.get("name", ""),
                            product_type=node.get("product_type"),
                        ))
                    elif "Protocol" in labels:
                        context.related_protocols.append(GraphProtocol(
                            name=node.get("name", ""),
                        ))
                    elif "Issue" in labels:
                        context.related_issues.append(GraphIssue(
                            description=node.get("description", ""),
                        ))
                    elif "Solution" in labels:
                        context.related_solutions.append(GraphSolution(
                            description=node.get("description", ""),
                        ))
                    elif "Firmware" in labels:
                        context.related_firmware.append(GraphFirmware(
                            version=node.get("version", ""),
                        ))
            except Exception as e:
                logger.warning("graph_enrichment_failed", chunk_id=chunk_id, error=str(e))

        return context

    async def find_camera_compatibility(self, camera_name: str) -> dict:
        """Find all products, protocols, and ports compatible with a camera."""
        query = """
        MATCH (c:Camera {name: $camera_name})
        OPTIONAL MATCH (c)-[:COMPATIBLE_WITH]->(p:Product)
        OPTIONAL MATCH (c)-[:USES_PROTOCOL]->(proto:Protocol)
        OPTIONAL MATCH (c)-[:HAS_PORT]->(port:Port)
        OPTIONAL MATCH (c)-[:SUPPORTS_CONTROL]->(ctrl:Control)
        RETURN
            collect(DISTINCT p.name) as products,
            collect(DISTINCT proto.name) as protocols,
            collect(DISTINCT port.name) as ports,
            collect(DISTINCT ctrl.name) as controls
        """
        async with await self._session() as session:
            result = await session.run(query, camera_name=camera_name)
            record = await result.single()
            if record:
                return {
                    "products": record["products"],
                    "protocols": record["protocols"],
                    "ports": record["ports"],
                    "controls": record["controls"],
                }
        return {"products": [], "protocols": [], "ports": [], "controls": []}

    async def find_issues_for_product(self, product_name: str) -> list[dict]:
        """Find all issues affecting a product with their solutions."""
        query = """
        MATCH (p:Product {name: $product_name})<-[:AFFECTS]-(i:Issue)
        OPTIONAL MATCH (i)-[:RESOLVED_BY]->(s:Solution)
        RETURN i.description as issue, collect(s.description) as solutions
        """
        issues = []
        async with await self._session() as session:
            result = await session.run(query, product_name=product_name)
            async for record in result:
                issues.append({
                    "issue": record["issue"],
                    "solutions": record["solutions"],
                })
        return issues

    async def search_entities(self, query_text: str, limit: int = 10) -> list[dict]:
        """Full-text search across issues and solutions."""
        query = """
        CALL db.index.fulltext.queryNodes('issue_description', $query) YIELD node, score
        RETURN 'issue' as type, node.description as text, score
        UNION ALL
        CALL db.index.fulltext.queryNodes('solution_description', $query) YIELD node, score
        RETURN 'solution' as type, node.description as text, score
        ORDER BY score DESC
        LIMIT $limit
        """
        results = []
        async with await self._session() as session:
            try:
                result = await session.run(query, query=query_text, limit=limit)
                async for record in result:
                    results.append({
                        "type": record["type"],
                        "text": record["text"],
                        "score": record["score"],
                    })
            except Exception as e:
                logger.warning("fulltext_search_failed", error=str(e))
        return results

    # =========================================================================
    # Admin Browser Operations
    # =========================================================================

    async def get_entity_type_counts(self) -> list[dict]:
        """Get counts for each entity type.

        Returns:
            List of dicts with type and count
        """
        query = """
        CALL {
            MATCH (c:Camera) RETURN 'Camera' as type, count(c) as count
            UNION ALL
            MATCH (p:Product) RETURN 'Product' as type, count(p) as count
            UNION ALL
            MATCH (p:Protocol) RETURN 'Protocol' as type, count(p) as count
            UNION ALL
            MATCH (p:Port) RETURN 'Port' as type, count(p) as count
            UNION ALL
            MATCH (c:Control) RETURN 'Control' as type, count(c) as count
            UNION ALL
            MATCH (i:Issue) RETURN 'Issue' as type, count(i) as count
            UNION ALL
            MATCH (s:Solution) RETURN 'Solution' as type, count(s) as count
            UNION ALL
            MATCH (f:Firmware) RETURN 'Firmware' as type, count(f) as count
            UNION ALL
            MATCH (c:Chunk) RETURN 'Chunk' as type, count(c) as count
        }
        RETURN type, count
        ORDER BY count DESC
        """
        results = []
        async with await self._session() as session:
            result = await session.run(query)
            async for record in result:
                results.append({"type": record["type"], "count": record["count"]})
        return results

    async def list_entities_by_type(
        self,
        entity_type: str,
        page: int = 1,
        page_size: int = 20,
        search: str | None = None,
    ) -> tuple[list[dict], int]:
        """List entities of a specific type with pagination.

        Args:
            entity_type: Node label (Camera, Product, Protocol, etc.)
            page: Page number (1-indexed)
            page_size: Items per page
            search: Optional search filter for name/description

        Returns:
            Tuple of (list of entity dicts, total count)
        """
        # Validate entity type to prevent injection
        valid_types = {
            "Camera", "Product", "Protocol", "Port", "Control",
            "Issue", "Solution", "Firmware", "Chunk"
        }
        if entity_type not in valid_types:
            return [], 0

        skip = (page - 1) * page_size

        # Build query based on entity type (different primary fields)
        if entity_type == "Issue":
            name_field = "description"
        elif entity_type == "Solution":
            name_field = "description"
        elif entity_type == "Firmware":
            name_field = "version"
        elif entity_type == "Chunk":
            name_field = "chunk_id"
        else:
            name_field = "name"

        # Count query
        if search:
            count_query = f"""
            MATCH (n:{entity_type})
            WHERE toLower(n.{name_field}) CONTAINS toLower($search)
            RETURN count(n) as total
            """
        else:
            count_query = f"""
            MATCH (n:{entity_type})
            RETURN count(n) as total
            """

        # Data query with relationship count
        if search:
            data_query = f"""
            MATCH (n:{entity_type})
            WHERE toLower(n.{name_field}) CONTAINS toLower($search)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(DISTINCT r) as rel_count
            RETURN elementId(n) as id, properties(n) as props, rel_count
            ORDER BY n.{name_field}
            SKIP $skip LIMIT $limit
            """
        else:
            data_query = f"""
            MATCH (n:{entity_type})
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(DISTINCT r) as rel_count
            RETURN elementId(n) as id, properties(n) as props, rel_count
            ORDER BY n.{name_field}
            SKIP $skip LIMIT $limit
            """

        entities = []
        total = 0

        async with await self._session() as session:
            # Get total count
            count_result = await session.run(
                count_query,
                search=search if search else "",
            )
            count_record = await count_result.single()
            total = count_record["total"] if count_record else 0

            # Get entities
            data_result = await session.run(
                data_query,
                search=search if search else "",
                skip=skip,
                limit=page_size,
            )
            async for record in data_result:
                props = dict(record["props"])
                entities.append({
                    "id": record["id"],
                    "name": props.get(name_field, ""),
                    "entity_type": entity_type,
                    "properties": props,
                    "relationship_count": record["rel_count"],
                })

        return entities, total

    async def get_entity_with_relationships(
        self,
        entity_type: str,
        entity_id: str,
    ) -> dict | None:
        """Get entity details with all its relationships.

        Args:
            entity_type: Node label
            entity_id: Neo4j element ID

        Returns:
            Dict with entity properties and relationships list
        """
        valid_types = {
            "Camera", "Product", "Protocol", "Port", "Control",
            "Issue", "Solution", "Firmware", "Chunk"
        }
        if entity_type not in valid_types:
            return None

        query = f"""
        MATCH (n:{entity_type})
        WHERE elementId(n) = $entity_id
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n, collect({{
            direction: CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END,
            rel_type: type(r),
            target_type: labels(m)[0],
            target_name: COALESCE(m.name, m.description, m.version, m.chunk_id),
            target_id: elementId(m)
        }}) as relationships
        """

        async with await self._session() as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            if not record:
                return None

            node = record["n"]
            rels = record["relationships"]

            # Determine primary name field
            props = dict(node)
            if entity_type in ("Issue", "Solution"):
                name = props.get("description", "")
            elif entity_type == "Firmware":
                name = props.get("version", "")
            elif entity_type == "Chunk":
                name = props.get("chunk_id", "")
            else:
                name = props.get("name", "")

            return {
                "id": entity_id,
                "name": name,
                "entity_type": entity_type,
                "properties": props,
                "relationships": [r for r in rels if r["target_name"]],
            }

    async def list_relationships(
        self,
        source_type: str | None = None,
        source_name: str | None = None,
        relationship_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List relationships with optional filtering.

        Args:
            source_type: Filter by source node type
            source_name: Filter by source node name/description
            relationship_type: Filter by relationship type
            limit: Maximum results

        Returns:
            List of relationship dicts
        """
        # Build dynamic query
        where_clauses = []
        params: dict = {"limit": limit}

        if source_type:
            valid_types = {
                "Camera", "Product", "Protocol", "Port", "Control",
                "Issue", "Solution", "Firmware", "Chunk"
            }
            if source_type not in valid_types:
                return []

        if source_name:
            where_clauses.append(
                "(a.name = $source_name OR a.description = $source_name "
                "OR a.version = $source_name OR a.chunk_id = $source_name)"
            )
            params["source_name"] = source_name

        # Base query
        if source_type:
            match_clause = f"MATCH (a:{source_type})-[r]->(b)"
        else:
            match_clause = "MATCH (a)-[r]->(b)"

        if relationship_type:
            match_clause = match_clause.replace("-[r]->", f"-[r:{relationship_type}]->")

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        query = f"""
        {match_clause}
        {where_clause}
        RETURN
            labels(a)[0] as source_type,
            COALESCE(a.name, a.description, a.version, a.chunk_id) as source_name,
            type(r) as relationship,
            labels(b)[0] as target_type,
            COALESCE(b.name, b.description, b.version, b.chunk_id) as target_name,
            properties(r) as properties
        LIMIT $limit
        """

        results = []
        async with await self._session() as session:
            result = await session.run(query, **params)
            async for record in result:
                results.append({
                    "source_type": record["source_type"],
                    "source_name": record["source_name"],
                    "relationship": record["relationship"],
                    "target_type": record["target_type"],
                    "target_name": record["target_name"],
                    "properties": dict(record["properties"]) if record["properties"] else {},
                })
        return results

    async def get_relationship_type_counts(self) -> list[dict]:
        """Get counts for each relationship type.

        Returns:
            List of dicts with type and count
        """
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
        """
        results = []
        async with await self._session() as session:
            result = await session.run(query)
            async for record in result:
                results.append({"type": record["type"], "count": record["count"]})
        return results


# Factory function
_graph_store: GraphStore | None = None
_store_lock = asyncio.Lock()


async def get_graph_store() -> GraphStore:
    """Get or create the GraphStore singleton."""
    global _graph_store

    async with _store_lock:
        if _graph_store is None:
            driver = await get_graph_driver()
            _graph_store = GraphStore(driver)
            await _graph_store.init_schema()

    return _graph_store
