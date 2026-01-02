"""Pydantic models for GraphRAG entities and relationships."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """Types of relationships between graph entities."""

    # Camera relationships
    COMPATIBLE_WITH = "COMPATIBLE_WITH"  # Camera -> Product
    USES_PROTOCOL = "USES_PROTOCOL"  # Camera -> Protocol
    HAS_PORT = "HAS_PORT"  # Camera/Product -> Port
    SUPPORTS_CONTROL = "SUPPORTS_CONTROL"  # Camera -> Control

    # Product relationships
    SUPPORTS_PROTOCOL = "SUPPORTS_PROTOCOL"  # Product -> Protocol

    # Issue/Solution relationships
    AFFECTS = "AFFECTS"  # Issue -> Product/Camera
    RESOLVED_BY = "RESOLVED_BY"  # Issue -> Solution
    MENTIONED_IN = "MENTIONED_IN"  # Solution/Entity -> Chunk

    # Firmware relationships
    FOR_PRODUCT = "FOR_PRODUCT"  # Firmware -> Product
    FIXES = "FIXES"  # Firmware -> Issue

    # Chunk relationships
    MENTIONS = "MENTIONS"  # Chunk -> Any entity


class GraphNodeBase(BaseModel):
    """Base class for all graph nodes."""

    id: str | None = Field(default=None, description="Neo4j internal node ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    human_edited: bool = Field(default=False, description="Protected from automated updates")


class GraphCamera(GraphNodeBase):
    """Camera node in the knowledge graph."""

    name: str = Field(..., description="Camera model name")
    manufacturer: str | None = Field(default=None, description="Camera manufacturer")
    doc_url: str | None = Field(default=None, description="Documentation URL")
    manufacturer_url: str | None = Field(default=None, description="Manufacturer product page")
    camera_db_id: int | None = Field(default=None, description="SQLite camera database ID")


class GraphProduct(GraphNodeBase):
    """Cyanview product node (RIO, RCP, CI0, VP4, Live Composer)."""

    name: str = Field(..., description="Product name")
    product_type: str | None = Field(default=None, description="Product type/category")
    doc_url: str | None = Field(default=None, description="Documentation URL")


class GraphProtocol(GraphNodeBase):
    """Communication protocol node (VISCA, LANC, Pelco, Canon XC, etc.)."""

    name: str = Field(..., description="Protocol name")
    protocol_type: str | None = Field(
        default=None, description="Protocol category (serial, network, proprietary)"
    )
    description: str | None = Field(default=None, description="Protocol description")


class GraphPort(GraphNodeBase):
    """Physical port/connector node (Ethernet, RS-485, HDMI, USB, etc.)."""

    name: str = Field(..., description="Port/connector name")
    port_type: str | None = Field(
        default=None, description="Port category (network, serial, video)"
    )


class GraphControl(GraphNodeBase):
    """Camera control function node (Iris, Focus, Zoom, etc.)."""

    name: str = Field(..., description="Control function name")
    description: str | None = Field(default=None, description="Control description")
    control_type: str | None = Field(
        default=None, description="Control category (exposure, lens, color)"
    )


class GraphIssue(GraphNodeBase):
    """Issue/problem node extracted from support cases."""

    description: str = Field(..., description="Issue description")
    symptoms: list[str] = Field(default_factory=list, description="Symptom keywords")
    error_codes: list[str] = Field(default_factory=list, description="Related error codes")


class GraphSolution(GraphNodeBase):
    """Solution/resolution node extracted from support cases."""

    description: str = Field(..., description="Solution description")
    steps: list[str] = Field(default_factory=list, description="Resolution steps")
    verified: bool = Field(default=False, description="Solution verified by support team")


class GraphFirmware(GraphNodeBase):
    """Firmware version node."""

    version: str = Field(..., description="Firmware version string")
    release_date: datetime | None = Field(default=None, description="Release date")
    changelog: str | None = Field(default=None, description="Changelog summary")


class GraphChunk(GraphNodeBase):
    """Reference node linking to Qdrant vector chunks."""

    chunk_id: str = Field(..., description="Qdrant point UUID")
    collection: str = Field(..., description="Qdrant collection name")
    title: str | None = Field(default=None, description="Chunk title/source")
    source_url: str | None = Field(default=None, description="Source document URL")


class GraphRelationship(BaseModel):
    """Relationship between two graph nodes."""

    source_type: str = Field(..., description="Source node type (e.g., 'Camera')")
    source_id: str = Field(..., description="Source node identifier (name or ID)")
    relationship: RelationType = Field(..., description="Relationship type")
    target_type: str = Field(..., description="Target node type")
    target_id: str = Field(..., description="Target node identifier")
    properties: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Relationship properties"
    )


class EntityExtractionResult(BaseModel):
    """Result of LLM entity extraction from text."""

    cameras: list[GraphCamera] = Field(default_factory=list)
    products: list[GraphProduct] = Field(default_factory=list)
    protocols: list[GraphProtocol] = Field(default_factory=list)
    ports: list[GraphPort] = Field(default_factory=list)
    controls: list[GraphControl] = Field(default_factory=list)
    issues: list[GraphIssue] = Field(default_factory=list)
    solutions: list[GraphSolution] = Field(default_factory=list)
    firmware_versions: list[GraphFirmware] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)

    @property
    def entity_count(self) -> int:
        """Total number of entities extracted."""
        return (
            len(self.cameras)
            + len(self.products)
            + len(self.protocols)
            + len(self.ports)
            + len(self.controls)
            + len(self.issues)
            + len(self.solutions)
            + len(self.firmware_versions)
        )

    @property
    def relationship_count(self) -> int:
        """Total number of relationships extracted."""
        return len(self.relationships)


class GraphEnrichmentContext(BaseModel):
    """Context enrichment from graph traversal for a query."""

    related_cameras: list[GraphCamera] = Field(default_factory=list)
    related_products: list[GraphProduct] = Field(default_factory=list)
    related_protocols: list[GraphProtocol] = Field(default_factory=list)
    related_issues: list[GraphIssue] = Field(default_factory=list)
    related_solutions: list[GraphSolution] = Field(default_factory=list)
    related_firmware: list[GraphFirmware] = Field(default_factory=list)
    paths: list[str] = Field(
        default_factory=list, description="Human-readable relationship paths"
    )

    def to_context_string(self) -> str:
        """Format enrichment as context string for Claude synthesis."""
        parts = []

        if self.related_cameras:
            camera_names = [c.name for c in self.related_cameras]
            parts.append(f"Related cameras: {', '.join(camera_names)}")

        if self.related_products:
            product_names = [p.name for p in self.related_products]
            parts.append(f"Related Cyanview products: {', '.join(product_names)}")

        if self.related_protocols:
            protocol_names = [p.name for p in self.related_protocols]
            parts.append(f"Communication protocols: {', '.join(protocol_names)}")

        if self.related_issues:
            issue_descs = [i.description[:100] for i in self.related_issues[:3]]
            parts.append(f"Known issues: {'; '.join(issue_descs)}")

        if self.related_solutions:
            solution_descs = [s.description[:100] for s in self.related_solutions[:3]]
            parts.append(f"Known solutions: {'; '.join(solution_descs)}")

        if self.related_firmware:
            fw_versions = [f.version for f in self.related_firmware[:5]]
            parts.append(f"Firmware versions: {', '.join(fw_versions)}")

        if self.paths:
            parts.append(f"Relationships: {'; '.join(self.paths[:5])}")

        return "\n".join(parts) if parts else ""
