"""LLM-based entity extraction for GraphRAG knowledge graph population."""

from __future__ import annotations

import asyncio
import json
import re

import anthropic
import structlog

from clorag.config import get_settings
from clorag.graph.schema import (
    EntityExtractionResult,
    GraphCamera,
    GraphControl,
    GraphFirmware,
    GraphIssue,
    GraphPort,
    GraphProduct,
    GraphProtocol,
    GraphRelationship,
    GraphSolution,
    RelationType,
)
from clorag.services.prompt_manager import get_prompt

logger = structlog.get_logger()

# Known Cyanview products for matching
CYANVIEW_PRODUCTS = ["RIO", "RCP", "CI0", "VP4", "Live Composer", "CY-CBL"]

# Known protocols for normalization
KNOWN_PROTOCOLS = [
    "VISCA", "LANC", "Pelco", "Canon XC", "Sony RCP", "Panasonic", "Blackmagic SDI",
    "IP", "NDI", "SRT", "RTMP", "HTTP API", "RS-422", "RS-232", "RS-485",
    "GPIO", "GPI", "GPO", "Tally", "Ethernet", "USB", "HDMI", "SDI",
]

# Known control functions
KNOWN_CONTROLS = [
    "Iris", "Focus", "Zoom", "Gain", "Shutter", "White Balance", "ND Filter",
    "ISO", "Gamma", "Color Matrix", "Knee", "Black Level", "Detail", "Skin Tone",
    "Pan", "Tilt", "PTZ", "Preset", "Tally", "Record", "Streaming",
]


class EntityExtractor:
    """Extract entities and relationships from text using Claude Sonnet."""

    def __init__(self) -> None:
        """Initialize the extractor with Anthropic client."""
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self._model = settings.sonnet_model

    async def extract_entities(
        self, content: str, source_chunk_id: str | None = None
    ) -> EntityExtractionResult:
        """Extract entities and relationships from text content.

        Args:
            content: Text content to analyze.
            source_chunk_id: Optional Qdrant chunk ID for linking.

        Returns:
            EntityExtractionResult with extracted entities and relationships.
        """
        if not content or len(content.strip()) < 50:
            return EntityExtractionResult()

        # Truncate very long content
        max_content = 10000
        if len(content) > max_content:
            content = content[:max_content] + "\n...[truncated]"

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": get_prompt("graph.entity_extractor", content=content),
                    }
                ],
            )

            result_text = response.content[0].text.strip()

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\{[\s\S]*\}", result_text)
            if not json_match:
                logger.debug("No entities found in content")
                return EntityExtractionResult()

            data = json.loads(json_match.group())

            result = self._parse_extraction_result(data, source_chunk_id)

            logger.info(
                "entity_extraction_complete",
                entities=result.entity_count,
                relationships=result.relationship_count,
            )
            return result

        except json.JSONDecodeError as e:
            logger.warning("entity_extraction_json_error", error=str(e))
            return EntityExtractionResult()
        except anthropic.APIError as e:
            logger.error("entity_extraction_api_error", error=str(e))
            return EntityExtractionResult()

    def _parse_extraction_result(
        self, data: dict, source_chunk_id: str | None
    ) -> EntityExtractionResult:
        """Parse raw LLM output into structured entities."""
        result = EntityExtractionResult()

        # Parse cameras
        for cam in data.get("cameras", []):
            name = self._normalize_name(cam.get("name", ""))
            if name:
                result.cameras.append(GraphCamera(
                    name=name,
                    manufacturer=cam.get("manufacturer"),
                ))

        # Parse products
        for prod in data.get("products", []):
            name = self._normalize_product(prod.get("name", ""))
            if name:
                result.products.append(GraphProduct(
                    name=name,
                    product_type=prod.get("product_type"),
                ))

        # Parse protocols
        for proto in data.get("protocols", []):
            name = self._normalize_protocol(proto.get("name", ""))
            if name:
                result.protocols.append(GraphProtocol(
                    name=name,
                    protocol_type=proto.get("protocol_type"),
                ))

        # Parse ports
        for port in data.get("ports", []):
            name = self._normalize_name(port.get("name", ""))
            if name:
                result.ports.append(GraphPort(
                    name=name,
                    port_type=port.get("port_type"),
                ))

        # Parse controls
        for ctrl in data.get("controls", []):
            name = self._normalize_control(ctrl.get("name", ""))
            if name:
                result.controls.append(GraphControl(
                    name=name,
                    control_type=ctrl.get("control_type"),
                ))

        # Parse issues
        for issue in data.get("issues", []):
            desc = issue.get("description", "")
            if desc and len(desc) > 10:
                result.issues.append(GraphIssue(
                    description=desc[:500],  # Limit length
                    symptoms=issue.get("symptoms", [])[:10],
                    error_codes=issue.get("error_codes", [])[:5],
                ))

        # Parse solutions
        for sol in data.get("solutions", []):
            desc = sol.get("description", "")
            if desc and len(desc) > 10:
                result.solutions.append(GraphSolution(
                    description=desc[:500],
                    steps=sol.get("steps", [])[:10],
                ))

        # Parse firmware
        for fw in data.get("firmware", []):
            version = fw.get("version", "")
            if version:
                result.firmware_versions.append(GraphFirmware(
                    version=version,
                    changelog=fw.get("changelog"),
                ))

        # Parse relationships
        for rel in data.get("relationships", []):
            rel_type_str = rel.get("relationship", "")
            try:
                rel_type = RelationType(rel_type_str)
                result.relationships.append(GraphRelationship(
                    source_type=rel.get("source_type", ""),
                    source_id=rel.get("source_id", ""),
                    relationship=rel_type,
                    target_type=rel.get("target_type", ""),
                    target_id=rel.get("target_id", ""),
                ))
            except ValueError:
                logger.debug("unknown_relationship_type", rel_type=rel_type_str)

        # Add chunk reference if provided
        if source_chunk_id:
            # Create MENTIONS relationships from chunk to entities
            for camera in result.cameras:
                result.relationships.append(GraphRelationship(
                    source_type="Chunk",
                    source_id=source_chunk_id,
                    relationship=RelationType.MENTIONS,
                    target_type="Camera",
                    target_id=camera.name,
                ))

        return result

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name."""
        return name.strip() if name else ""

    def _normalize_product(self, name: str) -> str:
        """Normalize Cyanview product name."""
        if not name:
            return ""

        name = name.strip().upper()

        # Map variations to canonical names
        mappings = {
            "RIO LIVE": "RIO",
            "RIO-LIVE": "RIO",
            "CI-0": "CI0",
            "VP-4": "VP4",
            "LIVE-COMPOSER": "Live Composer",
            "LIVECOMPOSER": "Live Composer",
        }

        for variant, canonical in mappings.items():
            if name == variant or name == variant.upper():
                return canonical

        # Check if it matches a known product
        for product in CYANVIEW_PRODUCTS:
            if name == product.upper():
                return product

        return name

    def _normalize_protocol(self, name: str) -> str:
        """Normalize protocol name."""
        if not name:
            return ""

        name = name.strip()

        # Case-insensitive matching to known protocols
        name_lower = name.lower()
        for proto in KNOWN_PROTOCOLS:
            if name_lower == proto.lower():
                return proto

        return name

    def _normalize_control(self, name: str) -> str:
        """Normalize control function name."""
        if not name:
            return ""

        name = name.strip()

        # Case-insensitive matching to known controls
        name_lower = name.lower()
        for ctrl in KNOWN_CONTROLS:
            if name_lower == ctrl.lower():
                return ctrl

        return name.title()

    async def extract_from_batch(
        self,
        chunks: list[tuple[str, str, str | None]],  # (chunk_id, content, title)
        concurrency: int = 5,
    ) -> EntityExtractionResult:
        """Extract entities from multiple chunks concurrently.

        Args:
            chunks: List of (chunk_id, content, title) tuples.
            concurrency: Max concurrent extractions.

        Returns:
            Merged EntityExtractionResult from all chunks.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def extract_with_limit(
            chunk_id: str, content: str, title: str | None
        ) -> EntityExtractionResult:
            async with semaphore:
                return await self.extract_entities(content, chunk_id)

        tasks = [
            extract_with_limit(chunk_id, content, title)
            for chunk_id, content, title in chunks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        merged = EntityExtractionResult()
        seen_cameras: set[str] = set()
        seen_products: set[str] = set()
        seen_protocols: set[str] = set()
        seen_ports: set[str] = set()
        seen_controls: set[str] = set()

        for result in results:
            if isinstance(result, Exception):
                logger.warning("batch_extraction_task_failed", error=str(result))
                continue

            # Deduplicate entities
            for camera in result.cameras:
                key = f"{camera.manufacturer}:{camera.name}".lower()
                if key not in seen_cameras:
                    seen_cameras.add(key)
                    merged.cameras.append(camera)

            for product in result.products:
                if product.name.lower() not in seen_products:
                    seen_products.add(product.name.lower())
                    merged.products.append(product)

            for protocol in result.protocols:
                if protocol.name.lower() not in seen_protocols:
                    seen_protocols.add(protocol.name.lower())
                    merged.protocols.append(protocol)

            for port in result.ports:
                if port.name.lower() not in seen_ports:
                    seen_ports.add(port.name.lower())
                    merged.ports.append(port)

            for control in result.controls:
                if control.name.lower() not in seen_controls:
                    seen_controls.add(control.name.lower())
                    merged.controls.append(control)

            # Issues and solutions are unique by description
            merged.issues.extend(result.issues)
            merged.solutions.extend(result.solutions)
            merged.firmware_versions.extend(result.firmware_versions)
            merged.relationships.extend(result.relationships)

        logger.info(
            "batch_extraction_complete",
            chunks=len(chunks),
            entities=merged.entity_count,
            relationships=merged.relationship_count,
        )
        return merged
