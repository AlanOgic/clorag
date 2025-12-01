"""Extract camera compatibility information using LLM analysis."""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

import anthropic
import httpx

from clorag.config import get_settings
from clorag.models.camera import CameraCreate, CameraEnrichment
from clorag.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

EXTRACTION_PROMPT = """Analyze this Cyanview documentation or support content and extract camera compatibility information.

Cyanview makes camera control equipment (RCP, RIO, CI0, VP4). The documentation contains info about which cameras can be controlled and how.

For each camera model mentioned with control/compatibility info, extract:
- model: The camera model name ONLY, without manufacturer prefix (e.g., "HDC-5500", "C500 Mark II", "URSA Mini Pro 12K")
- manufacturer: The camera manufacturer name ONLY (e.g., "Sony", "Canon", "Panasonic", "Blackmagic", "ARRI", "RED", "Grass Valley", "Ikegami", "Hitachi", "JVC")
- Control ports (RS-422, RS-232, Ethernet, GPIO, LANC, etc.)
- Protocols (VISCA, Sony RCP, Panasonic, LANC, IP, Blackmagic SDI, etc.)
- Supported controls (Iris, Gain, Shutter, White Balance, ND, Focus, Zoom, Color, Gamma, etc.)
- Important notes (firmware requirements, cable requirements, limitations, specific RIO/RCP needed)

Return ONLY a JSON array of cameras found. If no cameras mentioned, return [].

CRITICAL RULES:
- "model" field must NOT include the manufacturer name - WRONG: "Sony HDC-5500", RIGHT: "HDC-5500"
- "manufacturer" field must be the brand name ONLY - WRONG: "Sony HDC-5500", RIGHT: "Sony"
- Only extract cameras with actual compatibility information (ports, protocols, or controls)
- Do NOT include generic mentions like "any camera" or "most cameras"
- Use exact model names when available
- If a camera family is mentioned (e.g., "Sony HDC series"), list specific models if available
- Merge duplicates - if same camera mentioned multiple times, combine the info

Example output:
[
  {{
    "model": "HDC-5500",
    "manufacturer": "Sony",
    "ports": ["RS-422", "Ethernet"],
    "protocols": ["Sony RCP", "IP"],
    "supported_controls": ["Iris", "Gain", "Shutter", "White Balance", "ND"],
    "notes": ["Requires RIO for serial connection", "IP control available with firmware 2.0+"]
  }},
  {{
    "model": "C500 Mark II",
    "manufacturer": "Canon",
    "ports": ["Ethernet"],
    "protocols": ["IP"],
    "supported_controls": ["Iris", "ISO", "Shutter", "White Balance"],
    "notes": ["Use Cinema RAW Light for best results"]
  }},
  {{
    "model": "URSA Mini Pro 12K",
    "manufacturer": "Blackmagic",
    "ports": ["SDI", "Ethernet"],
    "protocols": ["Blackmagic SDI", "IP"],
    "supported_controls": ["Iris", "ISO", "Shutter", "White Balance", "ND"],
    "notes": []
  }}
]

Content to analyze:
{content}"""

ENRICHMENT_PROMPT = """Extract technical specifications from this camera product page.

Focus on:
- Remote control capabilities (protocols, ports, APIs)
- Connectivity options (SDI, HDMI, Ethernet, Serial)
- Any technical specs relevant to camera control

Return JSON:
{{
  "specs": {{"key": "value", ...}},
  "features": ["feature1", "feature2", ...],
  "connectivity": ["port1", "port2", ...],
  "remote_control": ["protocol1", "capability1", ...]
}}

Content:
{content}"""


KNOWN_MANUFACTURERS = [
    "Sony", "Canon", "Panasonic", "Blackmagic", "ARRI", "RED", "Grass Valley",
    "Ikegami", "Hitachi", "JVC", "Fujifilm", "Nikon", "Z CAM", "Kinefinity",
    "Ross", "AJA", "BMD", "GoPro", "DJI", "Atomos", "Marshall", "PTZOptics",
]


def clean_model_name(model: str, manufacturer: str | None) -> str:
    """Remove manufacturer prefix from model name if present.

    Examples:
        clean_model_name("Sony HDC-5500", "Sony") -> "HDC-5500"
        clean_model_name("HDC-5500", "Sony") -> "HDC-5500"
        clean_model_name("Blackmagic URSA Mini", "Blackmagic") -> "URSA Mini"
    """
    if not model:
        return model

    model = model.strip()

    # Remove known manufacturer prefixes
    for mfr in KNOWN_MANUFACTURERS:
        if model.lower().startswith(mfr.lower() + " "):
            model = model[len(mfr):].strip()
            break
        # Also check with "Design" suffix (Blackmagic Design)
        if model.lower().startswith(mfr.lower() + " design "):
            model = model[len(mfr) + 8:].strip()
            break

    # Also check the specific manufacturer if provided
    if manufacturer:
        if model.lower().startswith(manufacturer.lower() + " "):
            model = model[len(manufacturer):].strip()

    return model


def normalize_manufacturer(manufacturer: str | None) -> str | None:
    """Normalize manufacturer name to canonical form.

    Examples:
        normalize_manufacturer("SONY") -> "Sony"
        normalize_manufacturer("blackmagic design") -> "Blackmagic"
        normalize_manufacturer("grass valley") -> "Grass Valley"
    """
    if not manufacturer:
        return None

    manufacturer = manufacturer.strip()

    # Handle common variations
    lower = manufacturer.lower()

    if "blackmagic" in lower:
        return "Blackmagic"
    if "grass" in lower and "valley" in lower:
        return "Grass Valley"
    if lower == "bmd":
        return "Blackmagic"
    if lower == "gv":
        return "Grass Valley"

    # Find matching known manufacturer (case-insensitive)
    for mfr in KNOWN_MANUFACTURERS:
        if lower == mfr.lower():
            return mfr

    # Return with first letter capitalized for each word
    return " ".join(word.capitalize() for word in manufacturer.split())


class CameraExtractor:
    """Extract camera compatibility info using Claude Haiku."""

    def __init__(self) -> None:
        """Initialize the extractor with Anthropic client."""
        settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for web scraping."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; CyanviewBot/1.0; +https://cyanview.com)"
                },
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def extract_cameras(self, content: str, doc_url: str | None = None) -> list[CameraCreate]:
        """Extract camera info from text content.

        Args:
            content: Text content to analyze (documentation or support case).
            doc_url: Optional URL source of the content.

        Returns:
            List of CameraCreate objects extracted from content.
        """
        if not content or len(content.strip()) < 50:
            return []

        # Truncate very long content
        max_content = 12000
        if len(content) > max_content:
            content = content[:max_content] + "\n...[truncated]"

        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT.format(content=content),
                    }
                ],
            )

            result_text = response.content[0].text.strip()

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\[[\s\S]*\]", result_text)
            if not json_match:
                logger.debug("No cameras found in content")
                return []

            cameras_data = json.loads(json_match.group())

            cameras = []
            for cam_data in cameras_data:
                # Support both "model" (new format) and "name" (old format)
                raw_model = cam_data.get("model") or cam_data.get("name")
                if not raw_model:
                    continue

                # Normalize manufacturer
                raw_manufacturer = cam_data.get("manufacturer")
                manufacturer = normalize_manufacturer(raw_manufacturer)

                # Clean model name (remove manufacturer prefix if present)
                model = clean_model_name(raw_model, manufacturer)

                if not model:
                    continue

                camera = CameraCreate(
                    name=model,
                    manufacturer=manufacturer,
                    ports=cam_data.get("ports", []),
                    protocols=cam_data.get("protocols", []),
                    supported_controls=cam_data.get("supported_controls", []),
                    notes=cam_data.get("notes", []),
                    doc_url=doc_url,
                )
                cameras.append(camera)

            logger.info(
                "Extracted cameras from content",
                count=len(cameras),
                cameras=[c.name for c in cameras],
            )
            return cameras

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse camera extraction JSON", error=str(e))
            return []
        except anthropic.APIError as e:
            logger.error("Anthropic API error during extraction", error=str(e))
            return []

    async def extract_from_batch(
        self, contents: list[tuple[str, str | None]], concurrency: int = 5
    ) -> list[CameraCreate]:
        """Extract cameras from multiple documents concurrently.

        Args:
            contents: List of (content, doc_url) tuples.
            concurrency: Max concurrent extractions.

        Returns:
            Deduplicated list of CameraCreate objects.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def extract_with_limit(content: str, url: str | None) -> list[CameraCreate]:
            async with semaphore:
                return await self.extract_cameras(content, url)

        tasks = [extract_with_limit(content, url) for content, url in contents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_cameras: dict[str, CameraCreate] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Extraction task failed", error=str(result))
                continue
            for camera in result:
                # Create deduplication key: manufacturer + normalized model name
                # This handles cases like "HDC-5500" vs "hdc-5500" vs "HDC 5500"
                model_normalized = camera.name.lower().replace("-", "").replace(" ", "").replace("_", "")
                mfr_normalized = (camera.manufacturer or "unknown").lower()
                key = f"{mfr_normalized}:{model_normalized}"

                if key in all_cameras:
                    existing = all_cameras[key]
                    # Merge arrays, keeping first non-empty values
                    all_cameras[key] = CameraCreate(
                        name=camera.name if len(camera.name) > len(existing.name) else existing.name,
                        manufacturer=camera.manufacturer or existing.manufacturer,
                        ports=list(set(existing.ports + camera.ports)),
                        protocols=list(set(existing.protocols + camera.protocols)),
                        supported_controls=list(
                            set(existing.supported_controls + camera.supported_controls)
                        ),
                        notes=list(set(existing.notes + camera.notes)),
                        doc_url=camera.doc_url or existing.doc_url,
                        manufacturer_url=camera.manufacturer_url or existing.manufacturer_url,
                    )
                else:
                    all_cameras[key] = camera

        logger.info("Batch extraction complete", total_cameras=len(all_cameras))
        return list(all_cameras.values())

    async def enrich_from_manufacturer(self, camera_name: str, manufacturer_url: str) -> CameraEnrichment | None:
        """Scrape manufacturer website for additional camera specs.

        Args:
            camera_name: Camera model name for context.
            manufacturer_url: URL to the manufacturer product page.

        Returns:
            CameraEnrichment with scraped data, or None if failed.
        """
        try:
            client = await self._get_http_client()
            response = await client.get(manufacturer_url)
            response.raise_for_status()

            # Get page content
            html_content = response.text

            # Strip HTML tags for simpler processing
            text_content = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html_content)
            text_content = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text_content)
            text_content = re.sub(r"<[^>]+>", " ", text_content)
            text_content = re.sub(r"\s+", " ", text_content).strip()

            # Truncate
            if len(text_content) > 8000:
                text_content = text_content[:8000]

            # Use LLM to extract specs
            llm_response = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": ENRICHMENT_PROMPT.format(content=text_content),
                    }
                ],
            )

            result_text = llm_response.content[0].text.strip()

            # Extract JSON
            json_match = re.search(r"\{[\s\S]*\}", result_text)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            enrichment = CameraEnrichment(
                specs=data.get("specs", {}),
                features=data.get("features", []),
                connectivity=data.get("connectivity", []),
                remote_control=data.get("remote_control", []),
                source_url=manufacturer_url,
            )

            logger.info(
                "Enriched camera from manufacturer",
                camera=camera_name,
                url=manufacturer_url,
            )
            return enrichment

        except httpx.HTTPError as e:
            logger.warning(
                "Failed to fetch manufacturer page",
                url=manufacturer_url,
                error=str(e),
            )
            return None
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse enrichment JSON", error=str(e))
            return None
        except anthropic.APIError as e:
            logger.error("Anthropic API error during enrichment", error=str(e))
            return None

    async def search_manufacturer_url(self, camera_name: str, manufacturer: str | None = None) -> str | None:
        """Try to find the manufacturer product page URL for a camera.

        This uses a simple heuristic based on known manufacturer URL patterns.

        Args:
            camera_name: Camera model name.
            manufacturer: Manufacturer name if known.

        Returns:
            URL string or None if not found.
        """
        if not manufacturer:
            return None

        manufacturer_lower = manufacturer.lower()

        # Known manufacturer product page patterns
        url_patterns: dict[str, str] = {
            "sony": "https://pro.sony/products/broadcast-camcorders",
            "canon": "https://www.usa.canon.com/shop/c/professional-video-cameras",
            "panasonic": "https://na.panasonic.com/us/video-production",
            "blackmagic": "https://www.blackmagicdesign.com/products",
            "arri": "https://www.arri.com/en/camera-systems",
            "red": "https://www.red.com/cameras",
            "grass valley": "https://www.grassvalley.com/products/cameras",
        }

        # Return base URL - actual page search would need more sophisticated scraping
        for mfr, url in url_patterns.items():
            if mfr in manufacturer_lower:
                logger.debug("Found manufacturer URL pattern", manufacturer=manufacturer, url=url)
                return url

        return None
