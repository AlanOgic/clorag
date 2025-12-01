#!/usr/bin/env python3
"""Enrich camera database with official model codes and URLs from manufacturer websites."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

import anthropic
import httpx

from clorag.config import get_settings
from clorag.core.database import get_camera_database
from clorag.models.camera import Camera, CameraUpdate
from clorag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnrichmentResult:
    """Result of camera enrichment search."""

    code_model: str | None = None
    manufacturer_url: str | None = None

# Top manufacturers to enrich
TOP_MANUFACTURERS = [
    "Sony",
    "Canon",
    "Panasonic",
    "Blackmagic",
    "ARRI",
    "RED",
    "Grass Valley",
    "Ikegami",
    "Hitachi",
    "JVC",
    "Marshall",
    "Birddog",
    "Dreamchip",
    "Z CAM",
    "DJI",
]

# Prompt for extracting model code and URL from search results
EXTRACTION_PROMPT = """Given the camera model name "{camera_name}" from manufacturer "{manufacturer}",
analyze the search results and extract:
1. The official manufacturer model code/SKU
2. The official product page URL from the manufacturer's website

The model code is typically:
- An alphanumeric code like "ILME-FX6V" for Sony FX6
- A product reference number from the manufacturer's catalog
- Different from the marketing name (e.g., "Alpha 7S III" marketing name vs "ILCE-7SM3" model code)

The URL should be:
- From the official manufacturer website (e.g., sony.com, canon.com, panasonic.com)
- The product page or specifications page for this specific camera model
- NOT a retailer, review site, or third-party website

Search results:
{search_results}

Return a JSON object with this format:
{{"code_model": "MODEL_CODE_HERE", "manufacturer_url": "URL_HERE"}}

Use null for any field not found. Examples:
{{"code_model": "ILME-FX6V", "manufacturer_url": "https://pro.sony/en_US/products/cinema-line/ilme-fx6v"}}
{{"code_model": "EOS C300 Mark III", "manufacturer_url": null}}

JSON response:"""


# Known model codes for common cameras (manual mapping)
KNOWN_MODEL_CODES: dict[str, dict[str, str]] = {
    "Sony": {
        "FX6": "ILME-FX6V",
        "FX3": "ILME-FX3",
        "FX30": "ILME-FX30",
        "FX9": "PXW-FX9",
        "FS7": "PXW-FS7",
        "A1": "ILCE-1",
        "A7S": "ILCE-7S",
        "A7S3": "ILCE-7SM3",
        "Alpha": "ILCE",
        "Alpha A7S": "ILCE-7S",
        "Alpha FX3": "ILME-FX3",
        "Burano": "MPC-2610",
        "Venice": "MPC-3610",
        "Venice 1": "MPC-3610",
        "Venice 2": "MPC-3620",
        "FR7": "ILME-FR7",
        "BRC-X1000": "BRC-X1000",
        "BRC-X400": "BRC-X400",
        "BRC-H800": "BRC-H800",
        "BRC-H900": "BRC-H900",
        "PXW-Z90": "PXW-Z90V",
        "PXW-Z450": "PXW-Z450",
        "PXW-Z200": "PXW-Z200",
        "PXW-FS5": "PXW-FS5",
        "PXW-FS7": "PXW-FS7",
        "PMW-300": "PMW-300K1",
        "PMW-500": "PMW-500",
        "Z280": "PXW-Z280",
        "HDC-5500": "HDC-5500",
        "HXC-D70": "HXC-D70",
        "ILCE-1": "ILCE-1",
        "ILCE-9M2": "ILCE-9M2",
        "ILME-FX2": "ILME-FX2",
        "ILME-FX3": "ILME-FX3",
        "ILME-FX30": "ILME-FX30",
        "ILME-FX6": "ILME-FX6V",
        "ILX-LR1": "ILX-LR1",
        "FCB-7520": "FCB-EV7520",
        "FCB-8230": "FCB-EV8230",
        "FCB-8530": "FCB-EV8530",
        "FCB-8550": "FCB-EV8550",
        "FCB-H11": "FCB-EH6300",
        "CA-FB70": "CA-FB70",
    },
    "Canon": {
        "C70": "EOS C70",
        "C80": "EOS C80",
        "C100": "EOS C100",
        "C200": "EOS C200",
        "C300": "EOS C300",
        "C300 Mark I": "EOS C300",
        "C300 Mark II": "EOS C300 Mark II",
        "C300 Mark III": "EOS C300 Mark III",
        "C400": "EOS C400",
        "C500 Mark I": "EOS C500",
        "C500 Mark II": "EOS C500 Mark II",
        "C700": "EOS C700",
        "R5": "EOS R5",
        "XF605": "XF605",
        "XF705": "XF705",
        "CR-N100": "CR-N100",
        "CR-N300": "CR-N300",
        "CR-N500": "CR-N500",
        "CR-N700": "CR-N700",
        "CR-X300": "CR-X300",
        "CRN700": "CR-N700",
    },
    "Panasonic": {
        "DC-GH5S": "DC-GH5S",
        "DC-GH7": "DC-GH7",
        "DC-BGH1": "DC-BGH1",
        "DC-BS1H": "DC-BS1H",
        "AU-EVA1": "AU-EVA1",
        "AK-UC4000": "AK-UC4000",
        "AK-UC3300": "AK-UC3300",
        "AK-UCX100": "AK-UCX100",
        "AK-HC3900": "AK-HC3900GSJ",
        "AK-PLV100GSJ": "AK-PLV100GSJ",
        "Varicam": "AU-V35C1G",
        "UE-150": "AW-UE150",
        "UE150": "AW-UE150",
        "UE-100": "AW-UE100",
        "UE-70": "AW-UE70",
        "AW-UB10": "AW-UB10",
        "AW-UB50": "AW-UB50",
        "AG-CX350": "AG-CX350",
    },
    "Blackmagic": {
        "URSA Cine 17K": "URSA Cine 17K 65",
        "URSA Cine 12K LF": "URSA Cine 12K LF",
        "URSA G2": "URSA Broadcast G2",
        "PYXIS 6K": "PYXIS 6K",
        "Pocket Cinema Camera": "BMPCC",
        "MicroStudio": "Micro Studio Camera 4K G2",
        "ATEM mini series": "ATEM Mini",
        "ATEM Production series": "ATEM Production Studio 4K",
        "ATEM Television Studio series": "ATEM Television Studio",
    },
    "ARRI": {
        "Alexa 35": "ALEXA 35",
        "Alexa mini": "ALEXA Mini",
        "Alexa mini LF": "ALEXA Mini LF",
        "Amira": "AMIRA",
        "cforce mini": "cforce mini",
        "cforce mini RF": "cforce mini RF",
        "cforce plus": "cforce plus",
    },
    "RED": {
        "Komodo": "KOMODO 6K",
        "Raptor": "V-RAPTOR",
        "V-Raptor": "V-RAPTOR 8K VV",
    },
}

# Known manufacturer URLs for common cameras
KNOWN_MANUFACTURER_URLS: dict[str, dict[str, str]] = {
    "Sony": {
        "FX6": "https://pro.sony/en_US/products/cinema-line/ilme-fx6v",
        "FX3": "https://pro.sony/en_US/products/cinema-line/ilme-fx3",
        "FX30": "https://pro.sony/en_US/products/cinema-line/ilme-fx30",
        "FX9": "https://pro.sony/en_US/products/cinema-line/pxw-fx9",
        "FR7": "https://pro.sony/en_US/products/cinema-line/ilme-fr7",
        "Venice": "https://pro.sony/en_US/products/cinema-line/venice",
        "Venice 2": "https://pro.sony/en_US/products/cinema-line/venice-2",
        "Burano": "https://pro.sony/en_US/products/cinema-line/burano",
        "BRC-X1000": "https://pro.sony/en_US/products/ptz-network-cameras/brc-x1000",
        "BRC-X400": "https://pro.sony/en_US/products/ptz-network-cameras/brc-x400",
        "HDC-5500": "https://pro.sony/en_US/products/system-cameras/hdc-5500",
    },
    "Canon": {
        "C70": "https://www.usa.canon.com/shop/p/eos-c70",
        "C80": "https://www.usa.canon.com/shop/p/eos-c80",
        "C300 Mark III": "https://www.usa.canon.com/shop/p/eos-c300-mark-iii",
        "C400": "https://www.usa.canon.com/shop/p/eos-c400",
        "C500 Mark II": "https://www.usa.canon.com/shop/p/eos-c500-mark-ii",
        "CR-N500": "https://www.usa.canon.com/shop/p/cr-n500",
        "CR-N700": "https://www.usa.canon.com/shop/p/cr-n700",
        "CR-N300": "https://www.usa.canon.com/shop/p/cr-n300",
        "CR-N100": "https://www.usa.canon.com/shop/p/cr-n100",
    },
    "Panasonic": {
        "AW-UE150": "https://na.panasonic.com/us/audio-video-solutions/broadcast-cinema-pro-video/ptz-remote-camera-systems/aw-ue150",
        "AW-UE100": "https://na.panasonic.com/us/audio-video-solutions/broadcast-cinema-pro-video/ptz-remote-camera-systems/aw-ue100",
        "AK-UC4000": "https://na.panasonic.com/us/audio-video-solutions/broadcast-cinema-pro-video/studio-cameras/ak-uc4000",
        "DC-GH7": "https://shop.panasonic.com/cameras/mirrorless-cameras/dc-gh7",
        "DC-BGH1": "https://shop.panasonic.com/cameras/mirrorless-cameras/dc-bgh1",
        "DC-BS1H": "https://shop.panasonic.com/cameras/mirrorless-cameras/dc-bs1h",
        "AU-EVA1": "https://na.panasonic.com/us/audio-video-solutions/broadcast-cinema-pro-video/cinema-cameras/au-eva1",
    },
    "Blackmagic": {
        "PYXIS 6K": "https://www.blackmagicdesign.com/products/blackmagicpyxis",
        "URSA Cine 17K": "https://www.blackmagicdesign.com/products/blackmagicursacine",
        "URSA Cine 12K LF": "https://www.blackmagicdesign.com/products/blackmagicursacine",
        "URSA G2": "https://www.blackmagicdesign.com/products/blackmagicursabroadcast",
        "Pocket Cinema Camera": "https://www.blackmagicdesign.com/products/blackmagicpocketcinemacamera",
    },
    "ARRI": {
        "Alexa 35": "https://www.arri.com/en/camera-systems/cameras/alexa-35",
        "Alexa mini": "https://www.arri.com/en/camera-systems/cameras/alexa-mini",
        "Alexa mini LF": "https://www.arri.com/en/camera-systems/cameras/alexa-mini-lf",
        "Amira": "https://www.arri.com/en/camera-systems/cameras/amira",
    },
    "RED": {
        "Komodo": "https://www.red.com/komodo",
        "V-Raptor": "https://www.red.com/v-raptor",
    },
}


def get_known_enrichment(camera: Camera) -> EnrichmentResult:
    """Get enrichment data from known mappings.

    Args:
        camera: Camera to look up.

    Returns:
        EnrichmentResult with code_model and/or manufacturer_url if found.
    """
    result = EnrichmentResult()

    if not camera.manufacturer:
        return result

    # Get code_model from known mappings
    manufacturer_codes = KNOWN_MODEL_CODES.get(camera.manufacturer, {})
    result.code_model = manufacturer_codes.get(camera.name)

    # Get manufacturer_url from known mappings
    manufacturer_urls = KNOWN_MANUFACTURER_URLS.get(camera.manufacturer, {})
    result.manufacturer_url = manufacturer_urls.get(camera.name)

    return result


async def search_enrichment(
    camera: Camera,
    http_client: httpx.AsyncClient,
    anthropic_client: anthropic.AsyncAnthropic,
    searxng_url: str = "https://search.sapti.me",
) -> EnrichmentResult:
    """Search for camera enrichment data (model code and manufacturer URL).

    First checks known mappings, then falls back to web search.

    Args:
        camera: Camera to search for.
        http_client: HTTP client for web requests.
        anthropic_client: Anthropic client for LLM extraction.

    Returns:
        EnrichmentResult with code_model and/or manufacturer_url.
    """
    result = EnrichmentResult()

    if not camera.manufacturer:
        return result

    # First check known mappings
    known = get_known_enrichment(camera)
    if known.code_model:
        result.code_model = known.code_model
        logger.info("Found known model code", camera=camera.name, code_model=known.code_model)
    if known.manufacturer_url:
        result.manufacturer_url = known.manufacturer_url
        logger.info("Found known URL", camera=camera.name, url=known.manufacturer_url)

    # If we have both, return early
    if result.code_model and result.manufacturer_url:
        return result

    # Build search query for web search
    manufacturer_domain = _get_manufacturer_domain(camera.manufacturer)
    search_query = f"{camera.manufacturer} {camera.name} official product page specifications"
    if manufacturer_domain:
        search_query += f" site:{manufacturer_domain}"

    try:
        # Use SearXNG instance (configurable via SEARXNG_URL env var)
        response = await http_client.get(
            f"{searxng_url}/search",
            params={
                "q": search_query,
                "format": "json",
                "categories": "general",
            },
            headers={"User-Agent": "Mozilla/5.0 (compatible; CloragBot/1.0)"},
        )
        response.raise_for_status()

        # Parse JSON response from SearXNG
        search_data = response.json()
        results = search_data.get("results", [])

        # Extract URLs, titles, and snippets from results
        urls = [r.get("url", "") for r in results[:5]]
        titles = [r.get("title", "") for r in results[:5]]
        snippets = [r.get("content", "") for r in results[:5]]

        # Clean HTML tags
        def clean_html(text: str) -> str:
            return re.sub(r"<[^>]+>", "", text).strip()

        search_results = []
        for i, (title, snippet, url) in enumerate(zip(titles, snippets, urls, strict=False)):
            search_results.append(f"Title: {clean_html(title)}\nURL: {url}\nSnippet: {clean_html(snippet)}")

        if not search_results:
            logger.debug("No search results found", camera=camera.name)
            return result

        search_text = "\n---\n".join(search_results)

        # Use LLM to extract model code and URL
        llm_response = await anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(
                        camera_name=camera.name,
                        manufacturer=camera.manufacturer,
                        search_results=search_text,
                    ),
                }
            ],
        )

        response_text = llm_response.content[0].text.strip()

        # Parse JSON response
        try:
            # Handle potential markdown code blocks
            if "```" in response_text:
                response_text = re.sub(r"```json?\s*", "", response_text)
                response_text = re.sub(r"```\s*", "", response_text)

            parsed = json.loads(response_text)

            # Only update if we don't already have values from known mappings
            if not result.code_model and parsed.get("code_model"):
                code_model = parsed["code_model"]
                if code_model and code_model != "null" and len(code_model) < 50:
                    result.code_model = code_model
                    logger.info("Found model code via search", camera=camera.name, code_model=code_model)

            if not result.manufacturer_url and parsed.get("manufacturer_url"):
                url = parsed["manufacturer_url"]
                if url and url != "null" and url.startswith("http"):
                    result.manufacturer_url = url
                    logger.info("Found URL via search", camera=camera.name, url=url)

        except json.JSONDecodeError:
            # Fallback: try to extract just the model code from plain text
            if not result.code_model and response_text and "NOT_FOUND" not in response_text:
                # Try to extract first word/code from response
                match = re.match(r'^[\w\-]+', response_text)
                if match and len(match.group()) < 50:
                    result.code_model = match.group()
                    logger.info("Found model code via search (fallback)", camera=camera.name, code_model=result.code_model)

        return result

    except httpx.HTTPError as e:
        logger.debug("HTTP error during search", camera=camera.name, error=str(e))
        return result
    except anthropic.APIError as e:
        logger.warning("Anthropic API error", camera=camera.name, error=str(e))
        return result


def _get_manufacturer_domain(manufacturer: str) -> str:
    """Get the official domain for a manufacturer."""
    domains = {
        "Sony": "sony.com",
        "Canon": "canon.com",
        "Panasonic": "panasonic.com",
        "Blackmagic": "blackmagicdesign.com",
        "ARRI": "arri.com",
        "RED": "red.com",
        "Grass Valley": "grassvalley.com",
        "Ikegami": "ikegami.com",
        "Hitachi": "hitachi-kokusai.com",
        "JVC": "jvc.com",
        "Marshall": "marshall-usa.com",
        "Birddog": "birddog.tv",
        "Dreamchip": "dreamchip.de",
        "Z CAM": "z-cam.com",
        "DJI": "dji.com",
    }
    return domains.get(manufacturer, "")


async def enrich_cameras(
    manufacturers: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    enrich_urls: bool = True,
) -> int:
    """Enrich cameras with model codes and URLs from manufacturer websites.

    Args:
        manufacturers: List of manufacturers to process. Defaults to TOP_MANUFACTURERS.
        limit: Maximum number of cameras to process.
        dry_run: If True, don't update database, just log what would be done.
        enrich_urls: If True, also enrich manufacturer_url field.

    Returns:
        Number of cameras enriched.
    """
    if manufacturers is None:
        manufacturers = TOP_MANUFACTURERS

    settings = get_settings()
    db = get_camera_database()

    # Initialize clients
    http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    enriched_count = 0
    processed_count = 0

    try:
        for manufacturer in manufacturers:
            cameras = db.list_cameras(manufacturer=manufacturer)

            # Filter cameras needing enrichment (missing code_model or manufacturer_url)
            if enrich_urls:
                cameras_to_enrich = [c for c in cameras if not c.code_model or not c.manufacturer_url]
            else:
                cameras_to_enrich = [c for c in cameras if not c.code_model]

            logger.info(
                "Processing manufacturer",
                manufacturer=manufacturer,
                total=len(cameras),
                to_enrich=len(cameras_to_enrich),
            )

            for camera in cameras_to_enrich:
                if limit and processed_count >= limit:
                    logger.info("Reached limit", limit=limit)
                    break

                processed_count += 1

                # Search for enrichment data
                enrichment = await search_enrichment(
                    camera, http_client, anthropic_client, settings.searxng_url
                )

                # Build update with new data (only update missing fields)
                update_data: dict[str, str | None] = {}
                if enrichment.code_model and not camera.code_model:
                    # Check if this code_model already exists (must be unique)
                    existing = db.get_camera_by_code_model(enrichment.code_model)
                    if existing and existing.id != camera.id:
                        logger.warning(
                            "code_model already exists - skipping",
                            camera=camera.name,
                            code_model=enrichment.code_model,
                            existing_camera=existing.name,
                        )
                    else:
                        update_data["code_model"] = enrichment.code_model
                if enrich_urls and enrichment.manufacturer_url and not camera.manufacturer_url:
                    update_data["manufacturer_url"] = enrichment.manufacturer_url

                if update_data:
                    if dry_run:
                        logger.info(
                            "[DRY RUN] Would update camera",
                            camera=camera.name,
                            **update_data,
                        )
                    else:
                        # Update database
                        db.update_camera(camera.id, CameraUpdate(**update_data))
                        logger.info("Updated camera", camera=camera.name, **update_data)

                    enriched_count += 1

                # Rate limiting - be nice to search engines
                await asyncio.sleep(2)

            if limit and processed_count >= limit:
                break

    finally:
        await http_client.aclose()

    logger.info(
        "Enrichment complete",
        processed=processed_count,
        enriched=enriched_count,
    )

    return enriched_count


async def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Enrich camera database with model codes and URLs")
    parser.add_argument(
        "--manufacturer",
        "-m",
        action="append",
        help="Manufacturer(s) to process (can be specified multiple times)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Maximum number of cameras to process",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Don't update database, just show what would be done",
    )
    parser.add_argument(
        "--no-urls",
        action="store_true",
        help="Only enrich model codes, skip manufacturer URLs",
    )

    args = parser.parse_args()

    await enrich_cameras(
        manufacturers=args.manufacturer,
        limit=args.limit,
        dry_run=args.dry_run,
        enrich_urls=not args.no_urls,
    )


if __name__ == "__main__":
    asyncio.run(main())
