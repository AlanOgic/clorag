"""Verify and correct camera database entries using web search arbitration.

Scans cameras for issues (wrong manufacturer, missing data, suspicious entries)
and verifies against web sources via SearxNG + Jina Reader + LLM.

Pipeline: Load cameras → Detect issues → Web search → LLM arbitration → Update DB.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import anthropic
import httpx

from clorag.config import get_settings
from clorag.core.database import CameraDatabase, get_camera_database
from clorag.models.camera import Camera, CameraUpdate
from clorag.scripts.enrich_model_codes import (
    _fetch_with_jina,
    _get_manufacturer_domains,
    _pick_best_urls,
)
from clorag.services.prompt_manager import get_prompt
from clorag.utils.logger import get_logger

logger = get_logger(__name__)

JINA_READER_URL = "https://r.jina.ai/"


@dataclass
class VerificationResult:
    """Result of verifying a single camera."""

    camera_id: int
    camera_name: str
    issues: list[str] = field(default_factory=list)
    corrections: dict[str, str | None] = field(default_factory=dict)
    verified: bool = False
    source: str = "unverified"


def detect_issues(camera: Camera, all_cameras: list[Camera]) -> list[str]:
    """Detect potential issues with a camera entry.

    Returns list of issue descriptions, empty if no issues found.
    """
    issues: list[str] = []

    # Missing manufacturer
    if not camera.manufacturer:
        issues.append("missing_manufacturer")

    # Missing code_model
    if not camera.code_model:
        issues.append("missing_code_model")

    # Manufacturer mismatch: name contains known prefix from another manufacturer
    _manufacturer_prefixes = {
        "Sony": ["ilce-", "ilme-", "pxw-", "hdc-", "hxc-", "brc-", "src-", "fr"],
        "Panasonic": ["aw-", "ag-", "dc-"],
        "Canon": ["cr-", "eos", "xl-"],
        "Blackmagic": ["ursa", "pyxis", "micro"],
        "ARRI": ["alexa", "amira"],
        "JVC": ["gy-", "kh-"],
    }
    if camera.manufacturer:
        name_lower = camera.name.lower()
        for mfr, prefixes in _manufacturer_prefixes.items():
            if mfr != camera.manufacturer:
                for prefix in prefixes:
                    if name_lower.startswith(prefix):
                        issues.append(f"manufacturer_mismatch:{mfr}")
                        break

    # Suspiciously short name (likely partial)
    if len(camera.name) <= 2:
        issues.append("name_too_short")

    return issues


async def verify_camera(
    camera: Camera,
    issues: list[str],
    http_client: httpx.AsyncClient,
    anthropic_client: anthropic.AsyncAnthropic,
    searxng_url: str,
) -> VerificationResult:
    """Verify a camera against web sources and return corrections."""
    result = VerificationResult(
        camera_id=camera.id or 0,
        camera_name=camera.name,
        issues=issues,
    )

    # Build search query
    search_terms = camera.name
    if camera.manufacturer:
        search_terms = f"{camera.manufacturer} {camera.name}"
    search_query = f"{search_terms} camera specifications"

    try:
        response = await http_client.get(
            f"{searxng_url}/search",
            params={"q": search_query, "format": "json", "categories": "general"},
            headers={"User-Agent": "Mozilla/5.0 (compatible; CloragBot/1.0)"},
        )
        response.raise_for_status()
        search_data = response.json()
        raw_results = search_data.get("results", [])

        def clean_html(text: str) -> str:
            return re.sub(r"<[^>]+>", "", text).strip()

        search_results = [
            {
                "url": r.get("url", ""),
                "title": clean_html(r.get("title", "")),
                "snippet": clean_html(r.get("content", "")),
            }
            for r in raw_results[:8]
        ]

        if not search_results:
            logger.debug("No search results", camera=camera.name)
            return result

        # Fetch best pages via Jina
        manufacturer_domains = _get_manufacturer_domains(camera.manufacturer or "")
        best_urls = _pick_best_urls(search_results, manufacturer_domains, max_urls=2)

        settings = get_settings()
        jina_api_key = (
            settings.jina_api_key.get_secret_value() if settings.jina_api_key else None
        )

        fetched_pages: list[str] = []
        for url in best_urls:
            page_content = await _fetch_with_jina(http_client, url, jina_api_key)
            if page_content:
                fetched_pages.append(f"=== Page: {url} ===\n{page_content}")
            if not jina_api_key:
                await asyncio.sleep(3)

        # Build context
        if fetched_pages:
            context = "\n\n".join(fetched_pages)
            result.source = "jina_fetch"
        else:
            context = "\n---\n".join(
                f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}"
                for r in search_results
            )
            result.source = "snippet_only"

        # LLM arbitration
        current_data = json.dumps({
            "name": camera.name,
            "manufacturer": camera.manufacturer,
            "code_model": camera.code_model,
            "device_type": camera.device_type.value if camera.device_type else None,
        }, indent=2)

        prompt = f"""You are verifying camera database entries against web search results.

Current database entry:
{current_data}

Issues detected: {', '.join(issues)}

Web search results:
{context[:6000]}

Based on the web results, verify and correct the camera info. Return JSON:
{{
    "name": "correct model name (keep current if correct)",
    "manufacturer": "correct manufacturer name",
    "code_model": "official model code or null if unknown",
    "manufacturer_url": "official product page URL or null",
    "corrections_made": ["list of what was corrected and why"],
    "confidence": 0.0-1.0
}}

Only correct fields where the web evidence clearly contradicts the current data.
If unsure, keep the current value. Return ONLY the JSON object."""

        llm_response = await anthropic_client.messages.create(
            model=settings.sonnet_model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = llm_response.content[0].text.strip()
        if "```" in response_text:
            response_text = re.sub(r"```json?\s*", "", response_text)
            response_text = re.sub(r"```\s*", "", response_text)

        parsed = json.loads(response_text)
        confidence = parsed.get("confidence", 0.0)

        if confidence >= 0.7:
            # Apply corrections
            if parsed.get("manufacturer") and parsed["manufacturer"] != camera.manufacturer:
                result.corrections["manufacturer"] = parsed["manufacturer"]
            if parsed.get("code_model") and parsed["code_model"] != camera.code_model:
                result.corrections["code_model"] = parsed["code_model"]
            if parsed.get("name") and parsed["name"] != camera.name:
                result.corrections["name"] = parsed["name"]
            if parsed.get("manufacturer_url") and parsed["manufacturer_url"] != camera.manufacturer_url:
                result.corrections["manufacturer_url"] = parsed["manufacturer_url"]

            result.verified = True
            corrections_log = parsed.get("corrections_made", [])
            if corrections_log:
                logger.info(
                    "Corrections found",
                    camera=camera.name,
                    corrections=corrections_log,
                    confidence=confidence,
                )

        return result

    except httpx.HTTPError as e:
        logger.debug("HTTP error during verification", camera=camera.name, error=str(e))
        return result
    except (json.JSONDecodeError, anthropic.APIError) as e:
        logger.warning("Verification failed", camera=camera.name, error=str(e))
        return result


async def verify_cameras(
    manufacturer: str | None = None,
    limit: int | None = None,
    dry_run: bool = True,
    issues_only: bool = True,
) -> int:
    """Verify and correct cameras in the database.

    Args:
        manufacturer: Filter by manufacturer (None = all).
        limit: Max cameras to process.
        dry_run: If True, don't update DB.
        issues_only: If True, only verify cameras with detected issues.

    Returns:
        Number of cameras corrected.
    """
    settings = get_settings()
    db = get_camera_database()

    if manufacturer:
        cameras = db.list_cameras(manufacturer=manufacturer)
    else:
        cameras = db.list_cameras()

    all_cameras = cameras  # For cross-reference in issue detection

    # Detect issues
    cameras_to_verify: list[tuple[Camera, list[str]]] = []
    for camera in cameras:
        issues = detect_issues(camera, all_cameras)
        if issues or not issues_only:
            cameras_to_verify.append((camera, issues))

    if limit:
        cameras_to_verify = cameras_to_verify[:limit]

    logger.info(
        "Cameras to verify",
        total_cameras=len(cameras),
        with_issues=len(cameras_to_verify),
        limit=limit,
    )

    if not cameras_to_verify:
        logger.info("No cameras need verification")
        return 0

    http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    anthropic_client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value()
    )

    corrected = 0

    try:
        for camera, issues in cameras_to_verify:
            logger.info(
                "Verifying camera",
                camera=camera.name,
                manufacturer=camera.manufacturer,
                issues=issues,
            )

            result = await verify_camera(
                camera, issues, http_client, anthropic_client, settings.searxng_url
            )

            if result.corrections:
                corrected += 1
                if dry_run:
                    logger.info(
                        "[DRY RUN] Would correct camera",
                        camera=camera.name,
                        corrections=result.corrections,
                        source=result.source,
                    )
                else:
                    update = CameraUpdate(**result.corrections)
                    db.update_camera(camera.id, update)  # type: ignore[arg-type]
                    logger.info(
                        "Corrected camera",
                        camera=camera.name,
                        corrections=result.corrections,
                        source=result.source,
                    )
            else:
                logger.debug("No corrections needed", camera=camera.name)

            # Rate limit
            await asyncio.sleep(2)

    finally:
        await http_client.aclose()

    logger.info("Verification complete", verified=len(cameras_to_verify), corrected=corrected)
    return corrected


async def _main(
    manufacturer: str | None = None,
    limit: int | None = None,
    dry_run: bool = True,
    all_cameras: bool = False,
) -> None:
    """Async main entry point."""
    await verify_cameras(
        manufacturer=manufacturer,
        limit=limit,
        dry_run=dry_run,
        issues_only=not all_cameras,
    )


def cli() -> None:
    """Sync entry point for pyproject scripts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify and correct camera database entries against web sources"
    )
    parser.add_argument(
        "--manufacturer", "-m",
        help="Filter by manufacturer",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Maximum cameras to verify",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply corrections (default is dry-run)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_cameras",
        help="Verify all cameras, not just those with detected issues",
    )
    args = parser.parse_args()

    asyncio.run(_main(
        manufacturer=args.manufacturer,
        limit=args.limit,
        dry_run=not args.apply,
        all_cameras=args.all_cameras,
    ))


if __name__ == "__main__":
    cli()
