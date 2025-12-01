#!/usr/bin/env python3
"""Extract cameras from already-ingested documents in Qdrant.

This script reads existing documents from the vector database and extracts
camera information without re-running the full ingestion pipeline.
"""

from __future__ import annotations

import asyncio

import structlog

from clorag.analysis.camera_extractor import CameraExtractor
from clorag.core.database import get_camera_database
from clorag.core.vectorstore import VectorStore
from clorag.models.camera import CameraSource

log = structlog.get_logger()


async def extract_from_collection(
    vectorstore: VectorStore,
    collection: str,
    source: CameraSource,
    extractor: CameraExtractor,
    limit: int = 1000,
) -> int:
    """Extract cameras from a Qdrant collection."""
    log.info("Fetching documents from collection", collection=collection, limit=limit)

    # Scroll through all points in the collection
    # Access the private _client attribute
    points, _ = await vectorstore._client.scroll(
        collection_name=collection,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        log.warning("No documents found in collection", collection=collection)
        return 0

    log.info("Found documents", count=len(points), collection=collection)

    # Prepare contents for batch extraction
    contents: list[tuple[str, str | None]] = []
    for point in points:
        payload = point.payload or {}
        text = payload.get("text", "")
        url = payload.get("url") if source == CameraSource.DOCUMENTATION else None

        if text and len(text) > 100:  # Skip very short chunks
            contents.append((text, url))

    if not contents:
        log.warning("No valid content found for extraction", collection=collection)
        return 0

    log.info("Extracting cameras from documents", doc_count=len(contents))

    # Extract cameras
    cameras = await extractor.extract_from_batch(contents, concurrency=5)

    if not cameras:
        log.info("No cameras found in collection", collection=collection)
        return 0

    # Store in database
    db = get_camera_database()
    for camera in cameras:
        db.upsert_camera(camera, source)

    log.info("Cameras extracted and stored", count=len(cameras), source=source.value)
    return len(cameras)


async def main(
    docs_only: bool = False,
    cases_only: bool = False,
    limit: int = 1000,
) -> None:
    """Extract cameras from Qdrant collections."""
    vectorstore = VectorStore()
    extractor = CameraExtractor()
    total = 0

    if not cases_only:
        # Extract from documentation
        try:
            count = await extract_from_collection(
                vectorstore,
                vectorstore.docs_collection,
                CameraSource.DOCUMENTATION,
                extractor,
                limit=limit,
            )
            total += count
        except Exception as e:
            log.error("Failed to extract from docs", error=str(e))

    if not docs_only:
        # Extract from support cases
        try:
            count = await extract_from_collection(
                vectorstore,
                vectorstore.cases_collection,
                CameraSource.SUPPORT_CASE,
                extractor,
                limit=limit,
            )
            total += count
        except Exception as e:
            log.error("Failed to extract from cases", error=str(e))

    # Print stats
    db = get_camera_database()
    stats = db.get_stats()
    log.info(
        "Extraction complete",
        total_extracted=total,
        total_cameras=stats["total_cameras"],
        manufacturers=stats["manufacturers"],
        by_source=stats["by_source"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract cameras from Qdrant")
    parser.add_argument("--docs-only", action="store_true", help="Only extract from docs")
    parser.add_argument("--cases-only", action="store_true", help="Only extract from cases")
    parser.add_argument("--limit", type=int, default=1000, help="Max documents to process")
    args = parser.parse_args()

    asyncio.run(main(
        docs_only=args.docs_only,
        cases_only=args.cases_only,
        limit=args.limit,
    ))
