#!/usr/bin/env python3
"""One-time backfill: reconstruct support_cases SQLite table from Qdrant chunks.

This script exists because the gmail_cases chunks were ingested before
the SQLite storage step (5.5) was added to the curated Gmail pipeline.
It reads chunk metadata from Qdrant and rebuilds SupportCase records.

This is a recovery tool, not part of the normal pipeline.
Future ingestions via `ingest-curated` store cases in SQLite automatically.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime

from clorag.config import get_settings
from clorag.core.support_case_db import get_support_case_database
from clorag.core.vectorstore import VectorStore
from clorag.models.support_case import CaseStatus, ResolutionQuality, SupportCase
from clorag.utils.logger import get_logger

logger = get_logger(__name__)


async def _scroll_all_chunks(vs: VectorStore, collection: str) -> list[dict]:
    """Scroll through all chunks in a Qdrant collection."""
    all_chunks: list[dict] = []
    offset: str | None = None

    while True:
        chunks, next_offset = await vs.scroll_chunks(
            collection=collection,
            limit=100,
            offset=offset,
        )
        if not chunks:
            break
        all_chunks.extend(chunks)
        offset = next_offset
        if offset is None:
            break

    return all_chunks


def _build_case_from_chunks(thread_id: str, chunks: list[dict]) -> SupportCase:
    """Reconstruct a SupportCase from its Qdrant chunks."""
    # Sort by chunk_index to reconstruct document in order
    sorted_chunks = sorted(chunks, key=lambda c: c["payload"].get("chunk_index", 0))

    # All chunks share the same case-level metadata; use first chunk
    meta = sorted_chunks[0]["payload"]

    # Reconstruct document by joining chunk texts
    document = "\n\n".join(
        c["payload"].get("text", "") for c in sorted_chunks if c["payload"].get("text")
    )

    # Use parent_case_id if available, otherwise generate
    case_id = meta.get("parent_case_id") or str(uuid.uuid4())

    # Parse resolution_quality
    rq_raw = meta.get("resolution_quality")
    resolution_quality: ResolutionQuality | None = None
    if rq_raw is not None:
        try:
            resolution_quality = ResolutionQuality(int(rq_raw))
        except (ValueError, TypeError):
            pass

    # Parse status
    status_raw = meta.get("status", "resolved")
    try:
        status = CaseStatus(status_raw)
    except ValueError:
        status = CaseStatus.RESOLVED

    # Parse created_at
    created_at: datetime | None = None
    if meta.get("created_at"):
        try:
            created_at = datetime.fromisoformat(meta["created_at"])
        except (ValueError, TypeError):
            pass

    # Parse resolved_at
    resolved_at: datetime | None = None
    if meta.get("resolved_at"):
        try:
            resolved_at = datetime.fromisoformat(meta["resolved_at"])
        except (ValueError, TypeError):
            pass

    # Keywords: list or JSON string
    keywords = meta.get("keywords", [])
    if isinstance(keywords, str):
        import json
        try:
            keywords = json.loads(keywords)
        except (json.JSONDecodeError, TypeError):
            keywords = []

    return SupportCase(
        id=case_id,
        thread_id=thread_id,
        subject=meta.get("subject", "Support Case"),
        status=status,
        resolution_quality=resolution_quality,
        problem_summary=meta.get("problem_summary", ""),
        solution_summary=meta.get("solution_summary", ""),
        keywords=keywords,
        category=meta.get("category", ""),
        product=meta.get("product"),
        document=document,
        messages_count=meta.get("messages_count", 0),
        created_at=created_at,
        resolved_at=resolved_at,
    )


async def backfill() -> None:
    """Backfill support_cases table from Qdrant gmail_cases chunks."""
    settings = get_settings()
    collection = settings.qdrant_cases_collection

    logger.info("Scrolling all chunks from Qdrant", collection=collection)
    vs = VectorStore()
    all_chunks = await _scroll_all_chunks(vs, collection)
    logger.info("Retrieved chunks", count=len(all_chunks))

    if not all_chunks:
        logger.warning("No chunks found in collection", collection=collection)
        return

    # Group by thread_id
    by_thread: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for chunk in all_chunks:
        tid = chunk["payload"].get("thread_id")
        if tid:
            by_thread[tid].append(chunk)
        else:
            skipped += 1

    if skipped:
        logger.warning("Chunks without thread_id skipped", count=skipped)

    logger.info("Unique threads found", count=len(by_thread))

    # Build and upsert cases
    db = get_support_case_database()
    created = 0
    for thread_id, chunks in by_thread.items():
        case = _build_case_from_chunks(thread_id, chunks)
        db.upsert_case(case)
        created += 1

    logger.info(
        "Backfill complete",
        cases_created=created,
        total_chunks=len(all_chunks),
    )
    print(f"Backfilled {created} support cases from {len(all_chunks)} chunks")


def main() -> None:
    """Entry point for backfill-support-cases CLI."""
    asyncio.run(backfill())


if __name__ == "__main__":
    main()
