"""Legacy documentation management endpoints.

Provides scan, full and per-page re-ingestion of support.cyanview.com into the
separate docusaurus_docs_legacy collection. Completely independent from
the main CLORAG search collections.
"""

import re
from urllib.parse import urljoin

import httpx
import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from clorag.utils.url_validator import UrlNotAllowedError, validate_public_url
from clorag.web.auth import verify_admin
from clorag.web.dependencies import get_templates, limiter

router = APIRouter()
logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IngestPageRequest(BaseModel):
    """Request to ingest a single page by URL."""

    url: str = Field(..., min_length=10)


class IngestUrlsRequest(BaseModel):
    """Request to ingest multiple URLs."""

    urls: list[str] = Field(..., min_length=1, max_length=50)


class IngestResultEntry(BaseModel):
    """Result for a single ingested URL."""

    url: str
    status: str  # "success", "failed"
    chunks: int = 0
    error: str | None = None


class IngestResponse(BaseModel):
    """Response from ingestion."""

    results: list[IngestResultEntry]
    total_ingested: int
    total_chunks: int


class CollectionStats(BaseModel):
    """Stats for the legacy collection."""

    collection: str
    documents: int
    chunks: int


class ScanEntry(BaseModel):
    """A page found during scan."""

    url: str
    status: str  # "new", "indexed"


class ScanResponse(BaseModel):
    """Response from scanning the sitemap."""

    entries: list[ScanEntry]
    total_site: int
    total_indexed: int
    new_count: int


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------


@router.get("/legacy/manage", response_class=HTMLResponse)
async def legacy_manage_page(request: Request) -> Response:
    """Render the legacy management page."""
    templates = get_templates()
    return templates.TemplateResponse("legacy_manage.html", {"request": request})


@router.get("/legacy/help", response_class=HTMLResponse)
async def legacy_help_page(request: Request) -> Response:
    """Render the legacy help/documentation page."""
    templates = get_templates()
    return templates.TemplateResponse("legacy_help.html", {"request": request})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_indexed_urls(collection: str) -> set[str]:
    """Get all unique URLs currently in the legacy collection."""
    from clorag.core.vectorstore import VectorStore

    vs = VectorStore()
    urls: set[str] = set()
    offset: str | None = None

    while True:
        batch, next_offset = await vs.scroll_chunks(
            collection=collection, limit=100, offset=offset,
        )
        if not batch:
            break
        for chunk in batch:
            url = chunk.get("payload", {}).get("url")
            if url:
                urls.add(url)
        offset = next_offset
        if offset is None:
            break

    return urls


async def _fetch_sitemap(base_url: str) -> list[str]:
    """Fetch sitemap.xml and return filtered page URLs."""
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    excluded = ["/tags", "/search", "/download"]

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        response = await client.get(sitemap_url, timeout=30.0)
        response.raise_for_status()

    urls: list[str] = []
    for loc in re.findall(r"<loc>(.*?)</loc>", response.text):
        if not any(p in loc for p in excluded):
            urls.append(loc)

    return urls


# ---------------------------------------------------------------------------
# Stats API
# ---------------------------------------------------------------------------


@router.get("/api/legacy/stats", response_model=CollectionStats)
async def legacy_stats(
    request: Request, _: bool = Depends(verify_admin),
) -> CollectionStats:
    """Get stats for the legacy collection."""
    from clorag.config import get_settings
    from clorag.core.vectorstore import VectorStore

    settings = get_settings()
    legacy_collection = settings.qdrant_legacy_docs_collection
    vs = VectorStore()

    try:
        info = await vs.get_collection_info(legacy_collection)
        chunk_count = info.get("points_count", 0)
    except Exception:
        chunk_count = 0

    indexed_urls = await _get_indexed_urls(legacy_collection)

    return CollectionStats(
        collection=legacy_collection,
        documents=len(indexed_urls),
        chunks=chunk_count,
    )


# ---------------------------------------------------------------------------
# Scan for new pages
# ---------------------------------------------------------------------------


@router.post("/api/legacy/scan", response_model=ScanResponse)
@limiter.limit("10/minute")
async def legacy_scan(
    request: Request, _: bool = Depends(verify_admin),
) -> ScanResponse:
    """Scan sitemap and compare against indexed URLs.

    Returns list of all site pages with status: 'new' or 'indexed'.
    """
    from clorag.config import get_settings

    settings = get_settings()
    base_url = settings.docusaurus_url or "https://support.cyanview.com"
    legacy_collection = settings.qdrant_legacy_docs_collection

    site_urls = await _fetch_sitemap(base_url)
    indexed_urls = await _get_indexed_urls(legacy_collection)

    entries: list[ScanEntry] = []
    new_count = 0

    for url in sorted(site_urls):
        if url in indexed_urls:
            entries.append(ScanEntry(url=url, status="indexed"))
        else:
            entries.append(ScanEntry(url=url, status="new"))
            new_count += 1

    # Sort: new pages first
    entries.sort(key=lambda e: (0 if e.status == "new" else 1, e.url))

    return ScanResponse(
        entries=entries,
        total_site=len(site_urls),
        total_indexed=len(indexed_urls),
        new_count=new_count,
    )


# ---------------------------------------------------------------------------
# Ingest new pages (batch)
# ---------------------------------------------------------------------------


@router.post("/api/legacy/ingest-new", response_model=IngestResponse)
@limiter.limit("5/minute")
async def legacy_ingest_new(
    request: Request, req: IngestUrlsRequest,
    _: bool = Depends(verify_admin),
) -> IngestResponse:
    """Ingest selected new pages into the legacy collection."""
    from clorag.config import get_settings
    from clorag.core.vectorstore import VectorStore
    from clorag.ingestion.docusaurus import DocusaurusIngestionPipeline

    settings = get_settings()
    vs = VectorStore()
    legacy_collection = settings.qdrant_legacy_docs_collection

    vs._docs_collection = legacy_collection
    await vs._ensure_collection_hybrid(legacy_collection)

    pipeline = DocusaurusIngestionPipeline(
        base_url=settings.docusaurus_url,
        vector_store=vs,
        extract_cameras=False,
    )

    results: list[IngestResultEntry] = []
    total_chunks = 0

    for url in req.urls:
        try:
            validate_public_url(url)
        except UrlNotAllowedError as e:
            results.append(IngestResultEntry(
                url=url, status="failed", error=f"URL rejected: {e}",
            ))
            continue
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0),
            ) as client:
                doc = await pipeline._fetch_page(client, url)

            if doc is None:
                results.append(IngestResultEntry(
                    url=url, status="failed",
                    error="Page not found or empty",
                ))
                continue

            doc_chunks = await pipeline.process([doc])
            if not doc_chunks:
                results.append(IngestResultEntry(
                    url=url, status="failed",
                    error="No chunks produced",
                ))
                continue

            chunk_count = await pipeline.ingest(doc_chunks)
            total_chunks += chunk_count
            results.append(IngestResultEntry(
                url=url, status="success", chunks=chunk_count,
            ))

        except Exception as e:
            logger.error("Ingest new page failed", url=url, error=str(e))
            results.append(IngestResultEntry(
                url=url, status="failed", error=str(e),
            ))

    total_ingested = sum(1 for r in results if r.status == "success")
    return IngestResponse(
        results=results,
        total_ingested=total_ingested,
        total_chunks=total_chunks,
    )


# ---------------------------------------------------------------------------
# Full Re-ingest API
# ---------------------------------------------------------------------------


@router.post("/api/legacy/reingest-full", response_model=IngestResponse)
@limiter.limit("2/hour")
async def legacy_reingest_full(
    request: Request, _: bool = Depends(verify_admin),
) -> IngestResponse:
    """Full re-ingestion of support.cyanview.com into the legacy collection.

    Deletes all existing data and re-crawls the entire site.
    """
    from qdrant_client.http import models as qdrant_models

    from clorag.config import get_settings
    from clorag.core.vectorstore import VectorStore
    from clorag.ingestion.docusaurus import DocusaurusIngestionPipeline

    settings = get_settings()
    vs = VectorStore()
    legacy_collection = settings.qdrant_legacy_docs_collection

    logger.info("Starting full legacy re-ingestion", collection=legacy_collection)

    try:
        await vs._client.delete(
            collection_name=legacy_collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(must=[
                    qdrant_models.FieldCondition(
                        key="source",
                        match=qdrant_models.MatchValue(value="docusaurus"),
                    )
                ])
            ),
        )
    except Exception as e:
        logger.warning("Could not clear legacy collection", error=str(e))
        try:
            await vs.delete_collection(legacy_collection)
        except Exception:
            pass

    vs._docs_collection = legacy_collection
    await vs._ensure_collection_hybrid(legacy_collection)

    pipeline = DocusaurusIngestionPipeline(
        base_url=settings.docusaurus_url,
        vector_store=vs,
        extract_cameras=False,
    )

    try:
        documents = await pipeline.fetch()
        if not documents:
            return IngestResponse(
                results=[], total_ingested=0, total_chunks=0,
            )

        doc_chunks = await pipeline.process(documents)
        chunk_count = await pipeline.ingest(doc_chunks)

        results = [
            IngestResultEntry(
                url=doc.metadata.get("url", "unknown"),
                status="success",
                chunks=len(chunks),
            )
            for doc, chunks in doc_chunks
        ]

        logger.info(
            "Full legacy re-ingestion complete",
            documents=len(documents),
            chunks=chunk_count,
        )

        return IngestResponse(
            results=results,
            total_ingested=len(documents),
            total_chunks=chunk_count,
        )

    except Exception as e:
        logger.error("Full legacy re-ingestion failed", error=str(e))
        return IngestResponse(
            results=[IngestResultEntry(
                url="*", status="failed", error=str(e),
            )],
            total_ingested=0,
            total_chunks=0,
        )


# ---------------------------------------------------------------------------
# Single Page Ingest API
# ---------------------------------------------------------------------------


@router.post("/api/legacy/ingest-page", response_model=IngestResponse)
@limiter.limit("30/minute")
async def legacy_ingest_page(
    request: Request, req: IngestPageRequest,
    _: bool = Depends(verify_admin),
) -> IngestResponse:
    """Re-ingest a single page by URL into the legacy collection.

    Deletes existing chunks for this URL, fetches fresh content, re-embeds.
    """
    from qdrant_client.http import models as qdrant_models

    from clorag.config import get_settings
    from clorag.core.vectorstore import VectorStore
    from clorag.ingestion.docusaurus import DocusaurusIngestionPipeline

    settings = get_settings()
    vs = VectorStore()
    legacy_collection = settings.qdrant_legacy_docs_collection

    vs._docs_collection = legacy_collection
    await vs._ensure_collection_hybrid(legacy_collection)

    pipeline = DocusaurusIngestionPipeline(
        base_url=settings.docusaurus_url,
        vector_store=vs,
        extract_cameras=False,
    )

    url = req.url.strip()
    try:
        validate_public_url(url)
    except UrlNotAllowedError as e:
        return IngestResponse(
            results=[IngestResultEntry(
                url=url, status="failed", error=f"URL rejected: {e}",
            )],
            total_ingested=0,
            total_chunks=0,
        )
    logger.info("Single page legacy ingest", url=url)

    try:
        # Delete existing chunks for this URL
        await vs._client.delete(
            collection_name=legacy_collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(must=[
                    qdrant_models.FieldCondition(
                        key="url",
                        match=qdrant_models.MatchValue(value=url),
                    )
                ])
            ),
        )

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
        ) as client:
            doc = await pipeline._fetch_page(client, url)

        if doc is None:
            return IngestResponse(
                results=[IngestResultEntry(
                    url=url, status="failed",
                    error="Page not found or empty",
                )],
                total_ingested=0,
                total_chunks=0,
            )

        doc_chunks = await pipeline.process([doc])
        if not doc_chunks:
            return IngestResponse(
                results=[IngestResultEntry(
                    url=url, status="failed",
                    error="No chunks produced",
                )],
                total_ingested=0,
                total_chunks=0,
            )

        chunk_count = await pipeline.ingest(doc_chunks)

        return IngestResponse(
            results=[IngestResultEntry(
                url=url, status="success", chunks=chunk_count,
            )],
            total_ingested=1,
            total_chunks=chunk_count,
        )

    except Exception as e:
        logger.error("Single page ingest failed", url=url, error=str(e))
        return IngestResponse(
            results=[IngestResultEntry(
                url=url, status="failed", error=str(e),
            )],
            total_ingested=0,
            total_chunks=0,
        )
