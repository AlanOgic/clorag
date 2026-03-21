"""Ingestion and maintenance tools for CLORAG MCP server.

These tools trigger data ingestion pipelines and maintenance operations.
Most are long-running (minutes to hours) depending on data volume.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from clorag.mcp.server import MCPServices


def register_ingestion_tools(mcp: FastMCP[MCPServices]) -> None:
    """Register ingestion and maintenance MCP tools.

    Args:
        mcp: FastMCP server instance to register tools on.
    """

    @mcp.tool()
    async def ingest_docs(
        fresh: bool = False,
        extract_cameras: bool = True,
    ) -> dict[str, Any]:
        """Ingest Docusaurus documentation into the RAG knowledge base.

        Fetches all pages from the Docusaurus support site, extracts content
        using Jina Reader (with BeautifulSoup fallback), generates keywords
        via Sonnet, chunks text, generates hybrid embeddings, and stores in Qdrant.

        WARNING: This is a long-running operation (5-30 minutes depending on site size).

        Args:
            fresh: Delete existing docs collection before re-ingesting (full refresh).
            extract_cameras: Extract camera compatibility info from docs (default: True).

        Returns:
            Result with count of documents ingested.
        """
        from clorag.scripts.ingest_docs import run_ingestion

        start = time.monotonic()
        try:
            count = await run_ingestion(
                url=None,
                fresh=fresh,
                extract_cameras=extract_cameras,
            )
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "documents_ingested": count,
                "fresh": fresh,
                "cameras_extracted": extract_cameras,
                "duration_seconds": duration,
                "summary": f"Ingested {count} documentation pages in {duration}s.",
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}

    @mcp.tool()
    async def ingest_curated(
        max_threads: int | None = None,
        offset: int = 0,
        min_confidence: float = 0.7,
        fresh: bool = False,
        extract_cameras: bool = True,
        since_days: int | None = None,
    ) -> dict[str, Any]:
        """Ingest curated Gmail support threads into the RAG knowledge base.

        7-step pipeline: Fetch Gmail → Filter RMA → Anonymize → Sonnet Analysis →
        Quality Control → Chunk & Embed → Store in Qdrant + SQLite.

        WARNING: Long-running operation (10-60 minutes depending on thread count).

        Args:
            max_threads: Maximum number of Gmail threads to fetch (None for all).
            offset: Skip first N threads (for incremental ingestion).
            min_confidence: Minimum confidence for resolved classification (0.0-1.0, default 0.7).
            fresh: Delete existing cases collection before re-ingesting (with auto-snapshot).
            extract_cameras: Extract camera compatibility info from cases (default: True).
            since_days: Only fetch threads from the last N days (None for all).

        Returns:
            Result with count of cases ingested.
        """
        from clorag.scripts.ingest_curated import run_ingestion

        start = time.monotonic()
        try:
            count = await run_ingestion(
                max_threads=max_threads,
                offset=offset,
                min_confidence=min_confidence,
                fresh=fresh,
                extract_cameras=extract_cameras,
                since_days=since_days,
                snapshot=True,
            )
            duration = round(time.monotonic() - start, 1)
            summary = f"Ingested {count} support cases in {duration}s"
            if offset:
                summary += f" (offset {offset})"
            if since_days is not None:
                summary += f" (last {since_days} days)"
            summary += "."
            return {
                "status": "success",
                "cases_ingested": count,
                "max_threads": max_threads,
                "offset": offset,
                "min_confidence": min_confidence,
                "fresh": fresh,
                "since_days": since_days,
                "duration_seconds": duration,
                "summary": summary,
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}

    @mcp.tool()
    async def import_custom_documents(
        folder: str,
        category: str = "other",
        tags: str = "",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Bulk import custom documents (.txt, .md, .pdf) into the knowledge base.

        Recursively scans a folder for supported files, generates embeddings,
        and stores them as custom documents in Qdrant.

        Args:
            folder: Absolute path to folder containing documents to import.
            category: Document category. Valid values: product_info, troubleshooting,
                configuration, firmware, release_notes, faq, best_practices,
                pre_sales, internal, other.
            tags: Comma-separated tags to assign to all imported documents.
            dry_run: Preview what would be imported without actually importing.

        Returns:
            Result with count of imported and skipped documents.
        """
        from pathlib import Path

        from clorag.config import get_settings
        from clorag.models.custom_document import DocumentCategory
        from clorag.scripts.import_documents import import_documents

        settings = get_settings()
        base_dir = Path(settings.mcp_import_base_dir).resolve()
        folder_path = Path(folder).resolve()

        if not folder_path.is_relative_to(base_dir):
            return {
                "status": "error",
                "error": f"Path must be under {base_dir}. Got: {folder_path}",
            }
        if not folder_path.exists():
            return {"status": "error", "error": f"Folder not found: {folder_path}"}
        if not folder_path.is_dir():
            return {"status": "error", "error": f"Not a directory: {folder_path}"}

        try:
            doc_category = DocumentCategory(category)
        except ValueError:
            valid = [c.value for c in DocumentCategory]
            return {"status": "error", "error": f"Invalid category '{category}'. Valid: {valid}"}

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        start = time.monotonic()
        try:
            imported, skipped = await import_documents(
                folder=folder_path,
                category=doc_category,
                tags=tag_list,
                dry_run=dry_run,
            )
            duration = round(time.monotonic() - start, 1)
            action = "[DRY RUN] Would import" if dry_run else "Imported"
            return {
                "status": "success",
                "imported": imported,
                "skipped": skipped,
                "folder": str(folder_path),
                "category": category,
                "tags": tag_list,
                "dry_run": dry_run,
                "duration_seconds": duration,
                "summary": (
                    f"{action} {imported} documents, skipped {skipped} in {duration}s."
                ),
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}

    @mcp.tool()
    async def enrich_cameras(
        manufacturer: str | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        enrich_urls: bool = True,
    ) -> dict[str, Any]:
        """Enrich camera database with official model codes and manufacturer URLs.

        Pipeline: Known mappings → SearXNG search → Jina Reader fetch → LLM extraction.
        Uses Jina Reader to fetch actual page content (not just search snippets)
        for much more accurate model code and URL extraction.

        WARNING: Can be slow due to web search + Jina rate limiting.

        Args:
            manufacturer: Specific manufacturer to enrich (None for top 15 manufacturers).
            limit: Maximum cameras to process (None for all matching).
            dry_run: Preview enrichment results without updating the database.
            enrich_urls: Also enrich manufacturer product URLs (default: True).

        Returns:
            Result with count of enriched cameras.
        """
        from clorag.scripts.enrich_model_codes import enrich_cameras as _enrich

        manufacturers = [manufacturer] if manufacturer else None

        start = time.monotonic()
        try:
            count = await _enrich(
                manufacturers=manufacturers,
                limit=limit,
                dry_run=dry_run,
                enrich_urls=enrich_urls,
            )
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "cameras_enriched": count,
                "manufacturer": manufacturer,
                "limit": limit,
                "dry_run": dry_run,
                "duration_seconds": duration,
                "summary": f"Enriched {count} cameras in {duration}s.",
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}

    @mcp.tool()
    async def populate_graph(
        collections: list[str] | None = None,
        batch_size: int = 50,
        max_chunks: int | None = None,
        concurrency: int = 5,
    ) -> dict[str, Any]:
        """Populate Neo4j knowledge graph from Qdrant vector chunks.

        Extracts entities (Camera, Product, Protocol, Port, Issue, Solution,
        Firmware) and relationships using Sonnet, then stores them in Neo4j.

        Requires NEO4J_PASSWORD to be configured.

        WARNING: Long-running operation depending on chunk count and concurrency.

        Args:
            collections: Collections to process
                (default: all 3 collections).
            batch_size: Chunks to process per batch (default: 50).
            max_chunks: Maximum chunks per collection (None for all).
            concurrency: Number of concurrent LLM extractions (default: 5).

        Returns:
            Result with entity counts extracted.
        """
        from clorag.scripts.populate_graph import run_population

        target_collections = collections or [
            "docusaurus_docs",
            "gmail_cases",
            "custom_docs",
        ]

        start = time.monotonic()
        try:
            counts = await run_population(
                collections=target_collections,
                batch_size=batch_size,
                max_chunks=max_chunks,
                concurrency=concurrency,
            )
            duration = round(time.monotonic() - start, 1)
            formatted = (
                ", ".join(f"{k}: {v}" for k, v in counts.items())
                if isinstance(counts, dict)
                else str(counts)
            )
            n = len(target_collections)
            return {
                "status": "success",
                "collections": target_collections,
                "entity_counts": counts,
                "duration_seconds": duration,
                "summary": (
                    f"Extracted entities from {n} collections:"
                    f" {formatted} in {duration}s."
                ),
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}

    @mcp.tool()
    def rebuild_fts_index() -> dict[str, Any]:
        """Rebuild the camera FTS5 full-text search index.

        Drops and recreates the FTS5 virtual table, re-indexing all cameras
        from the SQLite database. Quick operation (seconds).

        Returns:
            Result with count of indexed cameras.
        """
        from clorag.core.database import get_camera_database

        start = time.monotonic()
        try:
            db = get_camera_database()
            count = db.rebuild_fts_index()
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "cameras_indexed": count,
                "duration_seconds": duration,
                "summary": f"Rebuilt FTS5 index for {count} cameras in {duration}s.",
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}

    @mcp.tool()
    async def fix_rio_preview(
        max_chunks: int | None = None,
    ) -> dict[str, Any]:
        """Scan vector database for RIO terminology issues and generate fix suggestions.

        Scans all 3 Qdrant collections for chunks mentioning RIO products,
        analyzes context with Sonnet to detect legacy terminology (RIO-Live,
        RIO Live, etc.), and saves suggestions to the terminology database.

        Suggestions must be reviewed and approved via the admin UI before applying.

        Args:
            max_chunks: Maximum chunks to scan (None for all). Useful for testing.

        Returns:
            Result with count of fix suggestions generated.
        """
        from clorag.analysis.rio_analyzer import RIOTerminologyAnalyzer
        from clorag.core.terminology_db import get_terminology_fix_database
        from clorag.core.vectorstore import VectorStore
        from clorag.scripts.fix_rio_terminology import scan_for_rio_mentions

        start = time.monotonic()
        try:
            vectorstore = VectorStore()
            analyzer = RIOTerminologyAnalyzer()

            fixes = await scan_for_rio_mentions(
                vectorstore=vectorstore,
                analyzer=analyzer,
                max_chunks=max_chunks,
            )

            # Save to database
            db = get_terminology_fix_database()
            saved = 0
            for fix in fixes:
                try:
                    db.save_fix(fix)
                    saved += 1
                except Exception:
                    pass  # Skip duplicates

            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "fixes_found": len(fixes),
                "fixes_saved": saved,
                "max_chunks": max_chunks,
                "message": (
                    "Fixes saved as suggestions. Review and"
                    " approve via /admin/terminology-fixes"
                    " before applying."
                ),
                "duration_seconds": duration,
                "summary": (
                    f"Found {len(fixes)} fixes, saved {saved} suggestions in {duration}s."
                ),
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}

    @mcp.tool()
    async def fix_rio_apply() -> dict[str, Any]:
        """Apply all approved RIO terminology fixes to vector database chunks.

        Groups approved fixes by document, fetches all sibling chunks,
        applies text corrections, and re-embeds the entire document with
        contextualized embeddings to preserve semantic understanding.

        Only applies fixes that have been approved via the admin UI.

        WARNING: Re-embeds affected documents (API cost). Only run after reviewing
        approved fixes at /admin/terminology-fixes.

        Returns:
            Result with count of fixes applied.
        """
        from clorag.core.embeddings import EmbeddingsClient
        from clorag.core.sparse_embeddings import SparseEmbeddingsClient
        from clorag.core.vectorstore import VectorStore
        from clorag.scripts.fix_rio_terminology import apply_approved_fixes

        start = time.monotonic()
        try:
            vectorstore = VectorStore()
            embeddings = EmbeddingsClient()
            sparse_embeddings = SparseEmbeddingsClient()

            count = await apply_approved_fixes(
                vectorstore=vectorstore,
                embeddings=embeddings,
                sparse_embeddings=sparse_embeddings,
            )
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "fixes_applied": count,
                "duration_seconds": duration,
                "summary": f"Applied {count} terminology fixes in {duration}s.",
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}

    @mcp.tool()
    def init_prompts_db(force: bool = False) -> dict[str, Any]:
        """Initialize the prompt database with default LLM prompts.

        Loads hardcoded default prompts into the SQLite database. By default,
        skips prompts that already exist (preserving admin customizations).

        Args:
            force: If True, reset ALL prompts to defaults (WARNING: loses customizations).

        Returns:
            Result with counts of created, updated, and skipped prompts.
        """
        from clorag.services.prompt_manager import get_prompt_manager

        start = time.monotonic()
        try:
            pm = get_prompt_manager()
            result = pm.initialize_defaults(force=force)
            duration = round(time.monotonic() - start, 1)
            return {
                "status": "success",
                "force": force,
                **result,
                "duration_seconds": duration,
                "summary": f"Initialized prompts in {duration}s.",
            }
        except Exception as e:
            duration = round(time.monotonic() - start, 1)
            return {"status": "error", "error": str(e), "duration_seconds": duration}
