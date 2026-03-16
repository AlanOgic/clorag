"""Ingestion job runner with log capture and SSE streaming.

Manages async execution of ingestion scripts, captures structlog output
per-job via contextvars, and provides pub-sub for real-time SSE streaming.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextvars import ContextVar
from datetime import datetime
from typing import Any

import structlog

from clorag.core.ingestion_db import IngestionDatabase, get_ingestion_database

logger = structlog.get_logger(__name__)

# Context variable to track which job is currently executing
_current_job_id: ContextVar[str | None] = ContextVar("_current_job_id", default=None)

# Singleton
_runner: IngestionJobRunner | None = None


def get_ingestion_runner() -> IngestionJobRunner:
    """Get or create the singleton IngestionJobRunner instance."""
    global _runner
    if _runner is None:
        _runner = IngestionJobRunner()
    return _runner


# =========================================================================
# Job Type Registry
# =========================================================================

JOB_TYPE_PARAM = dict[str, dict[str, Any]]


class JobTypeInfo:
    """Metadata about a registered job type."""

    def __init__(
        self,
        name: str,
        description: str,
        params: JOB_TYPE_PARAM,
    ) -> None:
        """Initialize job type info."""
        self.name = name
        self.description = description
        self.params = params

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
        }


def _p(
    typ: str, default: Any, desc: str,
) -> dict[str, Any]:
    """Build a parameter definition dict."""
    return {"type": typ, "default": default, "description": desc}


# Registry of all available job types
JOB_TYPES: dict[str, JobTypeInfo] = {
    "ingest_docs": JobTypeInfo(
        name="ingest_docs",
        description=(
            "Crawl the Cyanview Docusaurus support site via sitemap. "
            "Fetches pages with Jina Reader (BeautifulSoup fallback), "
            "chunks with token-based splitter, generates Voyage "
            "embeddings (dense + BM25 sparse), and upserts into the "
            "docusaurus_docs Qdrant collection. Also extracts 5-10 "
            "keywords per page via Sonnet and applies RIO terminology "
            "fixes before embedding."
        ),
        params={
            "fresh": _p(
                "bool", False,
                "Delete and recreate the docusaurus_docs collection "
                "before ingesting. Use for full re-index.",
            ),
            "extract_cameras": _p(
                "bool", True,
                "Run Sonnet camera extraction on new/updated chunks "
                "after ingestion to populate the camera database.",
            ),
        },
    ),
    "ingest_curated": JobTypeInfo(
        name="ingest_curated",
        description=(
            "Fetch curated Gmail support threads (label-filtered), "
            "run the 7-step pipeline: Fetch, Anonymize, Sonnet "
            "analysis (category/product/problem/solution), QC filter, "
            "chunk, embed (Voyage dense + BM25), and store in the "
            "gmail_cases Qdrant collection. Threads below the QC "
            "confidence threshold are skipped."
        ),
        params={
            "max_threads": _p(
                "int", 300,
                "Maximum number of Gmail threads to process in this "
                "run. Lower for faster test runs.",
            ),
            "offset": _p(
                "int", 0,
                "Skip the first N threads. Use for incremental "
                "ingestion after a previous partial run.",
            ),
            "min_confidence": _p(
                "float", 0.6,
                "Minimum Sonnet QC confidence score (0.0-1.0). "
                "Threads below this are skipped as low quality.",
            ),
            "fresh": _p(
                "bool", False,
                "Delete and recreate the gmail_cases collection "
                "before ingesting. Use for full re-index.",
            ),
            "extract_cameras": _p(
                "bool", True,
                "Run Sonnet camera extraction on new/updated chunks "
                "after ingestion to populate the camera database.",
            ),
        },
    ),
    "import_docs": JobTypeInfo(
        name="import_docs",
        description=(
            "Bulk import custom knowledge documents (.txt, .md, .pdf) "
            "from a server folder into the custom_docs Qdrant "
            "collection. Files are chunked, embedded, and enriched "
            "with Sonnet-generated keywords. The folder is restricted "
            "to IMPORT_DOCS_DIR for security."
        ),
        params={
            "folder": _p(
                "str", "",
                "Server folder path to import from. Must be within "
                "IMPORT_DOCS_DIR (/opt/clorag/import/ by default). "
                "Leave blank to use the default import directory.",
            ),
            "category": _p(
                "str", "other",
                "Document category: product_info, troubleshooting, "
                "configuration, firmware, release_notes, faq, "
                "best_practices, pre_sales, internal, or other.",
            ),
            "tags": _p(
                "str", "",
                "Comma-separated tags to apply to all imported "
                "documents (e.g. 'rio,networking,v2').",
            ),
            "dry_run": _p(
                "bool", False,
                "Preview what would be imported without actually "
                "creating documents or embeddings.",
            ),
        },
    ),
    "enrich_cameras": JobTypeInfo(
        name="enrich_cameras",
        description=(
            "Enrich existing camera database entries by looking up "
            "manufacturer product pages via web search. Adds official "
            "model codes (code_model), manufacturer URLs, and "
            "doc URLs where missing."
        ),
        params={
            "manufacturers": _p(
                "str", "",
                "Comma-separated manufacturer filter (e.g. "
                "'Sony,Canon'). Leave blank to process all.",
            ),
            "limit": _p(
                "int", 0,
                "Maximum cameras to process. 0 means all matching "
                "cameras. Use a small number for testing.",
            ),
            "dry_run": _p(
                "bool", False,
                "Preview enrichment results without updating the "
                "database.",
            ),
            "enrich_urls": _p(
                "bool", True,
                "Search for and add manufacturer product page URLs "
                "for cameras missing them.",
            ),
        },
    ),
    "extract_cameras": JobTypeInfo(
        name="extract_cameras",
        description=(
            "Scan vector store chunks with Sonnet to extract camera "
            "compatibility info (model name, manufacturer, ports, "
            "protocols, controls). Extracted cameras are validated, "
            "normalized, and upserted into the SQLite camera database."
        ),
        params={
            "docs_only": _p(
                "bool", False,
                "Only process chunks from the docusaurus_docs "
                "collection (documentation pages).",
            ),
            "cases_only": _p(
                "bool", False,
                "Only process chunks from the gmail_cases "
                "collection (support threads).",
            ),
            "limit": _p(
                "int", 1000,
                "Maximum number of chunks to process. Sonnet API "
                "calls are made per chunk, so lower values are "
                "faster and cheaper.",
            ),
        },
    ),
    "populate_graph": JobTypeInfo(
        name="populate_graph",
        description=(
            "Build the Neo4j knowledge graph by extracting entities "
            "(Camera, Product, Protocol, Port, Issue, Solution, "
            "Firmware) and relationships from vector store chunks "
            "using Sonnet. Requires Neo4j to be configured and "
            "reachable (see NEO4J_* env vars)."
        ),
        params={
            "collections": _p(
                "str", "docusaurus_docs,gmail_cases",
                "Comma-separated Qdrant collections to process. "
                "Options: docusaurus_docs, gmail_cases, custom_docs.",
            ),
            "batch_size": _p(
                "int", 10,
                "Number of chunks per Sonnet entity extraction "
                "batch. Higher values use more memory.",
            ),
            "max_chunks": _p(
                "int", 0,
                "Maximum total chunks to process across all "
                "collections. 0 means process everything.",
            ),
            "concurrency": _p(
                "int", 5,
                "Number of concurrent Sonnet extraction batches. "
                "Higher values are faster but use more API quota.",
            ),
        },
    ),
    "rebuild_fts": JobTypeInfo(
        name="rebuild_fts",
        description=(
            "Drop and rebuild the SQLite FTS5 full-text search index "
            "for the camera database. Run this after bulk camera "
            "imports or if search results seem stale. Fast operation, "
            "no external API calls."
        ),
        params={},
    ),
    "fix_rio_preview": JobTypeInfo(
        name="fix_rio_preview",
        description=(
            "Scan all vector store chunks for incorrect RIO product "
            "terminology (e.g. 'RIO-Live' should be 'RIO +LAN', "
            "generic 'RIO' in WAN context should be 'RIO +WAN'). "
            "Uses Sonnet to analyze context and suggest fixes. "
            "Results are saved to the terminology fixes database "
            "for review at /admin/terminology-fixes."
        ),
        params={
            "max_chunks": _p(
                "int", 0,
                "Maximum chunks to scan. 0 means scan all chunks "
                "across all collections. Use a smaller number for "
                "faster preview runs.",
            ),
        },
    ),
    "fix_rio_apply": JobTypeInfo(
        name="fix_rio_apply",
        description=(
            "Apply all 'approved' RIO terminology fixes from the "
            "fixes database. Updates chunk text in Qdrant and "
            "re-embeds affected chunks (all sibling chunks in the "
            "same document are re-embedded together for context). "
            "Only processes fixes with status='approved'."
        ),
        params={},
    ),
    "init_prompts": JobTypeInfo(
        name="init_prompts",
        description=(
            "Populate the prompts SQLite database with default LLM "
            "prompts from the hardcoded registry (11 prompts across "
            "agent, analysis, synthesis, drafts, graph, scripts "
            "categories). Skips prompts that already exist unless "
            "'force' is enabled."
        ),
        params={
            "force": _p(
                "bool", False,
                "Overwrite existing prompts with the hardcoded "
                "defaults. Warning: this discards any admin edits "
                "made via /admin/prompts.",
            ),
        },
    ),
}


# =========================================================================
# Structlog Processor for Log Capture
# =========================================================================


def ingestion_log_capture(
    logger_instance: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor that captures log entries for the current ingestion job.

    If a job is running in the current context, copies the log entry to the
    job's in-memory buffer and notifies SSE subscribers. Always passes through
    unchanged so stdout logging is unaffected.
    """
    job_id = _current_job_id.get(None)
    if job_id is not None:
        runner = _runner
        if runner is not None:
            # Build a log entry dict
            _skip = {
                "event", "timestamp", "_record",
                "_from_structlog", "logger", "level",
            }
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": method_name.upper(),
                "message": str(event_dict.get("event", "")),
                "event": event_dict.get("event"),
                "extra": {
                    k: _safe_serialize(v)
                    for k, v in event_dict.items()
                    if k not in _skip
                },
            }
            runner._capture_log(job_id, log_entry)

    return event_dict


def _safe_serialize(value: Any) -> Any:
    """Make a value JSON-safe for log extra data."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


# =========================================================================
# Job Runner
# =========================================================================


class IngestionJobRunner:
    """Manages ingestion job execution with log capture and SSE streaming.

    Features:
    - One job per job_type concurrency guard
    - Per-job log capture via contextvars + structlog processor
    - SSE pub-sub for real-time log streaming
    - Cancellation via asyncio.Task.cancel()
    """

    def __init__(self) -> None:
        """Initialize the job runner."""
        self._db: IngestionDatabase = get_ingestion_database()
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._job_id_by_type: dict[str, str] = {}
        self._log_buffers: dict[str, deque[dict[str, Any]]] = {}
        self._sse_queues: dict[str, list[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._log_batch_buffers: dict[str, list[dict[str, Any]]] = {}
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}

    async def startup(self) -> None:
        """Initialize on app startup."""
        stale_count = self._db.mark_stale_running_as_failed()
        cleaned = self._db.cleanup_old_jobs(days=30)
        if stale_count > 0 or cleaned > 0:
            logger.info(
                "Ingestion runner startup cleanup",
                stale_marked_failed=stale_count,
                old_jobs_cleaned=cleaned,
            )

    async def shutdown(self) -> None:
        """Gracefully cancel running tasks on shutdown."""
        for job_type, task in list(self._running_tasks.items()):
            if not task.done():
                task.cancel()
                logger.info("Cancelling running ingestion task", job_type=job_type)
        # Wait briefly for tasks to finish
        if self._running_tasks:
            tasks = [t for t in self._running_tasks.values() if not t.done()]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Flush remaining log batches
        for job_id in list(self._log_batch_buffers):
            await self._flush_logs(job_id)

    # =====================================================================
    # Job Execution
    # =====================================================================

    async def start_job(
        self,
        job_type: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Start an ingestion job.

        Args:
            job_type: Type of job from JOB_TYPES registry.
            parameters: Job-specific parameters.

        Returns:
            Job ID.

        Raises:
            ValueError: If job_type is unknown.
            RuntimeError: If a job of this type is already running.
        """
        if job_type not in JOB_TYPES:
            raise ValueError(f"Unknown job type: {job_type}")

        if job_type in self._running_tasks and not self._running_tasks[job_type].done():
            raise RuntimeError(f"Job type '{job_type}' is already running")

        # Create job record
        job = self._db.create_job(job_type, parameters)

        # Initialize log buffer
        self._log_buffers[job.id] = deque(maxlen=5000)
        self._log_batch_buffers[job.id] = []

        # Start async task
        task = asyncio.create_task(
            self._execute_job(job.id, job_type, parameters or {}),
            name=f"ingestion-{job_type}-{job.id[:8]}",
        )
        self._running_tasks[job_type] = task
        self._job_id_by_type[job_type] = job.id

        logger.info("Started ingestion job", job_id=job.id, job_type=job_type)
        return job.id

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: The job UUID.

        Returns:
            True if cancellation was initiated.
        """
        # Find the task by job_id
        for job_type, jid in self._job_id_by_type.items():
            if jid == job_id:
                task = self._running_tasks.get(job_type)
                if task and not task.done():
                    task.cancel()
                    logger.info("Cancellation requested", job_id=job_id)
                    return True
        return False

    def get_running_jobs(self) -> list[dict[str, Any]]:
        """Get info about currently running jobs.

        Returns:
            List of running job dicts with type and id.
        """
        result: list[dict[str, Any]] = []
        for job_type, task in self._running_tasks.items():
            if not task.done():
                job_id = self._job_id_by_type.get(job_type, "")
                job = self._db.get_job(job_id)
                if job:
                    result.append(job.to_dict())
        return result

    async def _execute_job(
        self,
        job_id: str,
        job_type: str,
        parameters: dict[str, Any],
    ) -> None:
        """Execute a job, capturing logs and updating status.

        This runs in an asyncio.Task with the job_id set in contextvars.
        """
        # Set context var for log capture
        token = _current_job_id.set(job_id)

        # Start flush task for periodic log persistence
        self._flush_tasks[job_id] = asyncio.create_task(
            self._periodic_flush(job_id),
            name=f"flush-{job_id[:8]}",
        )

        try:
            # Mark as running
            self._db.update_status(job_id, "running")
            self._broadcast_sse(job_id, {"type": "status", "status": "running"})

            start_time = time.monotonic()

            # Execute the actual job function
            result = await self._dispatch_job(job_type, parameters)

            elapsed = time.monotonic() - start_time

            # Mark as completed
            result_summary = result if isinstance(result, dict) else {"result": result}
            result_summary["elapsed_seconds"] = round(elapsed, 1)
            self._db.complete_job(job_id, result_summary)
            self._broadcast_sse(
                job_id,
                {"type": "status", "status": "completed", "result": result_summary},
            )
            logger.info(
                "Ingestion job completed",
                job_id=job_id,
                job_type=job_type,
                elapsed=round(elapsed, 1),
            )

        except asyncio.CancelledError:
            self._db.fail_job(job_id, "Job was cancelled by user")
            self._broadcast_sse(
                job_id, {"type": "status", "status": "cancelled"}
            )
            # Update status to cancelled
            with self._db._get_connection() as conn:
                conn.execute(
                    "UPDATE ingestion_jobs SET status = 'cancelled' WHERE id = ?",
                    (job_id,),
                )
                conn.commit()
            logger.info("Ingestion job cancelled", job_id=job_id, job_type=job_type)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            self._db.fail_job(job_id, error_msg)
            self._broadcast_sse(
                job_id, {"type": "status", "status": "failed", "error": error_msg}
            )
            logger.error(
                "Ingestion job failed",
                job_id=job_id,
                job_type=job_type,
                error=error_msg,
                exc_info=True,
            )

        finally:
            _current_job_id.reset(token)

            # Flush remaining logs
            await self._flush_logs(job_id)

            # Cancel flush task
            flush_task = self._flush_tasks.pop(job_id, None)
            if flush_task and not flush_task.done():
                flush_task.cancel()
                try:
                    await flush_task
                except asyncio.CancelledError:
                    pass

            # Send SSE end signal
            self._broadcast_sse(job_id, {"type": "end"})

            # Clean up running task reference
            if job_type in self._running_tasks:
                del self._running_tasks[job_type]
            if job_type in self._job_id_by_type:
                del self._job_id_by_type[job_type]

            # Schedule buffer cleanup (keep for 5 min for late SSE subscribers)
            asyncio.get_event_loop().call_later(
                300, self._cleanup_buffers, job_id
            )

    async def _dispatch_job(
        self,
        job_type: str,
        parameters: dict[str, Any],
    ) -> Any:
        """Dispatch to the actual job function.

        Args:
            job_type: Registered job type name.
            parameters: Job parameters.

        Returns:
            Job result (varies by type).
        """
        if job_type == "ingest_docs":
            from clorag.scripts.ingest_docs import (
                run_ingestion as run_docs_ingestion,
            )

            return await run_docs_ingestion(
                fresh=parameters.get("fresh", False),
                extract_cameras=parameters.get("extract_cameras", True),
            )

        elif job_type == "ingest_curated":
            from clorag.scripts.ingest_curated import (
                run_ingestion as run_curated_ingestion,
            )

            return await run_curated_ingestion(
                max_threads=parameters.get("max_threads", 300),
                offset=parameters.get("offset", 0),
                min_confidence=parameters.get("min_confidence", 0.6),
                fresh=parameters.get("fresh", False),
                extract_cameras=parameters.get("extract_cameras", True),
            )

        elif job_type == "import_docs":
            import os
            from pathlib import Path as PathLib

            from clorag.models.custom_document import DocumentCategory
            from clorag.scripts.import_documents import import_documents

            # Security: restrict folder to IMPORT_DOCS_DIR
            import_dir = os.environ.get("IMPORT_DOCS_DIR", "/opt/clorag/import/")
            folder_str = parameters.get("folder", "")
            if not folder_str:
                folder = PathLib(import_dir)
            else:
                folder = PathLib(folder_str).resolve()
                allowed = PathLib(import_dir).resolve()
                if not str(folder).startswith(str(allowed)):
                    raise ValueError(
                        f"Folder must be within {import_dir}. Got: {folder}"
                    )

            category_str = parameters.get("category", "other")
            category = DocumentCategory(category_str)
            tags_str = parameters.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []

            imported, skipped = await import_documents(
                folder=folder,
                category=category,
                tags=tags,
                dry_run=parameters.get("dry_run", False),
            )
            return {"imported": imported, "skipped": skipped}

        elif job_type == "enrich_cameras":
            from clorag.scripts.enrich_model_codes import enrich_cameras

            mfrs_str = parameters.get("manufacturers", "")
            manufacturers = (
                [m.strip() for m in mfrs_str.split(",") if m.strip()]
                if mfrs_str
                else None
            )
            limit = parameters.get("limit", 0)

            return await enrich_cameras(
                manufacturers=manufacturers,
                limit=limit if limit > 0 else None,
                dry_run=parameters.get("dry_run", False),
                enrich_urls=parameters.get("enrich_urls", True),
            )

        elif job_type == "extract_cameras":
            from clorag.scripts.extract_cameras import main as extract_main

            await extract_main(
                docs_only=parameters.get("docs_only", False),
                cases_only=parameters.get("cases_only", False),
                limit=parameters.get("limit", 1000),
            )
            return {"status": "completed"}

        elif job_type == "populate_graph":
            from clorag.scripts.populate_graph import run_population

            collections_str = parameters.get(
                "collections", "docusaurus_docs,gmail_cases"
            )
            collections = [c.strip() for c in collections_str.split(",") if c.strip()]
            max_chunks = parameters.get("max_chunks", 0)

            return await run_population(
                collections=collections,
                batch_size=parameters.get("batch_size", 10),
                max_chunks=max_chunks if max_chunks > 0 else None,
                concurrency=parameters.get("concurrency", 5),
            )

        elif job_type == "rebuild_fts":
            from clorag.core.database import get_camera_database

            db = get_camera_database()
            count = await asyncio.to_thread(db.rebuild_fts_index)
            return {"cameras_indexed": count}

        elif job_type == "fix_rio_preview":
            from clorag.scripts.fix_rio_terminology import run_preview

            max_chunks = parameters.get("max_chunks", 0)
            count = await run_preview(
                max_chunks=max_chunks if max_chunks > 0 else None
            )
            return {"suggestions_found": count}

        elif job_type == "fix_rio_apply":
            from clorag.scripts.fix_rio_terminology import run_apply

            count = await run_apply()
            return {"fixes_applied": count}

        elif job_type == "init_prompts":
            from clorag.scripts.init_prompts import initialize_prompts

            await asyncio.to_thread(
                initialize_prompts,
                force=parameters.get("force", False),
            )
            return {"status": "completed"}

        else:
            raise ValueError(f"No dispatcher for job type: {job_type}")

    # =====================================================================
    # Log Capture & Persistence
    # =====================================================================

    def _capture_log(self, job_id: str, log_entry: dict[str, Any]) -> None:
        """Capture a log entry for a job (called from structlog processor).

        This runs synchronously in the structlog processing pipeline.
        Appends to in-memory buffer and notifies SSE subscribers.
        """
        # Add to in-memory buffer
        buf = self._log_buffers.get(job_id)
        if buf is not None:
            buf.append(log_entry)

        # Add to batch buffer for periodic DB flush
        batch = self._log_batch_buffers.get(job_id)
        if batch is not None:
            batch.append({"job_id": job_id, **log_entry})

        # Notify SSE subscribers
        self._broadcast_sse(job_id, {"type": "log", **log_entry})

    async def _periodic_flush(self, job_id: str) -> None:
        """Periodically flush log batch buffer to SQLite."""
        try:
            while True:
                await asyncio.sleep(2.0)
                await self._flush_logs(job_id)
        except asyncio.CancelledError:
            pass

    async def _flush_logs(self, job_id: str) -> None:
        """Flush accumulated logs to the database."""
        batch = self._log_batch_buffers.get(job_id)
        if not batch:
            return

        # Swap buffer
        to_flush = batch[:]
        batch.clear()

        if to_flush:
            try:
                await asyncio.to_thread(self._db.insert_logs_batch, to_flush)
            except Exception:
                logger.debug("Failed to flush ingestion logs", job_id=job_id)

    def _cleanup_buffers(self, job_id: str) -> None:
        """Clean up in-memory buffers for a finished job."""
        self._log_buffers.pop(job_id, None)
        self._log_batch_buffers.pop(job_id, None)
        # Close any remaining SSE queues
        queues = self._sse_queues.pop(job_id, [])
        for q in queues:
            q.put_nowait(None)

    # =====================================================================
    # SSE Pub-Sub
    # =====================================================================

    def subscribe_sse(self, job_id: str) -> asyncio.Queue[dict[str, Any] | None]:
        """Subscribe to SSE events for a job.

        Returns an asyncio.Queue that receives log/status dicts.
        None signals end of stream.
        """
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=1000)

        if job_id not in self._sse_queues:
            self._sse_queues[job_id] = []
        self._sse_queues[job_id].append(queue)

        # Send buffered logs as replay
        buf = self._log_buffers.get(job_id)
        if buf:
            for entry in buf:
                try:
                    queue.put_nowait({"type": "log", **entry})
                except asyncio.QueueFull:
                    break

        return queue

    def unsubscribe_sse(
        self, job_id: str, queue: asyncio.Queue[dict[str, Any] | None]
    ) -> None:
        """Unsubscribe from SSE events for a job."""
        queues = self._sse_queues.get(job_id, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._sse_queues.pop(job_id, None)

    def _broadcast_sse(self, job_id: str, event: dict[str, Any]) -> None:
        """Broadcast an event to all SSE subscribers for a job."""
        queues = self._sse_queues.get(job_id, [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def get_buffered_logs(self, job_id: str) -> list[dict[str, Any]]:
        """Get in-memory buffered logs for a job.

        Returns:
            List of log entry dicts.
        """
        buf = self._log_buffers.get(job_id)
        if buf:
            return list(buf)
        return []
