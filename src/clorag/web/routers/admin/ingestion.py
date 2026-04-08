"""Admin ingestion management endpoints.

Provides job management, log retrieval, and SSE streaming
for ingestion processes.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from clorag.web.auth import verify_admin, verify_csrf
from clorag.web.dependencies import limiter


class IngestionJobRequest(BaseModel):
    """Request body for starting an ingestion job."""

    job_type: str = Field(..., description="Ingestion job type (e.g. 'ingest_docs')")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Job-specific parameters")

router = APIRouter(tags=["Ingestion"])
logger = structlog.get_logger(__name__)


def _get_runner() -> Any:
    """Lazy import of ingestion runner to avoid circular imports."""
    from clorag.services.ingestion_runner import get_ingestion_runner

    return get_ingestion_runner()


# =========================================================================
# Job Types & Status
# =========================================================================


@router.get("/ingestion/job-types")
async def api_ingestion_job_types(
    _: bool = Depends(verify_admin),
) -> list[dict[str, Any]]:
    """List available ingestion job types with parameter schemas."""
    from clorag.services.ingestion_runner import JOB_TYPES

    return [info.to_dict() for info in JOB_TYPES.values()]


@router.get("/ingestion/status")
async def api_ingestion_status(
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get current ingestion status (running jobs)."""
    runner = _get_runner()
    running = runner.get_running_jobs()
    return {
        "running_count": len(running),
        "running_jobs": running,
    }


# =========================================================================
# Job CRUD
# =========================================================================


@router.get("/ingestion/jobs")
async def api_ingestion_jobs_list(
    job_type: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """List ingestion jobs with optional filtering."""
    from clorag.core.ingestion_db import get_ingestion_database

    db = get_ingestion_database()

    # Clamp limit
    limit = min(limit, 200)

    jobs = db.list_jobs(
        job_type=job_type,
        status=status,
        limit=limit,
        offset=offset,
    )
    total = db.get_job_count(job_type=job_type, status=status)

    return {
        "jobs": [j.to_dict() for j in jobs],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/ingestion/jobs/{job_id}")
async def api_ingestion_job_get(
    job_id: str,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get a single job detail."""
    from clorag.core.ingestion_db import get_ingestion_database

    db = get_ingestion_database()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.post("/ingestion/jobs")
@limiter.limit("10/minute")
async def api_ingestion_job_start(
    request: Request,
    body: IngestionJobRequest,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Start a new ingestion job."""
    job_type = body.job_type
    parameters = body.parameters

    runner = _get_runner()
    try:
        job_id = await runner.start_job(job_type, parameters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"job_id": job_id, "status": "started"}


@router.post("/ingestion/jobs/{job_id}/cancel")
@limiter.limit("10/minute")
async def api_ingestion_job_cancel(
    request: Request,
    job_id: str,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Cancel a running ingestion job."""
    runner = _get_runner()
    cancelled = await runner.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=404, detail="Job not found or not running"
        )
    return {"status": "cancelling", "job_id": job_id}


@router.delete("/ingestion/jobs/{job_id}")
@limiter.limit("10/minute")
async def api_ingestion_job_delete(
    request: Request,
    job_id: str,
    _admin: bool = Depends(verify_admin),
    _csrf: bool = Depends(verify_csrf),
) -> dict[str, Any]:
    """Delete a job and its logs from history."""
    from clorag.core.ingestion_db import get_ingestion_database

    db = get_ingestion_database()
    deleted = db.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "deleted", "job_id": job_id}


# =========================================================================
# Logs
# =========================================================================


@router.get("/ingestion/jobs/{job_id}/logs")
async def api_ingestion_job_logs(
    job_id: str,
    limit: int = 500,
    offset: int = 0,
    level: str | None = None,
    _: bool = Depends(verify_admin),
) -> dict[str, Any]:
    """Get stored logs for a job (paginated)."""
    from clorag.core.ingestion_db import get_ingestion_database

    db = get_ingestion_database()

    # Verify job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    limit = min(limit, 2000)
    logs = db.get_logs(job_id, limit=limit, offset=offset, level=level)

    return {
        "logs": [log.to_dict() for log in logs],
        "limit": limit,
        "offset": offset,
        "job_status": job.status,
    }


@router.get("/ingestion/jobs/{job_id}/logs/stream")
async def api_ingestion_job_logs_stream(
    request: Request,
    job_id: str,
    _: bool = Depends(verify_admin),
) -> StreamingResponse:
    """SSE stream of real-time logs for a running job.

    Sends events as Server-Sent Events:
    - type: "log" — log entry with level, message, extra
    - type: "status" — job status change (completed, failed, cancelled)
    - type: "end" — stream is ending
    """
    from clorag.core.ingestion_db import get_ingestion_database

    db = get_ingestion_database()
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    runner = _get_runner()

    async def event_generator() -> AsyncGenerator[str, None]:
        queue = runner.subscribe_sse(job_id)
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                if event is None:
                    # End of stream
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    break

                event_type = event.get("type", "log")
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

                if event_type in ("end",):
                    break

        finally:
            runner.unsubscribe_sse(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
